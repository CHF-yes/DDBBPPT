"""V4.2: identity-preserving spatial fusion with residual correspondence.

Distances are never interpolated here: grid_sample operates on learned features.
The three sensors already share a nominal image grid.  A low descriptor match is
therefore not evidence that an auxiliary modality is useless: it means that the
learned residual warp is uncertain.  Uncertain matches fall back to the identity
grid, while confident matches interpolate towards the warped feature.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def resize_flow(flow, size):
    h, w = flow.shape[-2:]
    out = F.interpolate(flow.float(), size, mode="bilinear", align_corners=False)
    return out * out.new_tensor([size[1] / w, size[0] / h])[None, :, None, None]


def warp(x, flow):
    """flow[:,0/1] is a SOURCE sampling displacement in feature pixels."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    grid = torch.stack(((xx[None] + flow[:, 0] + .5) * (2 / w) - 1,
                        (yy[None] + flow[:, 1] + .5) * (2 / h) - 1), -1)
    # CUDA grid_sample does not natively support bf16; retain differentiability.
    with torch.autocast(x.device.type, enabled=False):
        out = F.grid_sample(x.float(), grid.float(), align_corners=False, padding_mode="zeros")
    return out.to(x.dtype)


def identity_residual_align(x, flow, confidence, warped_valid=None):
    """Blend from the nominal sensor grid towards a residual warp.

    ``confidence=0`` is exactly the unwarped input and ``confidence=1`` is the
    warped input.  This is deliberately different from multiplying evidence by
    match confidence: an uncertain descriptor must not erase an already aligned
    IR/Depth observation.
    """
    confidence = confidence.float().clamp(0, 1)
    moved = warp(x, flow)
    if warped_valid is not None:
        moved = moved * warped_valid.to(moved.dtype)
    return x + confidence.to(x.dtype) * (moved - x)


class EvidenceEmbedding(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.common = nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False), nn.GroupNorm(8, dim))
        self.private = nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.SiLU())
        self.reconstruct = nn.Conv2d(2 * dim, channels, 1, bias=False)

    def forward(self, x, valid):
        c, u = self.common(x) * valid, self.private(x) * valid
        # Stop the backbone from minimizing the reconstruction target by shrinking.
        target = x.detach().float()
        scale = target.square().mean(1, keepdim=True).sqrt().clamp_min(.1)
        error = ((self.reconstruct(torch.cat((c, u), 1)).float() - target) / scale).square().mean(1, keepdim=True)
        loss = (error * valid).sum() / valid.sum().clamp_min(1)
        overlap = (F.normalize(c.float(),dim=1)*F.normalize(u.float(),dim=1)).sum(1,keepdim=True).square()
        loss = loss + .05*(overlap*valid).sum()/valid.sum().clamp_min(1)
        return c, u, loss


class LocalCorrespondence(nn.Module):
    def __init__(self, radius=1):
        super().__init__()
        self.radius = radius
        self.null = nn.Parameter(torch.tensor(0.0))
        offsets = [(x, y) for y in range(-radius, radius + 1) for x in range(-radius, radius + 1)]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32))

    def match(self, query, key, valid, flow, scene=None):
        b, d, h, w = query.shape
        k = len(self.offsets)
        key = warp(F.normalize(key.float(), dim=1), flow)
        valid = warp(valid.float(), flow)
        keys = F.unfold(key, 2*self.radius+1, padding=self.radius).view(b, d, k, h*w)
        mask = F.unfold(valid, 2*self.radius+1, padding=self.radius).view(b, k, h*w) > .99
        if scene is not None:
            ids = F.unfold(warp(scene, flow),2*self.radius+1,padding=self.radius).view(b,k,h*w)
            mask = mask & ((ids-scene.flatten(2)).abs()<.01)
        score = (F.normalize(query.float(), dim=1).flatten(2)[:, :, None] * keys).sum(1) * 6
        prior = -.35 * self.offsets.square().sum(1)
        score = (score + prior[None, :, None]).masked_fill(~mask, -1e4)
        p = torch.cat((score, self.null.expand(b, 1, h*w)), 1).softmax(1)[:, :k] * mask
        mass = p.sum(1, keepdim=True)
        cond = p / mass.clamp_min(1e-6)
        entropy = -(cond * cond.clamp_min(1e-8).log()).sum(1, keepdim=True) / math.log(k)
        delta = torch.einsum("bkn,kd->bdn", cond, self.offsets).view(b, 2, h, w)
        confidence = (mass * (1 - .75 * entropy)).view(b, 1, h, w)
        return flow + delta * mass.view(b, 1, h, w), confidence

    def forward(self, query, key, query_valid, valid, prior=None, scene=None):
        size = query.shape[-2:]
        base = query.new_zeros(query.shape[0], 2, *size, dtype=torch.float32) if prior is None else resize_flow(prior, size)
        flow, conf = self.match(query, key, valid, base, scene)
        # Reverse lookup provides a soft cycle reliability, not a hard RGB edge rule.
        backward, back_conf = self.match(key, query, query_valid, -base, scene)
        # Exact-zero cycles/all-invalid candidates are common in padding/Depth holes.
        # sqrt(0) has an infinite derivative; multiplying by a later zero mask does
        # not repair 0*inf NaNs in backward. Stabilize BEFORE the square roots.
        cycle = ((flow + warp(backward, flow)).square().sum(1, keepdim=True)+1e-6).sqrt()
        conf = conf * warp(back_conf, flow).clamp_min(1e-6).sqrt() * torch.exp(-cycle / 2) * query_valid
        return flow, conf


class ComplementaryFusion(nn.Module):
    def __init__(self, channels, dim=64, memory_dim=128, rounds=2):
        super().__init__()
        self.rounds = rounds
        self.query = nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False), nn.GroupNorm(8, dim))
        self.identity = nn.Parameter(torch.randn(3, dim) * .02)
        self.context = nn.Linear(memory_dim, dim, bias=False)
        self.gates = nn.ModuleList([nn.Sequential(nn.Conv2d(dim*3+6, dim, 1), nn.SiLU(), nn.Conv2d(dim, 2, 1)) for _ in range(3)])
        self.outputs = nn.ModuleList([nn.Conv2d(dim*2, channels, 1, bias=False) for _ in range(3)])
        self.gain = nn.Parameter(torch.full((3,), math.log(.05/.95)))
        self.last_stats, self.last_health = {}, {}

    def forward(self, raw, common, private, valid, match, reliable, memory, quality=None):
        rgb = raw[0] * valid[0]
        fallback = sum(x*m for x, m in zip(raw[1:], valid[1:])) / (valid[1]+valid[2]).clamp_min(1)
        state = rgb + (1-valid[0]) * fallback
        context = self.context(F.layer_norm(memory[:, 3].mean(1).float(), (memory.shape[-1],)))[:, :, None, None]
        context = .25 * context / torch.sqrt(1 + context.float().square().mean(1, keepdim=True))
        stats = {}
        for _ in range(self.rounds):
            q = self.query(state) + context.to(state.dtype)
            update = torch.zeros_like(state)
            for m in range(3):
                # Identity lives in the content/gate path, never in matching descriptors.
                u = private[m] + self.identity[m][None, :, None, None] * valid[m]
                qual = q.new_zeros(q.shape[0],3,*q.shape[-2:]) if quality is None or quality[m] is None else quality[m]
                g = self.gates[m](torch.cat((q, common[m], u, valid[m], match[m], reliable[m],qual), 1)).float().sigmoid()
                # Match confidence already chose how far to move the feature
                # towards its residual warp.  It must not gate evidence a second
                # time: the nominal sensor grids are valid fallbacks.
                gc = g[:, :1] * reliable[m] * valid[m]
                gu = g[:, 1:] * reliable[m] * valid[m]
                residual = self.outputs[m](torch.cat((common[m]*gc, u*gu), 1).to(state.dtype))
                # Keep a random branch from overwhelming the pretrained spatial signal.
                bound = state.detach().float().square().mean(1, keepdim=True).sqrt().clamp_min(.1)
                residual = residual / torch.sqrt(1 + residual.float().square().mean(1, keepdim=True)/bound.square()).to(residual.dtype)
                update = update + self.gain[m].sigmoid() * residual
                stats[str(m)] = torch.stack((match[m].detach().mean(), ((gc+gu)/2).detach().mean()))
            state = state + update / self.rounds
        self.last_stats = stats
        self.last_health = {"context_rms": context.detach().square().mean().sqrt()}
        for m in range(3):
            self.last_health[f"{m}_reliable"] = reliable[m].detach().float().mean()
        return state
