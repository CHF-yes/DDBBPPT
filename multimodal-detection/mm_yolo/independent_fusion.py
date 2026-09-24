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


def affine_flow(params, size):
    """Convert a small canvas-normalized affine transform to feature flow.

    ``params`` is ``[angle_radians, shift_x/W, shift_y/H, scale_delta]``.
    The returned field follows :func:`warp`: every output point stores the
    source-sampling displacement in feature pixels.  Rotation is around the
    feature-map centre and follows OpenCV's image-coordinate convention.
    """
    b = params.shape[0]
    h, w = int(size[0]), int(size[1])
    angle, tx, ty, ds = params.float().unbind(1)
    scale = (1.0 + ds).clamp(.90, 1.10)
    ca, sa = angle.cos() * scale, angle.sin() * scale
    yy, xx = torch.meshgrid(
        torch.arange(h, device=params.device, dtype=torch.float32),
        torch.arange(w, device=params.device, dtype=torch.float32), indexing="ij")
    x = xx[None] - (w - 1) / 2.0
    y = yy[None] - (h - 1) / 2.0
    sx = ca[:, None, None] * x + sa[:, None, None] * y + tx[:, None, None] * w
    sy = -sa[:, None, None] * x + ca[:, None, None] * y + ty[:, None, None] * h
    return torch.stack((sx - x, sy - y), 1).reshape(b, 2, h, w)


class CoarseAffineAligner(nn.Module):
    """Predict a bounded per-image RGB/IR affine correction.

    The final layer is exactly zero initialized, so adding this module to a
    V4.4 checkpoint cannot move IR before it receives training signal.  Spatial
    pooling retains enough layout to estimate rotation/translation; a plain
    global vector cannot do that.
    """
    def __init__(self, dim):
        super().__init__()
        hidden = 32
        self.features = nn.Sequential(
            nn.Conv2d(dim * 4, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden * 4 * 8, 96), nn.SiLU(), nn.Linear(96, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.last_stats = {}

    def forward(self, rgb, ir, rgb_valid, ir_valid):
        mask = rgb_valid.float() * ir_valid.float()
        r = F.normalize(rgb.float(), dim=1) * mask
        i = F.normalize(ir.float(), dim=1) * mask
        x = torch.cat((r, i, r - i, r * i), 1)
        x = F.adaptive_avg_pool2d(self.features(x), (4, 8)).flatten(1)
        out = self.head(x)
        raw = out[:, :4].tanh()
        confidence = out[:, 4:5].sigmoid()
        accepted = raw * confidence
        self.last_stats = {
            "confidence": confidence.detach().mean(),
            # The affine field is built from the full geometric prediction.
            # Confidence is applied exactly once by identity_residual_align;
            # logging the accepted motion remains useful for health checks.
            "angle_norm": accepted[:, 0].detach().abs().mean(),
            "shift_norm": accepted[:, 1:3].detach().square().sum(1).sqrt().mean(),
            "scale_norm": accepted[:, 3].detach().abs().mean(),
        }
        return raw, confidence


class ConditionalIRAligner(nn.Module):
    """V5.1.1 A1 aligner conditioned on processing quality, not RGB texture.

    RGB/IR common embeddings provide geometry.  The explicit A0 descriptor may
    adjust the transform and its confidence, but it is never returned as visual
    evidence to the detector.
    """
    def __init__(self, dim, quality_channels=10):
        super().__init__()
        hidden = 32
        self.quality_channels = int(quality_channels)
        self.quality = nn.Sequential(
            nn.Conv2d(self.quality_channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())
        self.features = nn.Sequential(
            nn.Conv2d(dim * 5, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden * 4 * 8, 96), nn.SiLU(), nn.Linear(96, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.last_stats = {}

    def forward(self, rgb, ir, rgb_valid, ir_valid, quality=None):
        mask = rgb_valid.float() * ir_valid.float()
        r = F.normalize(rgb.float(), dim=1) * mask
        i = F.normalize(ir.float(), dim=1) * mask
        if quality is None:
            q = r.new_zeros(r.shape)
        else:
            quality = F.interpolate(quality.float(), r.shape[-2:], mode="bilinear",
                                    align_corners=False)
            if quality.shape[1] > self.quality_channels:
                quality = quality[:, :self.quality_channels]
            elif quality.shape[1] < self.quality_channels:
                quality = torch.cat((quality, quality.new_zeros(
                    quality.shape[0], self.quality_channels - quality.shape[1],
                    *quality.shape[-2:])), 1)
            q = self.quality(quality) * mask
        x = torch.cat((r, i, r - i, r * i, q), 1)
        x = F.adaptive_avg_pool2d(self.features(x), (4, 8)).flatten(1)
        out = self.head(x)
        raw, confidence = out[:, :4].tanh(), out[:, 4:5].sigmoid()
        accepted = raw * confidence
        self.last_stats = {
            "confidence": confidence.detach().mean(),
            "angle_norm": accepted[:, 0].detach().abs().mean(),
            "shift_norm": accepted[:, 1:3].detach().square().sum(1).sqrt().mean(),
            "scale_norm": accepted[:, 3].detach().abs().mean(),
        }
        return raw, confidence


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
        # A separate zero-centred switch gives Stage B an *exact* RGB identity
        # at initialization while retaining a healthy derivative at zero.  The
        # old sigmoid gain cannot do both: a very negative logit is almost zero
        # but also has an almost-zero gradient.  RGB remains the immutable
        # anchor (slot 0); only IR/Depth switches are released by the trainer.
        self.residual_scale = nn.Parameter(torch.zeros(3))
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
                coefficient = (self.residual_scale[m] * 0 if m == 0 else
                               .25 * self.residual_scale[m].tanh())
                update = update + coefficient * residual
                stats[str(m)] = torch.stack((match[m].detach().mean(), ((gc+gu)/2).detach().mean()))
            state = state + update / self.rounds
        self.last_stats = stats
        self.last_health = {"context_rms": context.detach().square().mean().sqrt()}
        for m in range(3):
            self.last_health[f"{m}_reliable"] = reliable[m].detach().float().mean()
            self.last_health[f"{m}_residual_scale"] = (.25 * self.residual_scale[m].detach().tanh())
        return state


class SpatialEvidenceRouter(nn.Module):
    """Route independently useful IR/Depth evidence into one RGB-anchored head.

    This is deliberately not detector voting.  Auxiliary modalities produce
    spatial evidence at every pyramid level, while one router and one YOLO head
    remain responsible for the final prediction.  The two output projections
    and the context projection start at zero, making the initial function an
    exact RGB identity without imposing the old global 0.25 residual ceiling.
    """
    def __init__(self, channels, dim=64, memory_dim=128, context_kernel=3):
        super().__init__()
        hidden = max(32, min(96, channels // 2))
        self.query = nn.Sequential(
            nn.Conv2d(channels, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.SiLU())
        self.memory = nn.Linear(memory_dim, dim, bias=False)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3 + 6, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv2d(hidden, channels, 1)) for _ in range(2)])
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, hidden, 1, bias=False), nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv2d(hidden, channels, 1, bias=False)) for _ in range(2)])
        for output in self.outputs:
            nn.init.zeros_(output[-1].weight)
        k = int(context_kernel)
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False),
            nn.GroupNorm(max(1, min(16, channels // 8)), channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False))
        nn.init.zeros_(self.context[-1].weight)
        self.last_stats, self.last_health = {}, {}

    def forward(self, raw, common, private, valid, match, reliable, memory,
                quality=None, anchor=None):
        state = raw[0] * valid[0] if anchor is None else anchor
        mem = self.memory(F.layer_norm(
            memory[:, 3].mean(1).float(), (memory.shape[-1],)))[:, :, None, None]
        q = self.query(state) + .25 * mem.to(state.dtype)
        updates, stats = [], {}
        for slot, m in enumerate((1, 2)):
            qual = (q.new_zeros(q.shape[0], 3, *q.shape[-2:])
                    if quality is None or quality[m] is None else quality[m])
            descriptor = torch.cat((q, common[m], private[m], valid[m],
                                    match[m], reliable[m], qual), 1)
            gate = self.gates[slot](descriptor).float().sigmoid().to(state.dtype)
            evidence = self.outputs[slot](torch.cat((common[m], private[m]), 1).to(state.dtype))
            update = evidence * gate * reliable[m].to(state.dtype) * valid[m].to(state.dtype)
            updates.append(update)
            stats[str(m)] = torch.stack((match[m].detach().mean(), gate.detach().mean()))
        context = self.context(state)
        state = state + context + sum(updates)
        base_rms = raw[0].detach().float().square().mean().sqrt().clamp_min(1e-6)
        self.last_stats = stats
        self.last_health = {
            "context_ratio": context.detach().float().square().mean().sqrt() / base_rms,
            "ir_route_ratio": updates[0].detach().float().square().mean().sqrt() / base_rms,
            "dep_route_ratio": updates[1].detach().float().square().mean().sqrt() / base_rms,
            "1_reliable": reliable[1].detach().float().mean(),
            "2_reliable": reliable[2].detach().float().mean(),
        }
        return state


class TrustedEvidenceRouter(nn.Module):
    """V4.7 target-evidence plugin around a frozen V4.4 feature.

    Unlike :class:`SpatialEvidenceRouter`, this block has no unconditional
    ``context(state)`` branch.  IR/Depth can modify the anchor only where their
    own training-supervised object evidence, validity and reliability agree.
    Output projections are zero initialized, so a migrated V4.4 checkpoint is
    an exact function-preserving starting point.
    """
    def __init__(self, channels, dim=64, memory_dim=128, max_gain=.20):
        super().__init__()
        hidden = max(32, min(96, channels // 2))
        self.max_gain = float(max_gain)
        self.query = nn.Sequential(
            nn.Conv2d(channels, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.SiLU())
        self.memory = nn.Linear(memory_dim, dim, bias=False)
        # Evidence is deliberately auxiliary-only: it is predicted from one
        # sensor's own common/private representation, not from RGB agreement.
        self.evidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2 + 2, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv2d(hidden, 1, 1)) for _ in range(2)])
        for head in self.evidence_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, -2.1972246)  # p(object)=0.10
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 3 + 7, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv2d(hidden, channels, 1)) for _ in range(2)])
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, hidden, 1, bias=False),
                nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv2d(hidden, channels, 1, bias=False)) for _ in range(2)])
        for output in self.outputs:
            nn.init.zeros_(output[-1].weight)
        # Per-modality bounded residual capacity.  The initial nominal limit is
        # 5%, but exact identity still comes from the zero output projections.
        initial_fraction = .05 / self.max_gain
        self.route_gain_logit = nn.Parameter(torch.full(
            (2,), math.log(initial_fraction / (1 - initial_fraction))))
        self.last_stats, self.last_health = {}, {}
        self.last_evidence_logits = []

    def forward(self, raw, common, private, valid, match, reliable, memory,
                quality=None, anchor=None):
        state = raw[0] * valid[0] if anchor is None else anchor
        mem = self.memory(F.layer_norm(
            memory[:, 3].mean(1).float(), (memory.shape[-1],)))[:, :, None, None]
        q = self.query(state) + .25 * mem.to(state.dtype)
        updates, stats, logits_all = [], {}, []
        base_rms = state.detach().float().square().mean(1, keepdim=True).sqrt().clamp_min(.1)
        gains = self.max_gain * self.route_gain_logit.sigmoid()
        for slot, m in enumerate((1, 2)):
            valid_m = valid[m].to(state.dtype)
            reliable_m = reliable[m].to(state.dtype)
            evidence_logits = self.evidence_heads[slot](torch.cat((
                common[m], private[m], valid_m, reliable_m), 1).to(state.dtype))
            evidence_prob = evidence_logits.float().sigmoid().to(state.dtype)
            logits_all.append(evidence_logits)
            qual = (q.new_zeros(q.shape[0], 3, *q.shape[-2:])
                    if quality is None or quality[m] is None else quality[m])
            descriptor = torch.cat((q, common[m], private[m], valid_m,
                                    match[m], reliable_m, qual, evidence_prob), 1)
            gate = self.gates[slot](descriptor).float().sigmoid().to(state.dtype)
            residual = self.outputs[slot](
                torch.cat((common[m], private[m]), 1).to(state.dtype))
            # Bound each local residual relative to the stable V4.4 anchor.  The
            # target evidence map then makes the global update spatially sparse.
            residual = residual / torch.sqrt(
                1 + residual.float().square().mean(1, keepdim=True) /
                base_rms.square()).to(residual.dtype)
            update = (gains[slot].to(state.dtype) * residual * gate *
                      evidence_prob * reliable_m * valid_m)
            updates.append(update)
            stats[str(m)] = torch.stack((
                match[m].detach().mean(), gate.detach().mean(),
                evidence_prob.detach().mean(), gains[slot].detach()))
        state = state + sum(updates)
        anchor_rms = anchor.detach().float().square().mean().sqrt().clamp_min(1e-6)
        self.last_evidence_logits = logits_all
        self.last_stats = stats
        self.last_health = {
            "ir_route_ratio": updates[0].detach().float().square().mean().sqrt() / anchor_rms,
            "dep_route_ratio": updates[1].detach().float().square().mean().sqrt() / anchor_rms,
            "ir_evidence": logits_all[0].detach().float().sigmoid().mean(),
            "dep_evidence": logits_all[1].detach().float().sigmoid().mean(),
            "ir_gain": gains[0].detach(), "dep_gain": gains[1].detach(),
            "1_reliable": reliable[1].detach().float().mean(),
            "2_reliable": reliable[2].detach().float().mean(),
        }
        return state


class EmbeddingComplementPlugin(nn.Module):
    """V4.8 additive RGB/IR complement plus Depth geometry support.

    The protected ``anchor`` is the complete learned V4.4 fused feature.  This
    module never replaces it.  IR common/private *spatial embeddings* generate
    two candidate residuals and a single softmax chooses their relative share;
    there is no V4.7-style product of evidence, gain, gate, match and reliability.

    Depth is intentionally asymmetric.  It sees the RGB/IR semantic embeddings
    and valid metric support, but its output is returned as a localization-only
    correction.  It cannot create class logits through an independent route.
    All final projections are zero initialized, making checkpoint migration an
    exact V4.4 function while retaining healthy gradients on the first step.
    """
    def __init__(self, channels, dim=64, memory_dim=128,
                 max_ir_gain=.20, max_depth_gain=.10):
        super().__init__()
        hidden = max(32, min(96, channels // 2))
        self.max_ir_gain = float(max_ir_gain)
        self.max_depth_gain = float(max_depth_gain)
        self.query = nn.Sequential(
            nn.Conv2d(channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())
        self.memory = nn.Linear(memory_dim, dim, bias=False)

        self.ir_shared = nn.Sequential(
            nn.Conv2d(dim * 5, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.ir_private = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.ir_mix = nn.Sequential(
            nn.Conv2d(dim * 4, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, 2, 1))

        # RGB query + RGB/IR/Depth common + Depth private + valid/reliability.
        self.depth_support = nn.Sequential(
            nn.Conv2d(dim * 5 + 2, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))

        for output in (self.ir_shared, self.ir_private, self.depth_support):
            nn.init.zeros_(output[-1].weight)
        nn.init.zeros_(self.ir_mix[-1].weight)
        nn.init.zeros_(self.ir_mix[-1].bias)

        ir_fraction = .05 / self.max_ir_gain
        dep_fraction = .03 / self.max_depth_gain
        self.ir_gain_logit = nn.Parameter(torch.tensor(
            math.log(ir_fraction / (1 - ir_fraction))))
        self.depth_gain_logit = nn.Parameter(torch.tensor(
            math.log(dep_fraction / (1 - dep_fraction))))
        self.last_stats, self.last_health = {}, {}

    @staticmethod
    def _bounded(residual, reference):
        scale = reference.detach().float().square().mean(
            1, keepdim=True).sqrt().clamp_min(.1)
        return residual / torch.sqrt(
            1 + residual.float().square().mean(1, keepdim=True) /
            scale.square()).to(residual.dtype)

    def forward(self, common, private, valid, reliable, memory, anchor):
        mem = self.memory(F.layer_norm(
            memory[:, 3].mean(1).float(), (memory.shape[-1],)))[:, :, None, None]
        q = self.query(anchor) + .25 * mem.to(anchor.dtype)
        rgb_c, ir_c, dep_c = common
        ir_u, dep_u = private[1], private[2]

        shared = self.ir_shared(torch.cat((
            q, rgb_c, ir_c, (rgb_c - ir_c).abs(), rgb_c * ir_c), 1))
        unique = self.ir_private(torch.cat((q, ir_c, ir_u), 1))
        mix = self.ir_mix(torch.cat((q, rgb_c, ir_c, ir_u), 1)).float().softmax(1)
        ir_delta = mix[:, :1].to(anchor.dtype) * shared + mix[:, 1:].to(anchor.dtype) * unique
        ir_delta = self._bounded(ir_delta, anchor) * valid[1].to(anchor.dtype)
        ir_gain = self.max_ir_gain * self.ir_gain_logit.sigmoid()
        fused = anchor + ir_gain.to(anchor.dtype) * ir_delta

        depth_descriptor = torch.cat((
            q, rgb_c, ir_c, dep_c, dep_u,
            valid[2].to(q.dtype), reliable[2].to(q.dtype)), 1)
        depth_delta = self.depth_support(depth_descriptor)
        depth_delta = self._bounded(depth_delta, anchor) * valid[2].to(anchor.dtype)
        depth_gain = self.max_depth_gain * self.depth_gain_logit.sigmoid()
        depth_delta = depth_gain.to(anchor.dtype) * depth_delta

        base_rms = anchor.detach().float().square().mean().sqrt().clamp_min(1e-6)
        self.last_stats = {
            "ir_shared_mix": mix[:, :1].detach().mean(),
            "ir_private_mix": mix[:, 1:].detach().mean(),
        }
        self.last_health = {
            "ir_route_ratio": (ir_gain * ir_delta).detach().float().square().mean().sqrt() / base_rms,
            "dep_geometry_ratio": depth_delta.detach().float().square().mean().sqrt() / base_rms,
            "ir_gain": ir_gain.detach(),
            "dep_gain": depth_gain.detach(),
            "1_reliable": reliable[1].detach().float().mean(),
            "2_reliable": reliable[2].detach().float().mean(),
        }
        return fused, depth_delta


class V5IRQualityFusion(nn.Module):
    """V5 additive evidence plugin around the learned V4.4 route.

    The V4.4 ``ComplementaryFusion`` remains the detector anchor.  This block
    only adds zero-initialized, locally gated residuals, so a V4.4 checkpoint
    keeps its initial function exactly.  IR common evidence and IR-private
    evidence have separate projections and gates; the latter never requires
    RGB agreement.  Depth is restricted to a valid-geometry support branch.
    """

    def __init__(self, channels, dim=64, memory_dim=128,
                 ir_quality_channels=9, depth_quality_channels=3):
        super().__init__()
        hidden = max(32, min(128, channels // 2))
        self.channels = int(channels)
        self.dim = int(dim)
        self.ir_quality_channels = int(ir_quality_channels)
        self.query = nn.Sequential(
            nn.Conv2d(channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())
        self.memory = nn.Linear(memory_dim, dim, bias=False)
        self.ir_quality = nn.Sequential(
            nn.Conv2d(self.ir_quality_channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())
        self.depth_quality = nn.Sequential(
            nn.Conv2d(depth_quality_channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())

        # Shared IR evidence uses RGB/IR agreement; private evidence is IR-only
        # apart from the current RGB query, so hot objects are not rejected.
        self.shared = nn.Sequential(
            nn.Conv2d(dim * 6, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.private = nn.Sequential(
            nn.Conv2d(dim * 4, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.depth_support = nn.Sequential(
            nn.Conv2d(dim * 6 + 2, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))

        # Every gate has both a spatial scalar and a per-channel component.
        # Zero initialization makes the added residual an exact no-op at handoff.
        gate_in = dim * 4 + 3  # query/common/private/quality + valid/match/reliability
        self.shared_spatial = self._gate(gate_in, hidden, 1)
        self.shared_channel = self._gate(gate_in, hidden, channels)
        self.private_spatial = self._gate(gate_in, hidden, 1)
        self.private_channel = self._gate(gate_in, hidden, channels)
        depth_gate_in = dim * 6 + 2
        self.depth_spatial = self._gate(depth_gate_in, hidden, 1)
        self.depth_channel = self._gate(depth_gate_in, hidden, channels)

        # Training-only evidence heads.  They are not detector voting; their
        # logits supervise whether a sensor has complete target evidence.
        self.ir_evidence = self._evidence_head(dim * 3, hidden)
        self.depth_evidence = self._evidence_head(dim * 3, hidden)
        self.last_stats, self.last_health = {}, {}
        self.last_evidence_logits = []

        for module in (self.shared, self.private, self.depth_support):
            nn.init.zeros_(module[-1].weight)

    @staticmethod
    def _gate(in_channels, hidden, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1))
        nn.init.zeros_(block[-1].weight)
        nn.init.zeros_(block[-1].bias)
        return block

    @staticmethod
    def _evidence_head(in_channels, hidden):
        block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(block[-1].weight)
        nn.init.constant_(block[-1].bias, -2.1972246)  # initial p(object)=0.10
        return block

    @staticmethod
    def _match_channels(value, channels):
        if value.shape[1] == channels:
            return value
        if value.shape[1] > channels:
            return value[:, :channels]
        return torch.cat((value, value.new_zeros(
            value.shape[0], channels - value.shape[1], *value.shape[-2:])), 1)

    @staticmethod
    def _bounded(residual, reference):
        # Per-location RMS stabilization prevents a bad sensor from exploding,
        # without the old global .20/.10 route ceiling.
        scale = reference.detach().float().square().mean(1, keepdim=True).sqrt().clamp_min(.1)
        return residual / torch.sqrt(
            1 + residual.float().square().mean(1, keepdim=True) /
            scale.square()).to(residual.dtype)

    def _quality(self, quality, index, ref, encoder):
        if quality is None or index >= len(quality) or quality[index] is None:
            return ref.new_zeros(ref.shape[0], self.dim, *ref.shape[-2:])
        value = F.interpolate(quality[index].float(), ref.shape[-2:],
                              mode="bilinear", align_corners=False)
        value = self._match_channels(value, encoder[0].in_channels)
        return encoder(value).to(ref.dtype)

    @staticmethod
    def _gate_pair(spatial, channel, descriptor):
        return spatial(descriptor).float().sigmoid().to(descriptor.dtype) * \
               channel(descriptor).float().sigmoid().to(descriptor.dtype)

    def forward(self, raw, common, private, valid, match, reliable, memory,
                quality=None, anchor=None):
        state = raw[0] * valid[0] if anchor is None else anchor
        mem = self.memory(F.layer_norm(
            memory[:, 3].mean(1).float(), (memory.shape[-1],)))[:, :, None, None]
        q = self.query(state) + .25 * mem.to(state.dtype)
        rgb_c, ir_c, dep_c = common
        ir_u, dep_u = private[1], private[2]
        ir_q = self._quality(quality, 1, q, self.ir_quality)
        dep_q = self._quality(quality, 2, q, self.depth_quality)

        ir_descriptor = torch.cat((q, ir_c, ir_u, ir_q,
                                   valid[1].to(q.dtype), match[1].to(q.dtype),
                                   reliable[1].to(q.dtype)), 1)
        shared_gate = self._gate_pair(self.shared_spatial, self.shared_channel,
                                      ir_descriptor)
        private_gate = self._gate_pair(self.private_spatial, self.private_channel,
                                       ir_descriptor)
        shared_input = torch.cat((q, rgb_c, ir_c, (rgb_c - ir_c).abs(),
                                  rgb_c * ir_c, ir_q), 1)
        private_input = torch.cat((q, ir_c, ir_u, ir_q), 1)
        shared_delta = self._bounded(self.shared(shared_input), state)
        private_delta = self._bounded(self.private(private_input), state)
        ir_evidence_logits = self.ir_evidence(torch.cat((q, ir_c, ir_u), 1))
        ir_evidence = ir_evidence_logits.float().sigmoid().to(state.dtype)
        ir_update = (shared_delta * shared_gate +
                     private_delta * private_gate * ir_evidence) * \
                    valid[1].to(state.dtype) * reliable[1].to(state.dtype)

        depth_input = torch.cat((q, rgb_c, ir_c, dep_c, dep_u, dep_q,
                                 valid[2].to(q.dtype), reliable[2].to(q.dtype)), 1)
        depth_descriptor = depth_input
        depth_gate = self._gate_pair(self.depth_spatial, self.depth_channel,
                                     depth_descriptor)
        depth_delta = self._bounded(self.depth_support(depth_input), state)
        depth_evidence_logits = self.depth_evidence(torch.cat((q, dep_c, dep_u), 1))
        depth_update = (depth_delta * depth_gate *
                        valid[2].to(state.dtype) * reliable[2].to(state.dtype))
        state = state + ir_update + depth_update

        reference = state if anchor is None else anchor
        base_rms = reference.detach().float().square().mean().sqrt().clamp_min(1e-6)
        self.last_evidence_logits = [ir_evidence_logits, depth_evidence_logits]
        self.last_stats = {
            "ir_shared_gate": shared_gate.detach().mean(),
            "ir_private_gate": private_gate.detach().mean(),
            "ir_evidence": ir_evidence.detach().mean(),
            "depth_gate": depth_gate.detach().mean(),
        }
        self.last_health = {
            "ir_route_ratio": ir_update.detach().float().square().mean().sqrt() / base_rms,
            "dep_route_ratio": depth_update.detach().float().square().mean().sqrt() / base_rms,
            "ir_evidence": ir_evidence.detach().float().mean(),
            "dep_evidence": depth_evidence_logits.detach().float().sigmoid().mean(),
            "ir_reliable": reliable[1].detach().float().mean(),
            "dep_reliable": reliable[2].detach().float().mean(),
        }
        return state


class V511TrustedIRFusion(nn.Module):
    """Conditional thermal fusion around the immutable V4.4 anchor.

    * aligned common IR is the only new localization evidence;
    * raw common/private IR supplies thermal-private classification evidence;
    * A0 processing descriptors control gates and never synthesize class content;
    * Depth is returned as geometry support and is not added to class features.
    """
    def __init__(self, channels, dim=64, memory_dim=128,
                 ir_quality_channels=10, depth_quality_channels=3):
        super().__init__()
        hidden = max(32, min(128, channels // 2))
        self.dim = int(dim)
        self.ir_quality_channels = int(ir_quality_channels)
        self.query = nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False),
                                   nn.GroupNorm(8, dim), nn.SiLU())
        self.memory = nn.Linear(memory_dim, dim, bias=False)
        self.ir_quality = nn.Sequential(
            nn.Conv2d(self.ir_quality_channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())
        self.depth_quality = nn.Sequential(
            nn.Conv2d(depth_quality_channels, dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.SiLU())
        self.shared = nn.Sequential(
            nn.Conv2d(dim * 5, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.private = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        self.depth_support = nn.Sequential(
            nn.Conv2d(dim * 5 + 2, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False))
        descriptor = dim * 5 + 3
        self.shared_spatial = self._gate(descriptor, hidden, 1)
        self.shared_channel = self._gate(descriptor, hidden, channels)
        self.private_spatial = self._gate(descriptor, hidden, 1)
        self.private_channel = self._gate(descriptor, hidden, channels)
        depth_descriptor = dim * 5 + 2
        self.depth_spatial = self._gate(depth_descriptor, hidden, 1)
        self.depth_channel = self._gate(depth_descriptor, hidden, channels)
        self.ir_evidence = self._evidence_head(dim * 3, hidden)
        self.last_stats, self.last_health, self.last_evidence_logits = {}, {}, []
        for module in (self.shared, self.private, self.depth_support):
            nn.init.zeros_(module[-1].weight)

    @staticmethod
    def _gate(in_channels, hidden, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1))
        nn.init.zeros_(block[-1].weight); nn.init.zeros_(block[-1].bias)
        return block

    @staticmethod
    def _evidence_head(in_channels, hidden):
        block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(max(1, min(16, hidden // 8)), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(block[-1].weight); nn.init.constant_(block[-1].bias, -2.1972246)
        return block

    @staticmethod
    def _bounded(residual, reference):
        scale = reference.detach().float().square().mean(1, keepdim=True).sqrt().clamp_min(.1)
        return residual / torch.sqrt(1 + residual.float().square().mean(
            1, keepdim=True) / scale.square()).to(residual.dtype)

    @staticmethod
    def _match_channels(value, channels):
        if value.shape[1] >= channels:
            return value[:, :channels]
        return torch.cat((value, value.new_zeros(
            value.shape[0], channels - value.shape[1], *value.shape[-2:])), 1)

    def _quality(self, quality, index, ref, encoder):
        if quality is None or index >= len(quality) or quality[index] is None:
            raw = ref.new_zeros(ref.shape[0], encoder[0].in_channels, *ref.shape[-2:])
        else:
            raw = F.interpolate(quality[index].float(), ref.shape[-2:], mode="bilinear",
                                align_corners=False)
            raw = self._match_channels(raw, encoder[0].in_channels)
        return raw, encoder(raw).to(ref.dtype)

    @staticmethod
    def _gate_pair(spatial, channel, descriptor):
        return spatial(descriptor).float().sigmoid().to(descriptor.dtype) * \
               channel(descriptor).float().sigmoid().to(descriptor.dtype)

    def forward(self, raw_features, raw_common, aligned_common, raw_private,
                valid, match, reliable, memory, quality=None, raw_quality=None,
                anchor=None):
        state = raw_features[0] * valid[0] if anchor is None else anchor
        mem = self.memory(F.layer_norm(
            memory[:, 3].mean(1).float(), (memory.shape[-1],)))[:, :, None, None]
        q = self.query(state) + .25 * mem.to(state.dtype)
        rgb_c, ir_raw_c, dep_c = raw_common
        ir_aligned_c = aligned_common[1]
        ir_u, dep_u = raw_private[1], raw_private[2]
        # Processing masks used by shared/aligned evidence live on the RGB
        # reference grid.  Thermal-private evidence stays on the untouched IR
        # grid and therefore must use the corresponding raw quality map.
        ir_q_aligned_raw, ir_q = self._quality(quality, 1, q, self.ir_quality)
        ir_q_raw, _ = self._quality(
            quality if raw_quality is None else raw_quality, 1, q, self.ir_quality)
        _, dep_q = self._quality(quality, 2, q, self.depth_quality)
        aligned_invalid = ir_q_aligned_raw[:, 3:4].clamp(0, 1)
        aligned_blur = ir_q_aligned_raw[:, 5:6].clamp(0, 1)
        aligned_ghost = ir_q_aligned_raw[:, 6:7].clamp(0, 1)
        aligned_leakage = ir_q_aligned_raw[:, 7:8].clamp(0, 1)
        registration = ir_q_aligned_raw[:, 9:10].clamp(0, 1)
        raw_invalid = ir_q_raw[:, 3:4].clamp(0, 1)
        raw_blur = ir_q_raw[:, 5:6].clamp(0, 1)
        raw_ghost = ir_q_raw[:, 6:7].clamp(0, 1)
        raw_leakage = ir_q_raw[:, 7:8].clamp(0, 1)
        thermal = ir_q_raw[:, 8:9].clamp(0, 1)
        aligned_trust = ((1 - aligned_invalid) * (1 - .4 * aligned_blur) *
                         (1 - .7 * aligned_ghost))
        raw_trust = (1 - raw_invalid) * (1 - .4 * raw_blur) * (1 - .7 * raw_ghost)

        descriptor = torch.cat((q, rgb_c, ir_aligned_c, ir_raw_c, ir_q,
                                valid[1].to(q.dtype), match[1].to(q.dtype),
                                reliable[1].to(q.dtype)), 1)
        shared_gate = self._gate_pair(self.shared_spatial, self.shared_channel, descriptor)
        private_gate = self._gate_pair(self.private_spatial, self.private_channel, descriptor)
        shared_delta = self._bounded(self.shared(torch.cat((
            q, rgb_c, ir_aligned_c, (rgb_c - ir_aligned_c).abs(),
            rgb_c * ir_aligned_c), 1)), state)
        private_delta = self._bounded(self.private(torch.cat((q, ir_raw_c, ir_u), 1)), state)
        evidence_logits = self.ir_evidence(torch.cat((q, ir_raw_c, ir_u), 1))
        evidence = evidence_logits.float().sigmoid().to(state.dtype)
        shared_weight = aligned_trust * registration * (1 - .5 * aligned_leakage)
        private_weight = raw_trust * thermal * (1 - raw_leakage)
        shared_update = shared_delta * shared_gate * shared_weight * valid[1]
        private_update = private_delta * private_gate * private_weight * evidence * valid[1]
        state = state + shared_update + private_update

        depth_descriptor = torch.cat((q, rgb_c, ir_raw_c, dep_c, dep_q,
                                      valid[2].to(q.dtype), reliable[2].to(q.dtype)), 1)
        depth_gate = self._gate_pair(self.depth_spatial, self.depth_channel, depth_descriptor)
        depth_delta = self._bounded(self.depth_support(torch.cat((
            q, rgb_c, ir_raw_c, dep_c, dep_u,
            valid[2].to(q.dtype), reliable[2].to(q.dtype)), 1)), state)
        depth_update = depth_delta * depth_gate * valid[2] * reliable[2]

        base_rms = anchor.detach().float().square().mean().sqrt().clamp_min(1e-6)
        self.last_evidence_logits = [evidence_logits]
        self.last_stats = {
            "ir_shared_gate": shared_gate.detach().mean(),
            "ir_private_gate": private_gate.detach().mean(),
            "ir_evidence": evidence.detach().mean(),
            "ir_leakage": raw_leakage.detach().mean(),
            "ir_registration": registration.detach().mean(),
            "depth_gate": depth_gate.detach().mean(),
        }
        self.last_health = {
            "ir_route_ratio": (shared_update + private_update).detach().float().square().mean().sqrt() / base_rms,
            "ir_shared_ratio": shared_update.detach().float().square().mean().sqrt() / base_rms,
            "ir_private_ratio": private_update.detach().float().square().mean().sqrt() / base_rms,
            "dep_geometry_ratio": depth_update.detach().float().square().mean().sqrt() / base_rms,
            "ir_reliable": reliable[1].detach().float().mean(),
            "dep_reliable": reliable[2].detach().float().mean(),
        }
        return state, shared_update, depth_update
