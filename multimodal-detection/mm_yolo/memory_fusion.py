"""Spatial-memory v1: immutable modality evidence, local rejectable matching,
two residual refinement rounds, modality-owned and shared cross-scale memory.

No registration ground truth is assumed: matching is task-learned, not a claim
of calibrated physical alignment. Global memory supplements spatial features.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def masked_pool(x, valid, size):
    """Valid-count normalized pooling; padding/holes must not dilute distance/features."""
    fraction = F.adaptive_avg_pool2d(valid.float(), size)
    pooled = F.adaptive_avg_pool2d(x * valid.to(x.dtype), size)
    return pooled / fraction.clamp_min(1e-6), fraction


def safe_read(attention, query, values, valid):
    # Only all-invalid samples get a valid zero sentinel, avoiding MHA NaNs.
    sentinel = values.new_zeros(values.shape[0], 1, values.shape[-1])
    values = torch.cat((values, sentinel), 1)
    valid = torch.cat((valid, ~valid.any(1, keepdim=True)), 1)
    return attention(query, values, values, key_padding_mask=~valid, need_weights=False)[0]


class CrossScaleMemory(nn.Module):
    """2 RGB + 2 IR + 2 Depth + 2 shared tokens, reset for every image forward."""
    def __init__(self, channels, dim=64, heads=4, pool=4, tokens_per_modality=2):
        super().__init__()
        self.dim, self.pool, self.k = dim, pool, tokens_per_modality
        self.seed = nn.Parameter(torch.randn(1, 4, self.k, dim) * .02)
        self.identity = nn.Parameter(torch.randn(3, dim) * .02)
        self.scales = tuple(channels)
        self.scale = nn.Parameter(torch.randn(len(channels), dim) * .02)
        self.project = nn.ModuleDict({s: nn.Linear(c, dim) for s, c in channels.items()})
        self.position = nn.Linear(5, dim, bias=False)  # x,y,sin(pi*x),sin(pi*y),valid fraction
        self.read_norm = nn.LayerNorm(dim)
        self.own_read = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.exchange = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))

    def forward(self, scale, evidence, masks, state=None):
        ref = next(x for x in evidence if x is not None)
        b = ref.shape[0]
        if state is None:
            state = self.seed.expand(b, -1, -1, -1)
        axis = torch.linspace(-1, 1, self.pool, device=ref.device, dtype=ref.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        pos = torch.stack((xx, yy, (xx*math.pi).sin(), (yy*math.pi).sin()), -1)
        own, presence = [], []
        for m, (x, mask) in enumerate(zip(evidence, masks)):
            old = state[:, m]
            if x is None:
                own.append(old * 0)
                presence.append(torch.zeros(b, dtype=torch.bool, device=ref.device))
                continue
            pooled, fraction = masked_pool(x, mask, self.pool)
            observed = fraction.flatten(1) > 0
            xyf = torch.cat((pos[None].expand(b, -1, -1, -1), fraction.permute(0, 2, 3, 1)), -1)
            tokens = self.project[scale](pooled.flatten(2).transpose(1, 2))
            tokens = tokens + self.position(xyf.flatten(1, 2)) + self.identity[m] + self.scale[self.scales.index(scale)]
            update = safe_read(self.own_read, self.read_norm(old), self.read_norm(tokens), observed)
            current = old + update
            current = current + .1 * self.ffn(current)
            present = observed.any(1)
            own.append(current * present[:, None, None])
            presence.append(present)
        source = torch.cat(own, 1)
        present = torch.stack(presence, 1)
        shared = state[:, 3]
        shared = shared + safe_read(self.exchange, self.read_norm(shared), self.read_norm(source),
                                    present.repeat_interleave(self.k, 1))
        shared = shared + .1 * self.ffn(shared)
        return torch.stack((*own, shared), 1)


class SpatialMemoryFusion(nn.Module):
    def __init__(self, channels, window=3, dim=32, memory_dim=64, rounds=2, residual=.05,
                 memory_control="unbounded_v1"):
        super().__init__()
        if memory_control not in ("unbounded_v1", "bounded_v2"):
            raise ValueError(memory_control)
        self.memory_control = memory_control
        self.dim, self.window, self.rounds = dim, window, rounds
        self.embed = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False),
                                                 nn.GroupNorm(4, dim), nn.SiLU()) for _ in range(3)])
        self.identity = nn.Parameter(torch.randn(3, dim) * .02)
        self.query = nn.Sequential(nn.Conv2d(channels, dim, 1, bias=False), nn.GroupNorm(4, dim))
        self.memory_query = nn.Linear(memory_dim, dim, bias=False)
        self.key = nn.ModuleList([nn.Conv2d(dim, dim, 1, bias=False) for _ in range(3)])
        self.value = nn.ModuleList([nn.Conv2d(dim, dim, 1, bias=False) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Conv2d(dim*3+5, dim, 1), nn.SiLU(),
                                                nn.Conv2d(dim, 1, 1)) for _ in range(3)])
        self.out = nn.ModuleList([nn.Conv2d(dim, channels, 1, bias=False) for _ in range(3)])
        self.null_logit = nn.Parameter(torch.ones(3) * 1.0)
        self.offset_prior = nn.Parameter(torch.zeros(3, window*window))
        with torch.no_grad():
            self.offset_prior[:, window*window//2] = 2.0
        self.gain = nn.Parameter(torch.full((3,), math.log(residual/(1-residual))))
        self.last_stats = {}

    @staticmethod
    def _unit_rms(x):
        # Gate descriptors only: keep the value/injection path's actual magnitude.
        return x * torch.rsqrt(x.float().square().mean(1, keepdim=True) + 1e-5).to(x.dtype)

    def memory_context(self, memory):
        pooled = memory[:, 3].mean(1)
        if self.memory_control == "bounded_v2":
            pooled = F.layer_norm(pooled.float(), (pooled.shape[-1],)).to(memory.dtype)
        context = self.memory_query(pooled)[:, :, None, None]
        if self.memory_control == "bounded_v2":
            # Global memory guides; it must not drown out local normalized query.
            # Smooth RMS bound (<= .25) retains direction and nonzero derivatives.
            context = .25 * context / torch.sqrt(1 + context.float().square().mean(1, keepdim=True)).to(context.dtype)
        return context

    def _match(self, query, key, value, valid, modality):
        b, d, h, w = query.shape
        k = self.window**2
        keys = F.unfold(F.normalize(key.float(), dim=1), self.window, padding=self.window//2)
        keys = keys.view(b, d, k, h*w)
        q = F.normalize(query.float(), dim=1).flatten(2).unsqueeze(2)
        score = (q * keys).sum(1) * 6.0 + self.offset_prior[modality][None, :, None]
        candidate_valid = F.unfold(valid.float(), self.window, padding=self.window//2) > 0
        score = score.masked_fill(~candidate_valid, -1e4)
        null = self.null_logit[modality].expand(b, 1, h*w)
        weights = torch.softmax(torch.cat((score, null), 1), 1)[:, :k]
        weights = weights * candidate_valid
        values = F.unfold(value, self.window, padding=self.window//2).view(b, d, k, h*w)
        matched = (values * weights[:, None].to(values.dtype)).sum(2).view(b, d, h, w)
        confidence = weights.sum(1).view(b, 1, h, w)
        return matched, confidence

    def forward(self, evidence, masks, memory, quality=None):
        ref = next(x for x in evidence if x is not None)
        b, c, h, w = ref.shape
        # RGB is the coordinate anchor, not a claim that RGB is always observed.
        rgb_present = masks[0] if evidence[0] is not None else ref.new_zeros(b, 1, h, w)
        state = evidence[0] * rgb_present if evidence[0] is not None else torch.zeros_like(ref)
        other = sum(x * m for x, m in zip(evidence[1:], masks[1:]) if x is not None)
        denom = sum(m for x, m in zip(evidence[1:], masks[1:]) if x is not None)
        if torch.is_tensor(other):
            state = state + (1-rgb_present) * other / denom.clamp_min(1)
        embeddings = [None if x is None else (self.embed[m](x) + self.identity[m][None, :, None, None]) * masks[m]
                      for m, x in enumerate(evidence)]
        keys = [None if e is None else self.key[m](e) for m, e in enumerate(embeddings)]
        values = [None if e is None else self.value[m](e) * masks[m] for m, e in enumerate(embeddings)]
        context = self.memory_context(memory)
        stats = {}
        gate_health = {}
        for _ in range(self.rounds):
            q = self.query(state) + context
            residual = torch.zeros_like(state)
            for m, value in enumerate(values):
                if value is None:
                    continue
                if m == 0:
                    read, confidence = value, masks[m]
                else:
                    read, confidence = self._match(q, keys[m], value, masks[m], m)
                observed = masks[m]
                qual = None if quality is None else quality[m]
                qual = (ref.new_zeros(b, 3, h, w) if qual is None else
                        F.interpolate(qual, (h, w), mode="bilinear", align_corners=False).to(ref.dtype))
                q_gate, read_gate = q, read
                if self.memory_control == "bounded_v2":
                    q_gate, read_gate = self._unit_rms(q), self._unit_rms(read)
                logits = self.gates[m](torch.cat((q_gate, read_gate, q_gate-read_gate, confidence, observed, qual), 1))
                gate = (logits.float().sigmoid() if self.memory_control == "bounded_v2" else logits.sigmoid())
                residual = residual + self.gain[m].sigmoid() * self.out[m](read * gate)
                stats[str(m)] = torch.stack((confidence.detach().mean(), gate.detach().mean()))
                if self.memory_control == "bounded_v2":
                    gate_health[f"{m}_gate_saturated"] = (gate.detach() > .99).float().mean()
            state = state + residual / self.rounds
        self.last_health = {"context_rms": context.detach().float().square().mean().sqrt(),
                            "query_rms": q.detach().float().square().mean().sqrt(), **gate_health}
        self.last_stats = stats
        return state


class NeckMemoryRead(nn.Module):
    """One spatial read at the high-resolution neck, no global replacement of boxes."""
    def __init__(self, channels, dim=64, heads=4):
        super().__init__()
        self.query = nn.Conv2d(channels, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Conv2d(dim, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.tensor(math.log(.02/.98)))

    def forward(self, x, memory, present):
        b, _, h, w = x.shape
        tokens = memory.flatten(1, 2)
        valid = torch.cat((present.bool(), present.any(1, keepdim=True)), 1).repeat_interleave(memory.shape[2], 1)
        read = safe_read(self.attention, self.norm(self.query(x).flatten(2).transpose(1, 2)),
                         self.norm(tokens), valid)
        return x + self.gain.sigmoid() * self.out(read.transpose(1, 2).reshape(b, -1, h, w))
