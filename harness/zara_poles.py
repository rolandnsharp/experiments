#!/usr/bin/env python3
"""Does a complex pole beat a real decay in Diakosmos's recurrence, on speech bytes?

    zara_poles.py train --variant real|cartesian|polar --bytes DIR --out DIR
        [--layers 4 --harmonics 32 --batch 16 --updates 3000 --eval-every 500]
    zara_poles.py check            # CPU: chunked scan == oracle's loop; gradients vs differences

Everything but the recurrence is Diakosmos's oracle (diakosmos_oracle.py beside
this file, scripts/alpha/diakosmos-oracle.py on main): the wire table, the
blocks, the features, the head, the optimizer and its schedule. The recurrence
per head h, channel l and head column d is

    S_t = c[h,l] S_{t-1} + k~_t[h,l] v_t[h,d],   read_t = Re( conj(q~_t[h,l]) S_t )  summed over l

with c real in (0, 1) for `real` (Diakosmos: sigmoid(g[h] + o[l]), q~ real),
c = a + i b with |c| <= 0.9995 kept by projection for `cartesian` (Manfred),
c = exp(-(gmin + e^rho)) (cos w, sin w) for `polar` (Ambrosio). The pole
variants give q~ an imaginary part (q_harmonic_weight is [H][2L][dh]).

The forward computes the recurrence in chunks of 64 positions as the causal
product A[t,s] = sum_l k_s[l] (q_re[t,l] Re c_l^(t-s) + q_im[t,l] Im c_l^(t-s))
read = A v, with the state carried between chunks; powers are closed-form
(exp(n log r), cos/sin(n theta)), so the gradient reaches r and theta directly.
`check` proves it equals the oracle's position loop for the real variant.

Scores: held-out next-byte loss in bits per byte (a subset while training, all
4,688 windows at the end), and free-running continuations from held-out seeds,
measured against the truth: RMS by 100 ms block, silent fraction, log-mel
distance. The Behold seed's continuation is written for listening only.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import mulaw  # noqa: E402

CHUNK = 64
# the matrix products as Diakosmos's device computes them (bfloat16 operands, binary32
# accumulation and results) when set; field_poles.py --half sets it
HALF = False
# the blocks' qkv and out projections ternary in the forward (BitNet b1.58) when set;
# field_poles.py --ternary sets it
TERNARY = False
# the blocks' qkv and out projection inputs rounded to int8 per position (BitNet b1.58's
# absmax activations) when set; field_poles.py --act8 sets it
ACT8 = False
POLE_SCALE = 1.0       # the polar pole's rho and omega are stored divided by this and multiplied back where used,
                       # so Adam (whose step does not depend on the gradient's scale) moves them this many times faster
GMIN = 0.0005          # Ambrosio's decay floor: |c| <= e^-gmin
RADIUS = 0.9995        # Manfred's projection radius
RATE = 16000
F_LOW, F_HIGH = 60.0, 7000.0
# the continuous-time variant: r = exp(-dt (1/TAU_MAX + 1/tau)), theta = 2 pi dt f
TAU_MAX = 4.0                     # the decay floor as a time: nothing rings longer than 4 s
TAU_LOW, TAU_HIGH = 1e-4, 2.0     # initial decay times, log-spaced over the bank
Q_INIT = 2.0                      # initial f = Q / (pi tau): a 2 s pole at 0.3 Hz, a 0.1 ms pole at 6.4 kHz


# ---- parameters ----
def make_params(c, variant, seed=0, dtype=torch.float32, mixing='dense', mlp=0, select=False,
                conv=0, gate=0, phase=False, erase=False, embed=False, vocab=256, pole_init='diakosmos'):
    """mlp F > 0: a SwiGLU MLP of hidden width F after every block (blocks.i.mlp.*); select: the polar
    pole's decay made input-dependent (temporal.attn.pole_decay_select, zero at the start). Both draw
    from their own generator after everything else, so without them the parameters are as before.
    conv K > 0: a causal depthwise convolution of width K added to q, k, v (temporal.attn.qkv_conv_kernel);
    gate R > 0: the read scaled by 2 sigmoid(u A' B') of rank R before the out projection
    (temporal.attn.out_gate_a, out_gate_b); phase: the polar pole's rotation made input-dependent
    (temporal.attn.pole_omega_select). Each is the identity at the start (the kernel, B and the
    selection zero) and draws from a third generator after the others. erase: before each write the state
    loses its component along the unit key, by beta = sigmoid(u w' + b) a head (temporal.attn.erase_gate_weight
    zero, erase_gate_bias -6, so beta starts at 0.0025); no draws. embed: the wire table replaced by a
    learned table of vocab rows (embed.harmonic_phasors, trained by Adam), each row the wire table's form at
    random phases and renormalized a group in the forward; vocab > 256 (tokens) needs it and gives the head
    vocab rows. Both draw from a fourth generator after the others."""
    p = O.initialize(c, seed, torch.float64)
    H, L, dh = c.heads, c.kan, c.head_width
    g = torch.Generator().manual_seed(seed + 7)
    if mixing == 'grouped':
        # per-head projections: head h sees only its own dh channels (fan-in dh, the oracle's bound rule)
        for i in range(c.layers):
            a = f'blocks.{i}.temporal.attn.'
            del p[a + 'qkv_proj.weight'], p[a + 'out_proj.weight']
            bound = 1 / math.sqrt(dh)
            p[a + 'qkv_group.weight'] = (torch.rand((H, 3 * dh, dh), generator=g, dtype=torch.float64) * 2 - 1) * bound
            p[a + 'out_group.weight'] = (torch.rand((H, dh, dh), generator=g, dtype=torch.float64) * 2 - 1) * bound
    elif mixing != 'dense':
        raise SystemExit(f'unknown mixing {mixing}')
    for i in range(c.layers):
        a = f'blocks.{i}.temporal.attn.'
        gamma = torch.sigmoid(p[a + '_gate_logit'][:, None] + p[a + '_harmonic_decay_offset'][None, :])
        if variant == 'real':
            continue
        del p[a + '_gate_logit'], p[a + '_harmonic_decay_offset']
        qim = torch.randn((H, L, dh), generator=g, dtype=torch.float64) * math.sqrt(2 / dh)
        p[a + 'q_harmonic_weight'] = torch.cat([p[a + 'q_harmonic_weight'], qim], dim=1)
        # the same decays as the real variant, frequencies log-spaced over the band, head-major
        omega = 2 * math.pi / RATE * torch.logspace(math.log10(F_LOW), math.log10(F_HIGH), H * L, dtype=torch.float64).view(H, L)
        if variant == 'polar':
            p[a + 'pole_decay_rho'] = torch.log(-torch.log(gamma) - GMIN)
            p[a + 'pole_omega'] = omega
            if pole_init == 'wide':
                # decay lengths log-spaced from 1 to 1500 bytes over the layer's poles, in a fixed shuffled order
                # (trained poles end mostly at 1-4 bytes with a few of hundreds; Diakosmos's start is 8-21 bytes)
                tau = torch.logspace(0, math.log10(1500), H * L, dtype=torch.float64)
                order = torch.randperm(H * L, generator=torch.Generator().manual_seed(17 + i))
                p[a + 'pole_decay_rho'] = torch.log(1 / tau[order] - GMIN).view(H, L)
            elif pole_init != 'diakosmos':
                raise SystemExit(f'unknown pole init {pole_init}')
            if POLE_SCALE != 1:
                p[a + 'pole_decay_rho'] = p[a + 'pole_decay_rho'] / POLE_SCALE
                p[a + 'pole_omega'] = p[a + 'pole_omega'] / POLE_SCALE
        elif variant == 'ct':
            # decay times log-spaced over the bank, harmonic-major so every head spans all timescales
            grid = torch.logspace(math.log10(TAU_LOW), math.log10(TAU_HIGH), H * L, dtype=torch.float64)
            tau = grid.view(L, H).t().contiguous()
            p[a + 'pole_decay_log_tau'] = torch.log(tau)
            p[a + 'pole_freq_log_hz'] = torch.log(Q_INIT / (math.pi * tau))
        elif variant == 'cartesian':
            p[a + 'pole_decay_a'] = gamma * torch.cos(omega)
            p[a + 'pole_decay_b'] = gamma * torch.sin(omega)
        else:
            raise SystemExit(f'unknown variant {variant}')
    if mlp or select:
        g2 = torch.Generator().manual_seed(seed + 11)
        D = c.d
        new = {}
        for i in range(c.layers):
            if mlp:
                m = f'blocks.{i}.mlp.'
                new[m + 'ln.weight'] = torch.ones(D, dtype=torch.float64)
                new[m + 'ln.bias'] = torch.zeros(D, dtype=torch.float64)
                for name, shape, fan in (('fc1.weight', (mlp, D), D), ('fc3.weight', (mlp, D), D), ('fc2.weight', (D, mlp), mlp)):
                    new[m + name] = (torch.rand(shape, generator=g2, dtype=torch.float64) * 2 - 1) / math.sqrt(fan)
            if select:
                if variant != 'polar':
                    raise SystemExit('--select needs the polar variant')
                new[f'blocks.{i}.temporal.attn.pole_decay_select'] = torch.zeros(H * L, D, dtype=torch.float64)
        p.update(O.centre_rotational(new))
    if erase:
        if variant != 'polar':
            raise SystemExit('--erase needs the polar variant')
        for i in range(c.layers):
            a = f'blocks.{i}.temporal.attn.'
            p[a + 'erase_gate_weight'] = torch.zeros(H, c.d, dtype=torch.float64)
            p[a + 'erase_gate_bias'] = torch.full((H,), -6.0, dtype=torch.float64)
    if conv or gate or phase:
        g3 = torch.Generator().manual_seed(seed + 13)
        D = c.d
        for i in range(c.layers):
            a = f'blocks.{i}.temporal.attn.'
            if conv:
                p[a + 'qkv_conv_kernel'] = torch.zeros(3 * D, conv, dtype=torch.float64)
            if gate:
                p[a + 'out_gate_a'] = (torch.rand((gate, D), generator=g3, dtype=torch.float64) * 2 - 1) / math.sqrt(D)
                p[a + 'out_gate_b'] = torch.zeros(D, gate, dtype=torch.float64)
            if phase:
                if variant != 'polar':
                    raise SystemExit('--phase needs the polar variant')
                p[a + 'pole_omega_select'] = torch.zeros(H * L, D, dtype=torch.float64)
    if vocab != 256 and not embed:
        raise SystemExit('a vocabulary other than bytes needs the learned embedding')
    if embed:
        g4 = torch.Generator().manual_seed(seed + 17)
        phases = torch.rand((vocab, O.GROUPS, c.harmonics), generator=g4, dtype=torch.float64) * 2 * math.pi
        z = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1) / math.sqrt(c.harmonics)
        p['embed.harmonic_phasors'] = z.reshape(vocab, c.d)
        if vocab != 256:
            bound = 1 / math.sqrt(c.features)
            head = {'head.weight': (torch.rand((vocab, c.features), generator=g4, dtype=torch.float64) * 2 - 1) * bound}
            p.update(O.centre_rotational(head))
    return {k: v.to(dtype) for k, v in p.items()}


def embed(c, p, table, x):
    """The block input for symbols x: the wire table's rows, or the learned table's renormalized a group."""
    if 'embed.harmonic_phasors' not in p:
        return table[x]
    e = p['embed.harmonic_phasors'][x]
    return O.normalize(e.view(*e.shape[:-1], O.GROUPS, c.width)).view(e.shape)


def polar_params(p, a):
    """The polar pole's (rho, omega), [H][L], undoing the storage scale POLE_SCALE."""
    rho, omega = p[a + 'pole_decay_rho'], p[a + 'pole_omega']
    if POLE_SCALE != 1:
        return rho * POLE_SCALE, omega * POLE_SCALE
    return rho, omega


def pole(p, a, variant):
    """(log r, theta) per [H][L]."""
    if variant == 'real':
        logit = p[a + '_gate_logit'][:, None] + p[a + '_harmonic_decay_offset'][None, :]
        return F.logsigmoid(logit), torch.zeros_like(logit)
    if variant == 'polar':
        rho, omega = polar_params(p, a)
        return -(GMIN + torch.exp(rho)), omega
    if variant == 'ct':
        logr = -(1.0 / TAU_MAX + torch.exp(-p[a + 'pole_decay_log_tau'])) / RATE
        return logr, 2 * math.pi / RATE * torch.exp(p[a + 'pole_freq_log_hz'])
    x, y = p[a + 'pole_decay_a'], p[a + 'pole_decay_b']
    return 0.5 * torch.log(x * x + y * y), torch.atan2(y, x)


def project(p, variant):
    if variant != 'cartesian':
        return
    with torch.no_grad():
        for name in list(p):
            if name.endswith('pole_decay_a'):
                a, b = p[name], p[name[:-1] + 'b']
                r = torch.sqrt(a * a + b * b)
                s = torch.clamp(RADIUS / (r + 1e-12), max=1.0)
                a.mul_(s)
                b.mul_(s)


# ---- the products ----
class FastHalfProduct(torch.autograd.Function):
    """O.HalfProduct with the same numbers on tensor cores: the operands are rounded to
    bfloat16 and held as binary32, so a TF32 product multiplies them exactly (TF32 keeps 10
    mantissa bits, bfloat16 has 7) and accumulates in binary32; TF32 is enabled for these
    products only, every other product stays binary32."""

    @staticmethod
    def forward(ctx, x, w):
        xb, wb = O.bf16(x), O.bf16(w)
        ctx.save_for_backward(xb, wb)
        with tf32():
            return xb @ wb.T

    @staticmethod
    def backward(ctx, g):
        xb, wb = ctx.saved_tensors
        gb = O.bf16(g)
        with tf32():
            dx = gb @ wb
            dw = gb.reshape(-1, gb.shape[-1]).T @ xb.reshape(-1, xb.shape[-1])
        return dx, dw


class tf32:
    def __enter__(self):
        self.before = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.before


def half_product(x, w):
    """x w': binary32 unless HALF, then as the device computes it."""
    if HALF:
        return FastHalfProduct.apply(x, w)
    return x @ w.T


def ternary(w):
    """BitNet b1.58: w's mean |w| times round(w / it) clipped to {-1, 0, +1}; the
    gradient passes straight through to the latent float weights."""
    s = w.abs().mean().clamp(min=1e-5)
    q = (w / s).round().clamp(-1, 1) * s
    return w + (q - w).detach()


def projection(p, name):
    """A block's projection weight as the forward uses it: ternary when TERNARY."""
    return ternary(p[name]) if TERNARY else p[name]


def act8(x):
    """BitNet b1.58's activations: x over each position's max |x| times 127, rounded and
    clipped to [-128, 127], scaled back; the gradient passes straight through."""
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    q = (x * (127 / s)).round().clamp(-128, 127) * (s / 127)
    return x + (q - x).detach()


def projection_input(x):
    """A block's projection input as the forward uses it: int8 per position when ACT8."""
    return act8(x) if ACT8 else x


# ---- the recurrence, chunked ----
def powers(logr, theta, n):
    """c^j for j = 0 .. n-1: ([H][L][n] real, imaginary)."""
    j = torch.arange(n, dtype=logr.dtype, device=logr.device)
    mag = torch.exp(logr[..., None] * j)
    ang = theta[..., None] * j
    return mag * torch.cos(ang), mag * torch.sin(ang)


def recurrence(qh, kh, v, logr, theta):
    """qh [B][T][H][2L or L], kh [B][T][H][L], v [B][T][H][dh] -> read [B][T][H][dh]."""
    B, T, H, L = kh.shape
    dh = v.shape[-1]
    C = CHUNK
    assert T % C == 0
    n = T // C
    pre, pim = powers(logr, theta, C + 1)                # [H][L][C+1]
    t = torch.arange(C, device=kh.device)
    lag = t[:, None] - t[None, :]                         # [C][C]
    causal = lag >= 0
    lagc = lag.clamp(min=0)
    Pre = pre[:, :, lagc] * causal                         # [H][L][C][C]
    Pim = pim[:, :, lagc] * causal
    q_re, q_im = qh[..., :L], (qh[..., L:] if qh.shape[-1] > L else None)
    qc = q_re.view(B, n, C, H, L)
    kc = kh.view(B, n, C, H, L)
    vc = v.view(B, n, C, H, dh)
    # within-chunk: A[b,n,h,t,s] = sum_l k[s,l] (q_re[t,l] Pre[l,t,s] + q_im[t,l] Pim[l,t,s])
    A = torch.einsum('bnthl,bnshl,hlts->bnhts', qc, kc, Pre)
    if q_im is not None:
        A = A + torch.einsum('bnthl,bnshl,hlts->bnhts', q_im.view(B, n, C, H, L), kc, Pim)
    read = torch.einsum('bnhts,bnshd->bnthd', A, vc)
    # the state carried between chunks: S_end = c^C S_prev + sum_s c^(C-1-s) k_s v_s
    end_re, end_im = pre[:, :, C - 1 - t], pim[:, :, C - 1 - t]      # [H][L][C]
    Sre = torch.zeros(B, H, L, dh, dtype=v.dtype, device=v.device)
    Sim = torch.zeros_like(Sre)
    step_re, step_im = pre[:, :, 1:C + 1], pim[:, :, 1:C + 1]          # c^(t+1), [H][L][C]
    cC_re, cC_im = pre[:, :, C], pim[:, :, C]
    carries = []
    for i in range(n):
        # read_t += Re( conj(q~_t) c^(t+1) S_prev ) = sum_l (q_re W_re + q_im W_im), W = c^(t+1) S_prev
        Wre = step_re[None, :, :, :, None] * Sre[:, :, :, None, :] - step_im[None, :, :, :, None] * Sim[:, :, :, None, :]
        Wim = step_re[None, :, :, :, None] * Sim[:, :, :, None, :] + step_im[None, :, :, :, None] * Sre[:, :, :, None, :]
        carry = torch.einsum('bthl,bhltd->bthd', qc[:, i], Wre)
        if q_im is not None:
            carry = carry + torch.einsum('bthl,bhltd->bthd', q_im.view(B, n, C, H, L)[:, i], Wim)
        carries.append(carry)
        inj_re = torch.einsum('bshl,hls,bshd->bhld', kc[:, i], end_re, vc[:, i])
        inj_im = torch.einsum('bshl,hls,bshd->bhld', kc[:, i], end_im, vc[:, i])
        Sre, Sim = (cC_re[None, :, :, None] * Sre - cC_im[None, :, :, None] * Sim + inj_re,
                    cC_re[None, :, :, None] * Sim + cC_im[None, :, :, None] * Sre + inj_im)
    read = read + torch.stack(carries, dim=1)
    return read.reshape(B, T, H, dh)


def recurrence_selective(qh, kh, v, la, theta):
    """recurrence() with a magnitude that varies along the sequence: c_t = exp(la_t) e^(i theta),
    la [B][T][H][L] <= 0, theta [H][L]. Products of the c_j are exp of sums of la within a chunk
    (G, a cumulative sum) times the rotation by theta per step; with la constant it is
    recurrence(qh, kh, v, la, theta) up to rounding."""
    B, T, H, L = kh.shape
    dh = v.shape[-1]
    C = CHUNK
    assert T % C == 0
    n = T // C
    G = la.reshape(B, n, C, H, L).permute(0, 1, 3, 4, 2).cumsum(-1)     # [B][n][H][L][C]: sum of la over 0..t
    t = torch.arange(C, device=kh.device)
    lag = t[:, None] - t[None, :]
    causal = lag >= 0
    if theta.dim() == 4:
        # a rotation per position, theta [B][T][H][L]: angles are differences of its cumulative sum
        P = theta.reshape(B, n, C, H, L).permute(0, 1, 3, 4, 2).cumsum(-1)
        ang = P[..., :, None] - P[..., None, :]                            # [B][n][H][L][t][s]
    else:
        ang = theta[:, :, None, None] * lag.clamp(min=0)                  # [H][L][C][C]
    cosL, sinL = torch.cos(ang) * causal, torch.sin(ang) * causal
    E = torch.exp((G[..., :, None] - G[..., None, :]).masked_fill(~causal, float('-inf')))   # [B][n][H][L][t][s]
    q_re, q_im = qh[..., :L], (qh[..., L:] if qh.shape[-1] > L else None)
    qc = q_re.view(B, n, C, H, L)
    kc = kh.view(B, n, C, H, L)
    vc = v.view(B, n, C, H, dh)
    A = torch.einsum('bnthl,bnshl,bnhlts->bnhts', qc, kc, E * cosL)
    if q_im is not None:
        A = A + torch.einsum('bnthl,bnshl,bnhlts->bnhts', q_im.view(B, n, C, H, L), kc, E * sinL)
    read = torch.einsum('bnhts,bnshd->bnthd', A, vc)
    # across chunks: c_0..c_t applied to the state at the chunk's start; the injection at s carried by
    # c_(s+1)..c_(C-1); the whole chunk's product
    if theta.dim() != 4:
        step_ang = theta[:, :, None] * (t + 1)                            # [H][L][C]
        end_ang = theta[:, :, None] * (C - 1 - t)
    Sre = torch.zeros(B, H, L, dh, dtype=v.dtype, device=v.device)
    Sim = torch.zeros_like(Sre)
    carries = []
    for i in range(n):
        Gi = G[:, i]                                                       # [B][H][L][C]
        if theta.dim() == 4:
            Pi = P[:, i]
            step_ang, end_ang, chunk_ang = Pi, Pi[..., -1:] - Pi, Pi[..., -1]
        else:
            chunk_ang = theta * C
        smag = torch.exp(Gi)
        step_re, step_im = smag * torch.cos(step_ang), smag * torch.sin(step_ang)
        emag = torch.exp(Gi[..., -1:] - Gi)
        end_re, end_im = emag * torch.cos(end_ang), emag * torch.sin(end_ang)
        cmag = torch.exp(Gi[..., -1])                                       # [B][H][L]
        cC_re, cC_im = cmag * torch.cos(chunk_ang), cmag * torch.sin(chunk_ang)
        Wre = step_re[..., None] * Sre[:, :, :, None, :] - step_im[..., None] * Sim[:, :, :, None, :]
        Wim = step_re[..., None] * Sim[:, :, :, None, :] + step_im[..., None] * Sre[:, :, :, None, :]
        carry = torch.einsum('bthl,bhltd->bthd', qc[:, i], Wre)
        if q_im is not None:
            carry = carry + torch.einsum('bthl,bhltd->bthd', q_im.view(B, n, C, H, L)[:, i], Wim)
        carries.append(carry)
        inj_re = torch.einsum('bshl,bhls,bshd->bhld', kc[:, i], end_re, vc[:, i])
        inj_im = torch.einsum('bshl,bhls,bshd->bhld', kc[:, i], end_im, vc[:, i])
        Sre, Sim = (cC_re[..., None] * Sre - cC_im[..., None] * Sim + inj_re,
                    cC_re[..., None] * Sim + cC_im[..., None] * Sre + inj_im)
    read = read + torch.stack(carries, dim=1)
    return read.reshape(B, T, H, dh)


def recurrence_erase(qh, kh, v, la, theta, beta):
    """The polar recurrence with an erase before each write: S_t = (I - beta_t k^_t k^_t') c_t S_(t-1) + k_t v_t',
    k^ the key over the harmonics made unit length, beta [B][T][H] in (0, 1); la [B][T][H][L], theta [H][L] or
    [B][T][H][L]. The erase is a second write, key k^_t and complex value e_t = -beta_t k^_t' c_t S_(t-1); within a
    chunk the e_t are the solution of a unit lower-triangular system built from pole powers forward in time only."""
    B, T, H, L = kh.shape
    dh = v.shape[-1]
    C = CHUNK
    assert T % C == 0
    n = T // C
    if theta.dim() != 4:
        theta = theta.expand(B, T, H, L)
    Z = torch.complex(la, theta).reshape(B, n, C, H, L).permute(0, 1, 3, 2, 4).cumsum(3)   # [B][n][H][C][L], sum of z over 0..t
    t = torch.arange(C, device=kh.device)
    causal = (t[:, None] - t[None, :]) >= 0
    F_ = torch.exp((Z[:, :, :, :, None, :] - Z[:, :, :, None, :, :]).masked_fill(~causal[..., None], float('-inf')))  # [B][n][H][t][s][L]
    q = torch.complex(qh[..., :L], qh[..., L:]).conj().reshape(B, n, C, H, L).transpose(2, 3)   # conj(q~), [B][n][H][C][L]
    k = kh.reshape(B, n, C, H, L).transpose(2, 3).to(F_.dtype)
    kn = kh / (kh.norm(dim=-1, keepdim=True) + 1e-6)
    kn = kn.reshape(B, n, C, H, L).transpose(2, 3).to(F_.dtype)
    vc = v.reshape(B, n, C, H, dh).transpose(2, 3).to(F_.dtype)                                # [B][n][H][C][dh]
    bt = beta.reshape(B, n, C, H).transpose(2, 3).to(F_.dtype)                                 # [B][n][H][C]
    strict = (t[:, None] - t[None, :]) > 0
    Mv = torch.einsum('bnhtl,bnhtsl,bnhsl->bnhts', kn, F_, k) * strict                          # k^_t' F[t,s] k_s, s < t
    Me = torch.einsum('bnhtl,bnhtsl,bnhsl->bnhts', kn, F_, kn) * strict
    A1 = torch.einsum('bnhtl,bnhtsl,bnhsl->bnhts', q, F_, k)                                    # conj(q~_t)' F[t,s] k_s, s <= t
    A2 = torch.einsum('bnhtl,bnhtsl,bnhsl->bnhts', q, F_, kn)
    eye = torch.eye(C, dtype=F_.dtype, device=kh.device)
    S = torch.zeros(B, H, L, dh, dtype=F_.dtype, device=kh.device)
    reads = []
    for i in range(n):
        Zi = Z[:, i]                                                       # [B][H][C][L]
        start = torch.exp(Zi)                                              # c_0..c_t applied to the chunk's start state
        carry_e = torch.einsum('bhtl,bhtl,bhld->bhtd', kn[:, i], start, S)
        rhs = -bt[:, i, :, :, None] * (Mv[:, i] @ vc[:, i] + carry_e)
        e = torch.linalg.solve_triangular(eye + bt[:, i, :, :, None] * Me[:, i], rhs, upper=False, unitriangular=True)
        read = A1[:, i] @ vc[:, i] + A2[:, i] @ e + torch.einsum('bhtl,bhtl,bhld->bhtd', q[:, i], start, S)
        reads.append(read.real)
        end = torch.exp(Zi[:, :, -1:, :] - Zi)                             # [B][H][C][L], c_(s+1)..c_(C-1)
        S = (torch.exp(Zi[:, :, -1, :])[..., None] * S
             + torch.einsum('bhsl,bhsl,bhsd->bhld', end, k[:, i], vc[:, i])
             + torch.einsum('bhsl,bhsl,bhsd->bhld', end, kn[:, i], e))
    return torch.stack(reads, 1).transpose(2, 3).reshape(B, T, H, dh).to(v.dtype)


def erase_strength(p, a, u):
    """beta = sigmoid(u w' + b), one a head: [B][T][H] (or [H] for one position)."""
    return torch.sigmoid(u @ p[a + 'erase_gate_weight'].T + p[a + 'erase_gate_bias'])


SELECT_CAP = 4.0       # rho + u Ws' is capped here: |c| >= e^-(gmin + e^4) ~ e^-55, already nothing, and
                       # e^(rho + s) cannot overflow (an infinite log-magnitude makes the chunk sums inf - inf)


def selective_logmag(p, a, u):
    """The input-dependent pole magnitude's log, -(gmin + e^min(rho + u Ws', cap)), [B][T][H][L] (or [H][L] for one position)."""
    rho = polar_params(p, a)[0]
    s = (u @ p[a + 'pole_decay_select'].T).view(*u.shape[:-1], *rho.shape)
    return -(GMIN + torch.exp((rho + s).clamp(max=SELECT_CAP)))


def selective_theta(p, a, u):
    """The input-dependent pole rotation, omega + u Wo', [B][T][H][L] (or [H][L] for one position)."""
    omega = polar_params(p, a)[1]
    return omega + (u @ p[a + 'pole_omega_select'].T).view(*u.shape[:-1], *omega.shape)


def qkv_conv(p, a, qkv, history=None):
    """qkv plus its causal depthwise convolution, sum_j kernel[:, j] qkv[t - j], over the last dimension's
    channels; qkv [B][T][3D] (zero before the window), or one position [3D] with history the K - 1
    positions before it, newest last."""
    w = p[a + 'qkv_conv_kernel']
    K = w.shape[1]
    if history is not None:
        x = torch.stack(history + [qkv])                                   # [K][3D], oldest first
        return qkv + (w.flip(1).t() * x).sum(0)
    x = F.pad(qkv, (0, 0, K - 1, 0))
    T = qkv.shape[1]
    return qkv + sum(w[:, j] * x[:, K - 1 - j:K - 1 - j + T] for j in range(K))


def out_gate(p, a, u, read):
    """read scaled channel by channel by 2 sigmoid(u A' B'), 1 at the start."""
    return read * 2 * torch.sigmoid((u @ p[a + 'out_gate_a'].T) @ p[a + 'out_gate_b'].T)


def mlp(p, prefix, z):
    """The SwiGLU MLP after a block, when the block has one: z + W2 (silu(W1 u) * W3 u), u = LN(z)."""
    m = prefix + 'mlp.'
    if m + 'fc1.weight' not in p:
        return z
    u = O.layer_norm(z, p[m + 'ln.weight'], p[m + 'ln.bias'])
    h = F.silu(half_product(u, p[m + 'fc1.weight'])) * half_product(u, p[m + 'fc3.weight'])
    return z + half_product(h, p[m + 'fc2.weight'])


def block(c, p, prefix, z, variant):
    """The oracle's block with the recurrence above."""
    B, T, D = z.shape
    H, L, dh, W = c.heads, c.kan, c.head_width, c.width
    a = prefix + 'temporal.attn.'
    u = O.layer_norm(z, p[prefix + 'temporal.ln.weight'], p[prefix + 'temporal.ln.bias'])
    if a + 'qkv_group.weight' in p:
        # grouped: [B][T][H][dh] x [H][3dh][dh] -> [B][T][H][3][dh], then the oracle's [3][H][dh] order
        qkv = torch.einsum('bthd,hed->bthe', u.view(B, T, H, dh), p[a + 'qkv_group.weight']).view(B, T, H, 3, dh).transpose(2, 3)
        qkv = qkv + p[a + 'qkv_proj.bias'].view(1, 1, 3, H, dh)
    else:
        qkv = (half_product(projection_input(u), projection(p, a + 'qkv_proj.weight')) + p[a + 'qkv_proj.bias']).view(B, T, 3, H, dh)
    if a + 'qkv_conv_kernel' in p:
        qkv = qkv_conv(p, a, qkv.reshape(B, T, 3 * D)).view(B, T, 3, H, dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    qh = torch.einsum('bthd,hld->bthl', q, p[a + 'q_harmonic_weight'])
    kh = torch.einsum('bthd,hld->bthl', k, p[a + 'k_harmonic_weight']) * p[a + 'spectral_coeffs']
    logr, theta = pole(p, a, variant)
    if a + 'erase_gate_weight' in p:
        la = selective_logmag(p, a, u) if a + 'pole_decay_select' in p else logr.expand(B, T, H, L)
        th = selective_theta(p, a, u) if a + 'pole_omega_select' in p else theta
        read = recurrence_erase(qh, kh, v, la, th, erase_strength(p, a, u))
    elif a + 'pole_omega_select' in p:
        la = selective_logmag(p, a, u) if a + 'pole_decay_select' in p else logr.expand(B, T, H, L)
        read = recurrence_selective(qh, kh, v, la, selective_theta(p, a, u))
    elif a + 'pole_decay_select' in p:
        read = recurrence_selective(qh, kh, v, selective_logmag(p, a, u), theta)
    else:
        read = recurrence(qh, kh, v, logr, theta)
    if a + 'out_gate_a' in p:
        read = out_gate(p, a, u, read.reshape(B, T, D)).view(B, T, H, dh)
    if a + 'out_group.weight' in p:
        temporal = torch.einsum('bthd,hed->bthe', read, p[a + 'out_group.weight']).reshape(B, T, D) + p[a + 'out_proj.bias']
    else:
        temporal = half_product(projection_input(read.reshape(B, T, D)), projection(p, a + 'out_proj.weight')) + p[a + 'out_proj.bias']
    n = O.normalize(z.view(B, T, O.GROUPS, W))
    pairs = O.GROUPS // 2
    th = c.dt * p[prefix + 'spatial.theta']
    cos, sin = torch.cos(th) - 1, torch.sin(th)
    first, second = n[:, :, 0:2 * pairs:2], n[:, :, 1:2 * pairs:2]
    rotated = torch.stack((cos * first - sin * second, sin * first + cos * second), dim=3)
    rotated = rotated.reshape(B, T, 2 * pairs, W)
    rotated = torch.cat((rotated, n.new_zeros(B, T, O.GROUPS - 2 * pairs, W)), dim=2)
    spatial = p[prefix + 'spatial.coupling'][:, None] * rotated
    gate = torch.sigmoid(p[prefix + 'gate_logit'])[:, None]
    return mlp(p, prefix, z + (gate * temporal.view(B, T, O.GROUPS, W) + (1 - gate) * spatial).reshape(B, T, D))


def forward(c, p, table, x, variant, checkpoint=False):
    z = embed(c, p, table, x)
    for i in range(c.layers):
        if checkpoint and torch.is_grad_enabled():
            z = torch.utils.checkpoint.checkpoint(block, c, p, f'blocks.{i}.', z, variant, use_reentrant=False)
        else:
            z = block(c, p, f'blocks.{i}.', z, variant)
    f = O.layer_norm(z, p['ln_f.weight'], p['ln_f.bias'])
    B, T, D = f.shape
    fg = f.view(B, T, O.GROUPS, c.width)
    counts = torch.arange(1, T + 1, dtype=f.dtype, device=f.device).view(1, T, 1, 1)
    mean = fg.cumsum(dim=1) / counts
    cos = (O.normalize(fg) * O.normalize(mean)).sum(-1)
    length = mean.norm(dim=-1)
    return half_product(torch.cat([f, cos, length], dim=-1), p['head.weight'])


def loss_of(c, p, table, w, variant, checkpoint=False):
    logits = forward(c, p, table, w[:, :-1], variant, checkpoint)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1))


# ---- generation: one position at a time, the recurrence as written ----
class Sampler:
    def __init__(self, c, p, table, variant):
        self.c, self.p, self.table, self.variant = c, p, table, variant
        H, L, dh = c.heads, c.kan, c.head_width
        self.S = [(torch.zeros(H, L, dh, device=table.device), torch.zeros(H, L, dh, device=table.device)) for _ in range(c.layers)]
        self.poles = [pole(p, f'blocks.{i}.temporal.attn.', variant) for i in range(c.layers)]
        K = {i: p[f'blocks.{i}.temporal.attn.qkv_conv_kernel'].shape[1] for i in range(c.layers)
             if f'blocks.{i}.temporal.attn.qkv_conv_kernel' in p}
        self.history = {i: [torch.zeros(3 * c.d, device=table.device)] * (k - 1) for i, k in K.items()}
        self.sum = torch.zeros(O.GROUPS, c.width, device=table.device)
        self.count = 0

    @torch.no_grad()
    def logits(self, byte):
        c, p = self.c, self.p
        H, L, dh, W = c.heads, c.kan, c.head_width, c.width
        z = embed(c, p, self.table, byte)
        for i in range(c.layers):
            prefix, a = f'blocks.{i}.', f'blocks.{i}.temporal.attn.'
            u = O.layer_norm(z, p[prefix + 'temporal.ln.weight'], p[prefix + 'temporal.ln.bias'])
            if a + 'qkv_group.weight' in p:
                qkv = torch.einsum('hd,hed->he', u.view(H, dh), p[a + 'qkv_group.weight']).view(H, 3, dh).transpose(0, 1) + p[a + 'qkv_proj.bias'].view(3, H, dh)
            else:
                qkv = (projection_input(u) @ projection(p, a + 'qkv_proj.weight').T + p[a + 'qkv_proj.bias']).view(3, H, dh)
            if i in self.history:
                flat = qkv.reshape(-1)
                qkv = qkv_conv(p, a, flat, self.history[i]).view(3, H, dh)
                self.history[i] = (self.history[i] + [flat])[1:]
            q, k, v = qkv[0], qkv[1], qkv[2]
            qh = torch.einsum('hd,hld->hl', q, p[a + 'q_harmonic_weight'])
            kh = torch.einsum('hd,hld->hl', k, p[a + 'k_harmonic_weight']) * p[a + 'spectral_coeffs']
            logr, theta = self.poles[i]
            if a + 'pole_decay_select' in p:
                logr = selective_logmag(p, a, u)
            if a + 'pole_omega_select' in p:
                theta = selective_theta(p, a, u)
            cre, cim = torch.exp(logr) * torch.cos(theta), torch.exp(logr) * torch.sin(theta)
            Sre, Sim = self.S[i]
            inj = kh[:, :, None] * v[:, None, :]
            Sre, Sim = cre[:, :, None] * Sre - cim[:, :, None] * Sim + inj, cre[:, :, None] * Sim + cim[:, :, None] * Sre
            if a + 'erase_gate_weight' in p:
                # the erase acts on the decayed state before the write: take the write back out, erase, put it back
                Xre, Xim = Sre - inj, Sim
                kn = kh / (kh.norm(dim=-1, keepdim=True) + 1e-6)
                beta = erase_strength(p, a, u)[:, None, None]
                Sre = Xre - beta * kn[:, :, None] * torch.einsum('hl,hld->hd', kn, Xre)[:, None, :] + inj
                Sim = Xim - beta * kn[:, :, None] * torch.einsum('hl,hld->hd', kn, Xim)[:, None, :]
            self.S[i] = (Sre, Sim)
            read = torch.einsum('hl,hld->hd', qh[:, :L], Sre)
            if qh.shape[1] > L:
                read = read + torch.einsum('hl,hld->hd', qh[:, L:], Sim)
            if a + 'out_gate_a' in p:
                read = out_gate(p, a, u, read.reshape(-1)).view(H, dh)
            if a + 'out_group.weight' in p:
                temporal = torch.einsum('hd,hed->he', read, p[a + 'out_group.weight']).reshape(-1) + p[a + 'out_proj.bias']
            else:
                temporal = projection_input(read.reshape(-1)) @ projection(p, a + 'out_proj.weight').T + p[a + 'out_proj.bias']
            n = O.normalize(z.view(O.GROUPS, W))
            pairs = O.GROUPS // 2
            th = c.dt * p[prefix + 'spatial.theta']
            cos, sin = torch.cos(th) - 1, torch.sin(th)
            first, second = n[0:2 * pairs:2], n[1:2 * pairs:2]
            rotated = torch.stack((cos * first - sin * second, sin * first + cos * second), dim=1).reshape(2 * pairs, W)
            rotated = torch.cat((rotated, n.new_zeros(O.GROUPS - 2 * pairs, W)), dim=0)
            spatial = p[prefix + 'spatial.coupling'][:, None] * rotated
            gate = torch.sigmoid(p[prefix + 'gate_logit'])[:, None]
            z = mlp(p, prefix, z + (gate * temporal.view(O.GROUPS, W) + (1 - gate) * spatial).reshape(-1))
        f = O.layer_norm(z, p['ln_f.weight'], p['ln_f.bias'])
        fg = f.view(O.GROUPS, W)
        self.sum = self.sum + fg
        self.count += 1
        mean = self.sum / self.count
        cos = (O.normalize(fg) * O.normalize(mean)).sum(-1)
        return torch.cat([f, cos, mean.norm(dim=-1)]) @ p['head.weight'].T

    def run(self, seed_bytes, count, temperature, generator):
        out = []
        logits = None
        for b in seed_bytes:
            logits = self.logits(int(b))
        for _ in range(count):
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator).item())
            out.append(nxt)
            logits = self.logits(nxt)
        return np.array(out, dtype=np.uint8 if logits.numel() == 256 else np.int64)


# ---- measurement ----
def log_mel(x, n_fft=512, hop=160, mels=40):
    xt = torch.from_numpy(x).float()
    spec = torch.stft(xt, n_fft, hop, window=torch.hann_window(n_fft), return_complex=True).abs() ** 2
    freqs = torch.linspace(0, RATE / 2, n_fft // 2 + 1)
    mel = 2595 * torch.log10(1 + freqs / 700)
    edges = torch.linspace(mel[0], mel[-1], mels + 2)
    fb = torch.zeros(mels, freqs.numel())
    for m in range(mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = torch.clamp(torch.minimum((mel - lo) / (mid - lo), (hi - mel) / (hi - mid)), min=0)
    return torch.log10(fb @ spec + 1e-8).numpy()


def measure(gen, truth):
    """gen, truth: float arrays of equal length."""
    blocks = RATE // 10
    nb = min(gen.size, truth.size) // blocks
    g, t = gen[:nb * blocks].reshape(nb, blocks), truth[:nb * blocks].reshape(nb, blocks)
    rms_g, rms_t = np.sqrt((g ** 2).mean(1)), np.sqrt((t ** 2).mean(1))
    floor = rms_t.max() / 30
    return {
        'rmsGenerated': float(rms_g.mean()), 'rmsTruth': float(rms_t.mean()),
        'rmsByBlockGenerated': [round(float(v), 5) for v in rms_g], 'silentFraction': float((rms_g < floor).mean()),
        'logMelDistance': float(np.abs(log_mel(gen[:nb * blocks]) - log_mel(truth[:nb * blocks])).mean()),
        'logMelDistanceSilence': float(np.abs(log_mel(np.zeros(nb * blocks, np.float32)) - log_mel(truth[:nb * blocks])).mean()),
    }


def write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(path, RATE, np.clip(x, -1, 1).astype(np.float32))


# ---- data ----
class Stream:
    def __init__(self, d, window=O.FULL_CONTEXT):
        m = json.loads((d / 'manifest.json').read_text())
        self.m = m
        self.T = window
        self.train = [np.fromfile(d / s['train']['file'], dtype=np.uint8) for s in m['shards']]
        self.cat = np.concatenate(self.train)
        h = m['heldout']
        text = np.fromfile(d / h['file'], dtype=np.uint8)
        count = -(-h['scoredBytes'] // self.T)
        stream = np.zeros(count * self.T + 1, dtype=np.uint8)
        n = min(text.size - h['contextBytes'] + 1, stream.size)   # the scored span may be shorter than the file
        stream[:n] = text[h['contextBytes'] - 1:h['contextBytes'] - 1 + n]
        self.heldout = np.lib.stride_tricks.as_strided(stream, shape=(count, self.T + 1), strides=(self.T, 1)).copy()
        self.scored = h['scoredBytes']
        self.heldout_audio = mulaw.decode(text[h['contextBytes']:])
        self.seeds = {k: np.fromfile(d / v['file'], dtype=np.uint8) for k, v in m['seeds'].items()}

    def windows(self, first, count):
        T = self.T
        return np.stack([self.cat[k * T:k * T + T + 1] for k in range(first, first + count)])


def score(c, p, table, windows, variant, device, batch=8):
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            w = torch.from_numpy(windows[i:i + batch].astype(np.int64)).to(device)
            logits = forward(c, p, table, w[:, :-1], variant)
            l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), w[:, 1:].reshape(-1), reduction='sum')
            tot += l.item()
            n += w[:, 1:].numel()
    return tot / n / math.log(2)


def held_seeds(stream, n=3, seed_s=1.0, gen_s=2.0):
    """Seeds from the held-out audio at loud spots, as gen_held.py chose them."""
    a = stream.heldout_audio
    rms = np.sqrt((a ** 2).mean())
    ns, ng = int(seed_s * RATE), int(gen_s * RATE)
    span = a.size - ns - ng - 1600
    out = []
    for i in range(n):
        start = (i + 1) * span // (n + 1)
        while start < span and np.sqrt((a[start:start + 1600] ** 2).mean()) < rms / 3:
            start += 1600
        out.append((mulaw.encode(a[start:start + ns]), a[start + ns:start + ns + ng]))
    return out


def poles_of(p, i, variant):
    logr, theta = pole(p, f'blocks.{i}.temporal.attn.', variant)
    return {'hz': (theta.abs() * RATE / (2 * math.pi)).flatten().tolist(),
            'decayMs': (-1000 / RATE / logr).flatten().tolist()}


def describe_poles(p, c, variant):
    rows = []
    for i in range(c.layers):
        q = poles_of(p, i, variant)
        d, hz = np.array(q['decayMs']), np.array(q['hz'])
        rows.append(f'block {i}: decay ms {d.min():.3g}/{np.median(d):.3g}/{d.max():.3g}, hz {hz.min():.3g}-{hz.max():.3g}')
    return '; '.join(rows)


def score_saved(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = O.Config(harmonics=args.harmonics, layers=args.layers, heads=7, kan=8, dt=0.3)
    stream = Stream(args.bytes, args.window)
    table = O.wire_table(c.harmonics).to(device)
    p = {k: v.to(device) for k, v in torch.load(args.params, map_location='cpu').items()}
    t0 = time.time()
    bpb = score(c, p, table, stream.heldout, args.variant, device, batch=args.batch)
    row = {'variant': args.variant, 'params': str(args.params), 'window': args.window, 'windows': int(stream.heldout.shape[0]),
           'heldoutBpb': bpb, 'seconds': time.time() - t0}
    print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, 'a') as f:
            f.write(json.dumps(row) + '\n')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = O.Config(harmonics=args.harmonics, layers=args.layers, heads=7, kan=8, dt=0.3)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train.log', 'a')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()
    stream = Stream(args.bytes, args.window)
    table = O.wire_table(c.harmonics).to(device)
    p = {k: v.to(device) for k, v in make_params(c, args.variant, args.seed).items()}
    count = sum(v.numel() for v in p.values())
    say(f'variant {args.variant} layers {c.layers} width {c.d} params {count} device {device} batch {args.batch} window {args.window} updates {args.updates}')
    say('initial ' + describe_poles(p, c, args.variant))
    upd = O.Update(p, args.updates)
    eval_windows = stream.heldout[:args.eval_windows]
    t0 = time.time()
    for step in range(args.updates):
        w = torch.from_numpy(stream.windows(step * args.batch, args.batch).astype(np.int64)).to(device)
        for v in p.values():
            v.requires_grad_(True)
        l = loss_of(c, p, table, w, args.variant, checkpoint=args.checkpoint)
        l.backward()
        grads = {k: v.grad for k, v in p.items()}
        with torch.no_grad():
            p = {k: v.detach() for k, v in p.items()}
            norm = upd(p, grads)
            project(p, args.variant)
        if step % 20 == 0 or step == args.updates - 1:
            say(f'step {step:6d}  loss {l.item() / math.log(2):.4f} bpb  gradnorm {norm:.3f}  {time.time() - t0:.0f} s')
        if (step + 1) % args.eval_every == 0 or step == args.updates - 1:
            bpb = score(c, p, table, eval_windows, args.variant, device)
            say(f'step {step:6d}  held-out ({eval_windows.shape[0]} windows) {bpb:.4f} bpb')
    torch.save({k: v.cpu() for k, v in p.items()}, out / 'params.pt')
    full = score(c, p, table, stream.heldout, args.variant, device)
    say(f'final held-out (all {stream.heldout.shape[0]} windows, {stream.scored} bytes) {full:.4f} bpb')
    result = {'variant': args.variant, 'layers': c.layers, 'harmonics': c.harmonics, 'params': count, 'window': args.window,
              'updates': args.updates, 'batch': args.batch, 'seconds': time.time() - t0, 'heldoutBpb': full, 'continuations': {}}
    # poles as learned: frequency in Hz and decay in ms, every block
    result['poles'] = [poles_of(p, i, args.variant) for i in range(c.layers)]
    result['block0Poles'] = result['poles'][0]
    say('learned ' + describe_poles(p, c, args.variant))
    # continuations
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for temperature in (1.0, 0.8):
        rows = []
        for i, (seed_bytes, truth) in enumerate(held_seeds(stream, args.held_seeds, 1.0, args.gen_seconds)):
            s = Sampler(c, p, table, args.variant)
            g = mulaw.decode(s.run(seed_bytes, truth.size, temperature, gen))
            rows.append(measure(g, truth))
            write_wav(out / f'held{i}_t{temperature}_generated.wav', np.concatenate([mulaw.decode(seed_bytes), g]))
            write_wav(out / f'held{i}_truth.wav', np.concatenate([mulaw.decode(seed_bytes), truth]))
        s = Sampler(c, p, table, args.variant)
        g = mulaw.decode(s.run(stream.seeds['behold'], int(args.gen_seconds * RATE), temperature, gen))
        write_wav(out / f'behold_t{temperature}_generated.wav', np.concatenate([mulaw.decode(stream.seeds['behold']), g]))
        summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'rmsByBlockGenerated'}
        summary['seeds'] = rows
        result['continuations'][str(temperature)] = summary
        say(f'continuation T={temperature}: rms gen {summary["rmsGenerated"]:.4f} vs truth {summary["rmsTruth"]:.4f}, '
            f'silent {summary["silentFraction"]:.2f}, log-mel distance {summary["logMelDistance"]:.3f} (silence would be {summary["logMelDistanceSilence"]:.3f})')
    (out / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    say('done')


def check():
    torch.manual_seed(0)
    c = O.Config(harmonics=4, layers=2, heads=7, kan=8, dt=0.3)
    table = O.wire_table(c.harmonics, torch.float64)
    x = torch.randint(0, 256, (2, 2 * CHUNK))
    # 1. the chunked recurrence equals the oracle's loop (real variant)
    p = make_params(c, 'real', 0, torch.float64)
    ours = forward(c, p, table, x, 'real')
    theirs = O.forward(c, p, table, x)
    print(f'real variant vs oracle loop: max |diff| {(ours - theirs).abs().max().item():.3e} (logits up to {theirs.abs().max().item():.2f})')
    # 2. the chunked recurrence equals the sampler's one-position loop (polar and continuous-time variants)
    for variant in ('polar', 'ct'):
        p = make_params(c, variant, 0, torch.float64)
        with torch.no_grad():
            ours = forward(c, p, table, x, variant)
            s = Sampler(c, p, table.to(torch.float64), variant)
            seq = torch.stack([s.logits(int(b)) for b in x[0]])
        print(f'{variant} variant vs position loop: max |diff| {(ours[0] - seq).abs().max().item():.3e}')
    p = make_params(c, 'ct', 0, torch.float64)
    print('ct initial ' + describe_poles(p, c, 'ct'))
    # 3. block recomputation in the backward pass changes nothing
    p = make_params(c, 'polar', 0, torch.float64)
    y = torch.randint(0, 256, (2, 2 * CHUNK + 1))
    grads = []
    for ck in (False, True):
        q = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
        loss_of(c, q, table, y, 'polar', checkpoint=ck).backward()
        grads.append(torch.cat([q[k].grad.flatten() for k in sorted(q)]))
    print(f'checkpointed backward vs plain: max |diff| {(grads[0] - grads[1]).abs().max().item():.3e}')
    # 4. gradients of the pole parameters against central differences
    for variant in ('ct', 'polar', 'cartesian', 'real'):
        p = make_params(c, variant, 0, torch.float64)
        y = torch.randint(0, 256, (2, 2 * CHUNK + 1))
        for v in p.values():
            v.requires_grad_(True)
        loss_of(c, p, table, y, variant).backward()
        worst = 0.0
        for name in [n for n in p if 'pole' in n or 'decay' in n or 'gate_logit' in n or 'q_harmonic' in n][:6]:
            t = p[name]
            flat = t.detach().view(-1)
            for j in (0, flat.numel() // 2):
                eps = 1e-6
                base = flat[j].item()
                flat[j] = base + eps
                with torch.no_grad():
                    lp = loss_of(c, {k: v.detach() for k, v in p.items()}, table, y, variant).item()
                flat[j] = base - eps
                with torch.no_grad():
                    lm = loss_of(c, {k: v.detach() for k, v in p.items()}, table, y, variant).item()
                flat[j] = base
                fd = (lp - lm) / (2 * eps)
                an = t.grad.view(-1)[j].item()
                worst = max(worst, abs(fd - an) / max(abs(fd), abs(an), 1e-8))
        print(f'{variant}: worst relative gradient error on pole/query parameters {worst:.2e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--variant', required=True, choices=['real', 'cartesian', 'polar', 'ct'])
    t.add_argument('--bytes', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--harmonics', type=int, default=32)
    t.add_argument('--batch', type=int, default=16)
    t.add_argument('--window', type=int, default=O.FULL_CONTEXT, help='bytes per training and scoring window (a multiple of 64)')
    t.add_argument('--updates', type=int, default=3000)
    t.add_argument('--eval-every', type=int, default=500)
    t.add_argument('--eval-windows', type=int, default=256)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--checkpoint', action='store_true', help='recompute each block in the backward pass to save memory')
    t.add_argument('--gen-seconds', type=float, default=2.0)
    t.add_argument('--held-seeds', type=int, default=3)
    s = sub.add_parser('score', help='score a saved params.pt on the held-out bytes at a given window')
    s.add_argument('--variant', required=True, choices=['real', 'cartesian', 'polar', 'ct'])
    s.add_argument('--params', type=Path, required=True)
    s.add_argument('--bytes', type=Path, required=True)
    s.add_argument('--window', type=int, default=O.FULL_CONTEXT)
    s.add_argument('--layers', type=int, default=4)
    s.add_argument('--harmonics', type=int, default=32)
    s.add_argument('--batch', type=int, default=8)
    s.add_argument('--out', type=Path, default=None, help='append the result as one JSON line')
    sub.add_parser('check')
    args = ap.parse_args()
    if args.command == 'check':
        check()
    elif args.command == 'score':
        score_saved(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
