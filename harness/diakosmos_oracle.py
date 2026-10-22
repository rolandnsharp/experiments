#!/usr/bin/env python3
"""Diakosmos's reference: harmonic-gpt's spectral_kan_unified_scan_planar from its equations.

Measurement and reference only (PyTorch on the CPU); never on the training
path. Written from the equations of MonumentalSystems/harmonic-gpt at bd71df2
(scripts/wire_lm_variants.py SpectralKANUnifiedWireLM with
spatial_kind="planar_givens", harmonic_gpt/layers/spectral_kan_attention.py,
harmonic_gpt/layers/wire_encoding.py, harmonic_gpt/core/readout.py), without
importing it.

    diakosmos-oracle.py table OUT.f32
        the wire table: 256 rows of 7 x 2 x 95 binary32, each byte's phasors,
        computed in binary64 and rounded once
    diakosmos-oracle.py check [--harmonic-gpt DIR] [--full]
        the reference's own checks: every parameter's gradient against central
        differences, causality, and (with --harmonic-gpt) its logits, wire
        table and four updates against harmonic-gpt's own modules and
        optimizers on the same weights; --full
        adds the 100M configuration's parameter count and one forward pass
    diakosmos-oracle.py checkpoint --seed S [--perturb X] OUT.payload
        a weights checkpoint's payload: the bank of Diakosmos.Model (every
        tensor padded to 1344-wide rows, 64-word aligned; the wire table
        after the parameters), binary32, from the seed's initialization;
        wrap it with scripts/alpha/checkpoint-envelope.py assemble
    diakosmos-oracle.py windows --bytes FILE --count N OUT.u32
        N windows of 2049 bytes (window k from byte 2048 k) as u32 ids
    diakosmos-oracle.py score --payload P --windows W.u32 OUT.f32 [--exact]
        every position's next-byte loss, window by window; the matrix
        products' operands rounded to bfloat16 as the device's are, unless
        --exact
    diakosmos-oracle.py gradient --payload P --windows W.u32 OUT.f32 [--exact]
        the mean loss's gradient over the windows (every position), as a bank
        (Diakosmos.Model's layout, zero at the padding and the wire table);
        the products and their pullbacks round as the device's
    diakosmos-oracle.py compare-gradient DEVICE.f32 REFERENCE.f32
        each tensor's gradient: largest difference over the largest entry, cosine
    diakosmos-oracle.py update --payload P --gradient G.f32 [--state S --step K] --total N OUT.payload OUT.state
        one update (clip, Muon, rotational AdamW, Adam) of the payload by the
        gradient bank, from the state after K updates of a run of N; the
        state as the device keeps it (m and v as banks, then each rotational
        row's first norm and running mean); Newton-Schulz's products round
        as the device's, unless --exact
    diakosmos-oracle.py compare-update --before P DEVICE.payload REFERENCE.payload [--device-state S --reference-state S]
        each tensor's change, and the states
    diakosmos-oracle.py training-checkpoint --payload P [--state S --step K] --total N [--peak-lr R --warmup W] OUT.payload
        a training checkpoint's payload (Diakosmos.Build.CheckpointTransfer):
        the master, the state (zero, or update's), the run's schedule (every
        update's scalars, binary64 rounded once; a fine-tune's own peak and
        warmup) and the updates made; wrap it with checkpoint-envelope.py
        assemble and the trainingPrefix artifact
    diakosmos-oracle.py training-parts CKPT OUT.payload OUT.state
        a training checkpoint's master and state, as update writes them
    diakosmos-oracle.py compare DEVICE.result REFERENCE.f32
        the device's scoring result (losses then its 160-byte status) against
        the reference's

The model, for bytes x_1 .. x_T (G = 7 wire groups of W = 2 x harmonics reals,
D = G W, H heads of D / H, L harmonic channels a head):

  z = table[x]                                        each group a unit vector
  each block:
    u = LayerNorm(z);  q, k, v = split(u Wqkv' + bqkv)  per head
    q~[h, l] = <q[h], Q[h, l]>;  k~[h, l] = c[h, l] <k[h], K[h, l]>
    gamma[h, l] = sigmoid(g[h] + o[l])
    S_t[h, l] = gamma[h, l] S_{t-1}[h, l] + k~_t[h, l] v_t[h];  S_0 = 0
    temporal_t = concat_h(sum_l q~_t[h, l] S_t[h, l]) Wo' + bo
    n = each group of z normalized;  for pairs (0, 1), (2, 3), (4, 5):
      a' = (cos(dt theta) - 1) a - sin(dt theta) b
      b' = sin(dt theta) a + (cos(dt theta) - 1) b      elementwise; group 6: 0
    spatial = coupling[group] x that
    z = z + sigmoid(beta[group]) temporal + (1 - sigmoid(beta[group])) spatial
  f = LayerNorm_f(z);  m_t = mean of f's groups over 1 .. t
  features = [f, cos(f's group, m's group) for each group, |m's group| for each]
  logits = features Whead'

Dropout is 0: harmonic-gpt's 0.1 on the recurrence's output is regularization
for repeated data, and this run reads each byte once (as llm.c's GPT-2 small
does, without dropout).
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

GROUPS = 7
EPS_LN = 1e-5
EPS_NORMALIZE = 1e-12


@dataclass(frozen=True)
class Config:
    harmonics: int = 95
    layers: int = 14
    heads: int = 7
    kan: int = 8
    dt: float = 0.3

    @property
    def width(self):
        return 2 * self.harmonics

    @property
    def d(self):
        return GROUPS * self.width

    @property
    def head_width(self):
        return self.d // self.heads

    @property
    def features(self):
        return self.d + 2 * GROUPS


FULL = Config()
FULL_CONTEXT = 2048


def wire_phases(harmonics):
    """(256, 7, harmonics) phases in binary64: harmonic_gpt.layers.wire_encoding.encode."""
    pi = math.pi
    qpsk = [pi / 4, 3 * pi / 4, -pi / 4, -3 * pi / 4]
    gray = [0, 1, 3, 2]

    def wrap(x):
        return math.fmod(math.fmod(x + pi, 2 * pi) + 2 * pi, 2 * pi) - pi

    rows = []
    for byte in range(256):
        d = [qpsk[gray[(byte >> s) & 3]] for s in (6, 4, 2, 0)]
        p = [wrap(d[0] + d[2] + d[3]), wrap(d[0] + d[1] + d[3]), wrap(d[1] + d[2] + d[3])]
        a_upper, d_upper = (d[0] + d[1]) / 2, (d[0] - d[1]) / 2
        a_lower, d_lower = (d[2] + d[3]) / 2, (d[2] - d[3]) / 2
        a_byte, d_byte = (a_upper + a_lower) / 2, (a_upper - a_lower) / 2
        a_par = (p[0] + p[1] + p[2]) / 3
        d_par1, d_par2 = p[0] - a_par, p[1] - a_par
        scales = [
            d + p,
            [a_upper, d_upper, a_lower, d_lower, a_par, d_par1, d_par2],
            [a_byte, d_byte, a_byte, d_byte, a_par, d_par1, d_par2],
            [a_upper * d_lower, d_upper * a_lower, a_byte * d_byte, d_upper * d_lower,
             p[0] * p[1], p[1] * p[2], p[0] * p[2]],
        ]
        base = len(scales)
        fiber = scales[:harmonics]
        for k in range(1, (harmonics + base - 1) // base):
            rotation = k * pi / (2 * base)
            for h in range(base):
                if len(fiber) >= harmonics:
                    break
                fiber.append([phase + rotation for phase in scales[h]])
        rows.append(fiber[:harmonics])
    return torch.tensor(rows, dtype=torch.float64).permute(0, 2, 1)


def wire_table(harmonics, dtype=torch.float32):
    """(256, D): each group's phasors (real, imaginary interleaved), unit length a group."""
    phases = wire_phases(harmonics)
    z = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1) / math.sqrt(harmonics)
    return z.reshape(256, GROUPS * 2 * harmonics).to(dtype)


def parameter_shapes(c):
    D, H, L, dh, W = c.d, c.heads, c.kan, c.head_width, c.width
    each = {
        'temporal.ln.weight': (D,), 'temporal.ln.bias': (D,),
        'temporal.attn.qkv_proj.weight': (3 * D, D), 'temporal.attn.qkv_proj.bias': (3 * D,),
        'temporal.attn.q_harmonic_weight': (H, L, dh), 'temporal.attn.k_harmonic_weight': (H, L, dh),
        'temporal.attn.spectral_coeffs': (H, L), 'temporal.attn._gate_logit': (H,),
        'temporal.attn._harmonic_decay_offset': (L,),
        'temporal.attn.out_proj.weight': (D, D), 'temporal.attn.out_proj.bias': (D,),
        'spatial.theta': (GROUPS // 2, W), 'spatial.coupling': (GROUPS,), 'gate_logit': (GROUPS,),
    }
    shapes = {f'blocks.{i}.{k}': s for i in range(c.layers) for k, s in each.items()}
    shapes.update({'ln_f.weight': (D,), 'ln_f.bias': (D,), 'head.weight': (256, c.features)})
    return shapes


def initialize(c, seed=0, dtype=torch.float32):
    """harmonic-gpt's initial distributions (PyTorch defaults where it sets none)."""
    g = torch.Generator().manual_seed(seed)
    D, H, L, dh = c.d, c.heads, c.kan, c.head_width

    def uniform(shape, fan_in):
        bound = 1 / math.sqrt(fan_in)
        return (torch.rand(shape, generator=g, dtype=torch.float64) * 2 - 1) * bound

    params = {}
    for name, shape in parameter_shapes(c).items():
        key = name.split('.', 2)[-1] if name.startswith('blocks.') else name
        if key.endswith(('ln.weight',)) or key == 'ln_f.weight':
            t = torch.ones(shape, dtype=torch.float64)
        elif key.endswith('ln.bias') or key == 'ln_f.bias':
            t = torch.zeros(shape, dtype=torch.float64)
        elif key.endswith(('qkv_proj.weight', 'qkv_proj.bias', 'out_proj.weight', 'out_proj.bias')):
            t = uniform(shape, D)
        elif key.endswith(('q_harmonic_weight', 'k_harmonic_weight')):
            t = torch.randn(shape, generator=g, dtype=torch.float64) * math.sqrt(2 / dh)
        elif key.endswith('spectral_coeffs'):
            t = (1 / (1 + torch.arange(L, dtype=torch.float64))).expand(H, L).clone()
        elif key.endswith('_gate_logit'):
            t = torch.full(shape, 3.0, dtype=torch.float64)
        elif key.endswith('_harmonic_decay_offset'):
            t = torch.linspace(0, -1, L, dtype=torch.float64)
        elif key == 'spatial.theta':
            t = torch.randn(shape, generator=g, dtype=torch.float64) * 0.02
        elif key == 'spatial.coupling':
            t = torch.full(shape, 0.5, dtype=torch.float64)
        elif key == 'gate_logit':
            t = torch.zeros(shape, dtype=torch.float64)
        elif key == 'head.weight':
            t = uniform(shape, c.features)
        else:
            raise SystemExit(f'no initialization for {name}')
        params[name] = t
    return {name: t.to(dtype) for name, t in centre_rotational(params).items()}


# The update (harmonic-gpt's riemannian_muon_adamw at its 100M settings,
# scripts/ultimate_chinchilla_bakeoff.py make_optimizers, set_optimizer_lrs,
# cosine_lr; harmonic_gpt/optim/riemannian_muon.py, rotational_adamw.py):
# the gradient is the mean over the update's windows, clipped at global norm
# 1.0 (torch's clip_grad_norm_: scaled by min(1, 1 / (norm + 1e-6))); then
#   Muon, the recurrences' output projections: b = 0.95 b + 0.05 g;
#     X = (0.05 g + 0.95 b) / (|.|_F + 1e-7); five times A = X X',
#     X = 3.4445 X + (-4.7750 A + 2.0315 A A) X; p -= lr_muon X
#   rotational AdamW, the qkv projections and the head (each row a neuron):
#     Adam's direction d = m^ / (sqrt(v^) + 1e-8), row-centred, its component
#     along the row removed; a running mean of its squared row norm (0.9) sets
#     the step: p -= eta d / (1e-8 + rms) |p0|, eta = sqrt(2 lr 0.01 0.1 / 1.9);
#     each row rescaled to its first norm |p0|. Rows are centred at the start.
#   Adam everything else: p -= lr m^ / (sqrt(v^) + 1e-8), no weight decay.
# lr = 0.003 s, lr_muon = 0.003 x 0.6666667 s: s rises linearly over 200
# updates, then falls by a cosine to 0.1.
DYNAMICAL_KEYWORDS = (
    'coupling', 'harmonic', 'gate', 'spectral', 'adj', 'chebyshev', 'kan', 'omega', 'kappa', 'freq',
    'coeffs', 'kernel', 'phi', 'psi', 'probe', 'rotor', 'delta', 'biv', 'gauge', 'transport',
    'decay', 'theta', 'parity', 'raw_lambda', 'input_scale', 'gamma_mag', 'gamma_phase', 'alpha',
)
PEAK_LR = 0.003
MUON_RATIO = 0.6666667
WARMUP = 200
MIN_FRACTION = 0.1
BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
MOMENTUM = 0.95
WEIGHT_DECAY = 0.01
CLIP = 1.0


def role(name, shape):
    if len(shape) == 2 and 'attn.out_proj.weight' in name:
        return 'muon'
    if len(shape) > 1 and not any(k in name for k in DYNAMICAL_KEYWORDS):
        return 'rotational'
    return 'adam'


def schedule(step, total, warmup=WARMUP, min_fraction=MIN_FRACTION):
    """The learning rate's multiple of its peak at update `step` (from 0)."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(total - warmup, 1)))
    return min_fraction + 0.5 * (1 - min_fraction) * (1 + math.cos(math.pi * progress))


def row_norm(t):
    return t.reshape(t.shape[0], -1).norm(dim=1).reshape(-1, *([1] * (t.dim() - 1)))


def row_centred(t):
    flat = t.reshape(t.shape[0], -1)
    return (flat - flat.mean(dim=1, keepdim=True)).view_as(t)


def centre_rotational(params):
    for name, t in params.items():
        if role(name, t.shape) == 'rotational':
            centred = row_centred(t)
            params[name] = row_norm(t) * centred / (row_norm(centred) + 1e-12)
    return params


def newton_schulz(g):
    x = g / (g.norm() + 1e-7)
    for _ in range(5):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return x


def newton_schulz_half(g):
    """newton_schulz as the device computes it: each product's operands rounded to
    bfloat16 (X, X X', and the combination b A + c A A), accumulated in binary32,
    the iterate kept in binary32 between steps."""
    x = g / (g.norm() + 1e-7)
    for _ in range(5):
        xh = bf16(x)
        a = xh @ xh.T
        ah = bf16(a)
        b = -4.7750 * a + 2.0315 * (ah @ ah.T)
        x = 3.4445 * x + bf16(b) @ xh
    return x


class Update:
    """The optimizer's state and one update of every parameter from its gradient."""

    def __init__(self, params, total, ns=None):
        self.total = total
        self.step = 0
        self.state = {name: {} for name in params}
        self.ns = ns or newton_schulz

    def __call__(self, params, grads):
        norm = torch.sqrt(sum(g.double().pow(2).sum() for g in grads.values()))
        scale = min(1.0, CLIP / (norm.item() + 1e-6))
        s = schedule(self.step, self.total)
        lr, lr_muon = PEAK_LR * s, PEAK_LR * MUON_RATIO * s
        self.step += 1
        bc1, bc2 = 1 - BETA1 ** self.step, 1 - BETA2 ** self.step
        for name, p in params.items():
            g = grads[name] * scale
            st = self.state[name]
            kind = role(name, p.shape)
            if kind == 'muon':
                buf = st.setdefault('momentum', torch.zeros_like(p))
                buf.mul_(MOMENTUM).add_(g, alpha=1 - MOMENTUM)
                params[name] = p - lr_muon * self.ns((1 - MOMENTUM) * g + MOMENTUM * buf)
                continue
            m = st.setdefault('m', torch.zeros_like(p))
            v = st.setdefault('v', torch.zeros_like(p))
            m.mul_(BETA1).add_(g, alpha=1 - BETA1)
            v.mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
            denom = v.sqrt() / math.sqrt(bc2) + ADAM_EPS
            if kind == 'adam':
                params[name] = p - (lr / bc1) * m / denom
                continue
            if 'norm' not in st:
                st['norm'] = row_norm(row_centred(p))
                st['rotations'] = 0
            st['rotations'] += 1
            d = row_centred(m / denom / bc1)
            d = d - p * ((d * p).reshape(p.shape[0], -1).sum(1).reshape(st['norm'].shape) / (st['norm'] ** 2 + 1e-8))
            st['update'] = 0.1 * row_norm(d) ** 2 + 0.9 * st.get('update', 0)
            rms = torch.sqrt(st['update'] / (1 - 0.9 ** st['rotations']))
            eta = math.sqrt(2 * lr * WEIGHT_DECAY * (1 - BETA1) / (1 + BETA1))
            moved = p - eta * (d / (1e-8 + rms)) * st['norm']
            params[name] = moved * st['norm'] / (row_norm(moved) + 1e-8)
        return norm.item()


def bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


class HalfProduct(torch.autograd.Function):
    """x w' as the device computes it and its pullbacks: every product's operands bfloat16
    (the incoming gradient rounded too), accumulated in binary32, results binary32."""

    @staticmethod
    def forward(ctx, x, w):
        xb, wb = bf16(x), bf16(w)
        ctx.save_for_backward(xb, wb)
        return xb @ wb.T

    @staticmethod
    def backward(ctx, g):
        xb, wb = ctx.saved_tensors
        gb = bf16(g)
        dx = gb @ wb
        dw = gb.reshape(-1, gb.shape[-1]).T @ xb.reshape(-1, xb.shape[-1])
        return dx, dw


def product(x, w, half):
    """x w': on the device a tensor-core product of bfloat16 operands, accumulated in binary32."""
    if half:
        return HalfProduct.apply(x, w)
    return x @ w.T


def layer_norm(x, w, b):
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + EPS_LN) * w + b


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS_NORMALIZE)


def block(c, p, prefix, z, record=None, half=False):
    B, T, D = z.shape
    H, L, dh, W = c.heads, c.kan, c.head_width, c.width
    a = prefix + 'temporal.attn.'
    u = layer_norm(z, p[prefix + 'temporal.ln.weight'], p[prefix + 'temporal.ln.bias'])
    qkv = (product(u, p[a + 'qkv_proj.weight'], half) + p[a + 'qkv_proj.bias']).view(B, T, 3, H, dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    qh = torch.einsum('bthd,hld->bthl', q, p[a + 'q_harmonic_weight'])
    kh = torch.einsum('bthd,hld->bthl', k, p[a + 'k_harmonic_weight']) * p[a + 'spectral_coeffs']
    gamma = torch.sigmoid(p[a + '_gate_logit'][:, None] + p[a + '_harmonic_decay_offset'][None, :])
    state = z.new_zeros(B, H, L, dh)
    reads = []
    for t in range(T):
        state = gamma[None, :, :, None] * state + kh[:, t, :, :, None] * v[:, t, :, None, :]
        reads.append(torch.einsum('bhl,bhld->bhd', qh[:, t], state))
    read = torch.stack(reads, dim=1).reshape(B, T, D)
    temporal = product(read, p[a + 'out_proj.weight'], half) + p[a + 'out_proj.bias']
    n = normalize(z.view(B, T, GROUPS, W))
    pairs = GROUPS // 2
    theta = c.dt * p[prefix + 'spatial.theta']
    cos, sin = torch.cos(theta) - 1, torch.sin(theta)
    first, second = n[:, :, 0:2 * pairs:2], n[:, :, 1:2 * pairs:2]
    rotated = torch.stack((cos * first - sin * second, sin * first + cos * second), dim=3)
    rotated = rotated.reshape(B, T, 2 * pairs, W)
    rotated = torch.cat((rotated, n.new_zeros(B, T, GROUPS - 2 * pairs, W)), dim=2)
    spatial = p[prefix + 'spatial.coupling'][:, None] * rotated
    gate = torch.sigmoid(p[prefix + 'gate_logit'])[:, None]
    out = z + (gate * temporal.view(B, T, GROUPS, W) + (1 - gate) * spatial).reshape(B, T, D)
    if record is not None:
        record[prefix] = {'ln': u, 'q': qh, 'k': kh, 'v': v, 'read': read, 'temporal': temporal,
                          'spatial': spatial.reshape(B, T, D), 'out': out}
    return out


def forward(c, p, table, x, record=None, half=False):
    """Logits (B, T, 256) for bytes x (B, T)."""
    z = table[x]
    for i in range(c.layers):
        z = block(c, p, f'blocks.{i}.', z, record, half)
    f = layer_norm(z, p['ln_f.weight'], p['ln_f.bias'])
    B, T, D = f.shape
    fg = f.view(B, T, GROUPS, c.width)
    counts = torch.arange(1, T + 1, dtype=f.dtype).view(1, T, 1, 1)
    mean = fg.cumsum(dim=1) / counts
    cos = (normalize(fg) * normalize(mean)).sum(-1)
    length = mean.norm(dim=-1)
    features = torch.cat([f, cos, length], dim=-1)
    if record is not None:
        record['final'] = {'ln_f': f, 'mean': mean.reshape(B, T, D), 'features': features}
    return product(features, p['head.weight'], half)


def loss(c, p, table, x, y):
    logits = forward(c, p, table, x)
    return torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))


def check_gradient(c, seed=1):
    """Every parameter tensor's gradient against central differences, in binary64."""
    torch.manual_seed(seed)
    p = initialize(c, seed, torch.float64)
    for name in p:
        p[name] = p[name] + 0.05 * torch.randn_like(p[name])
    table = wire_table(c.harmonics, torch.float64)
    seq = torch.randint(0, 256, (2, 13))
    x, y = seq[:, :-1], seq[:, 1:]
    for t in p.values():
        t.requires_grad_(True)
    loss(c, p, table, x, y).backward()
    worst = 0.0
    eps = 1e-6
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, t in p.items():
            flat = t.view(-1)
            picks = range(flat.numel()) if flat.numel() <= 24 else torch.randperm(flat.numel(), generator=gen)[:24].tolist()
            for i in picks:
                keep = flat[i].item()
                flat[i] = keep + eps
                up = loss(c, p, table, x, y).item()
                flat[i] = keep - eps
                down = loss(c, p, table, x, y).item()
                flat[i] = keep
                numeric = (up - down) / (2 * eps)
                analytic = t.grad.view(-1)[i].item()
                error = abs(numeric - analytic) / max(1e-3, abs(numeric), abs(analytic))
                worst = max(worst, error)
                if error > 1e-5:
                    raise SystemExit(f'{name}[{i}]: gradient {analytic:.9g}, central difference {numeric:.9g}')
    return worst


def check_causality(c, seed=2):
    p = initialize(c, seed, torch.float64)
    table = wire_table(c.harmonics, torch.float64)
    x = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(seed))
    base = forward(c, p, table, x)
    for t in range(x.shape[1] - 1):
        changed = x.clone()
        changed[:, t + 1:] = (changed[:, t + 1:] + 1 + t) % 256
        other = forward(c, p, table, changed)
        if not torch.equal(base[:, :t + 1], other[:, :t + 1]):
            raise SystemExit(f'position {t} depends on a later byte')


def check_against_harmonic_gpt(c, root):
    """Logits and the wire table against harmonic-gpt's modules on the same weights."""
    sys.path[:0] = [str(root), str(root / 'scripts')]
    import harmonic_gpt.layers.spectral_kan_attention as ska
    from harmonic_gpt.layers.wire_encoding import encode
    ska._get_compiled_cumsum = lambda: ska._gated_cumsum_loop
    ska._get_compiled_cumsum_vectorized = lambda: ska._gated_cumsum_vectorized
    from wire_lm_variants import SpectralKANUnifiedWireLM
    full = encode(torch.arange(256).unsqueeze(0), n_harmonics=FULL.harmonics).squeeze(0)
    theirs = torch.stack([full.real, full.imag], dim=-1).reshape(256, FULL.d)
    ours = wire_table(FULL.harmonics)
    table_difference = (theirs.double() - ours.double()).abs().max().item()
    if table_difference > 1e-6:
        raise SystemExit(f'the wire table differs from harmonic-gpt\'s by {table_difference}')
    torch.manual_seed(3)
    model = SpectralKANUnifiedWireLM(n_harmonics=c.harmonics, n_layers=c.layers, n_heads=c.heads,
                                     n_kan_harmonics=c.kan, dropout=0.0, dt=c.dt, gate_init=0.0,
                                     use_triton_kernel=False, spatial_kind='planar_givens',
                                     cumsum_mode='vectorized').double().eval()
    with torch.no_grad():
        for t in model.parameters():
            t.add_(0.05 * torch.randn_like(t))
    state = {k: v.detach().clone() for k, v in model.state_dict().items() if k != 'encoder.lut'}
    if set(state) != set(parameter_shapes(c)):
        raise SystemExit(f'parameter names differ: {sorted(set(state) ^ set(parameter_shapes(c)))}')
    for name, shape in parameter_shapes(c).items():
        if tuple(state[name].shape) != shape:
            raise SystemExit(f'{name}: harmonic-gpt {tuple(state[name].shape)}, here {shape}')
    x = torch.randint(0, 256, (2, 48), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        theirs = model(x)
        ours = forward(c, state, model.encoder.lut.double(), x)
    logit_difference = (theirs - ours).abs().max().item()
    if logit_difference > 1e-9:
        raise SystemExit(f'logits differ from harmonic-gpt\'s by {logit_difference}')
    return table_difference, logit_difference


def check_update_against_harmonic_gpt(c, root):
    """Four updates against harmonic-gpt's own optimizers, built as make_optimizers builds them.

    harmonic-gpt's Muon computes in binary32 whatever the parameters' type, so
    the projections it updates agree to binary32. The rest agree to 1e-8 of
    their largest entry: the two recurrences' gradients differ in their last
    digits (its cumulative sum against this stepped scan), and Adam's division
    by the root of the second moment enlarges that on near-zero gradients.
    """
    if str(root) not in sys.path:
        sys.path[:0] = [str(root), str(root / 'scripts')]
    from harmonic_gpt.optim.riemannian_muon import RiemannianMuon
    from harmonic_gpt.optim.rotational_adamw import RotationalAdamW
    from harmonic_gpt.utils.training import DYNAMICAL_PARAM_KEYWORDS
    from wire_lm_variants import SpectralKANUnifiedWireLM
    if tuple(DYNAMICAL_PARAM_KEYWORDS) != DYNAMICAL_KEYWORDS:
        raise SystemExit('harmonic-gpt\'s dynamical keywords have changed')
    torch.manual_seed(6)
    model = SpectralKANUnifiedWireLM(n_harmonics=c.harmonics, n_layers=c.layers, n_heads=c.heads,
                                     n_kan_harmonics=c.kan, dropout=0.0, dt=c.dt, gate_init=0.0,
                                     use_triton_kernel=False, spatial_kind='planar_givens',
                                     cumsum_mode='vectorized').double().train()
    raw = {k: v.detach().clone() for k, v in model.named_parameters()}
    muon, decay, rest = [], [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and 'attn.out_proj.weight' in name:
            muon.append(p)
        elif p.dim() <= 1 or any(k in name for k in DYNAMICAL_PARAM_KEYWORDS):
            rest.append(p)
        else:
            decay.append(p)
    opt_muon = RiemannianMuon(muon, lr=PEAK_LR * MUON_RATIO, momentum=MOMENTUM, ns_steps=5, weight_decay=0.0,
                              ns_variant='canonical_quintic', shape_compensation=True, nesterov=True)
    opt_rest = RotationalAdamW([{'params': decay, 'weight_decay': WEIGHT_DECAY},
                                {'params': rest, 'weight_decay': 0.0}], lr=PEAK_LR)
    theirs = dict(model.named_parameters())
    centred = centre_rotational(dict(raw))
    centring = max((centred[k] - theirs[k]).abs().max().item() for k in raw)
    if centring > 1e-12:
        raise SystemExit(f'initial row centring differs from harmonic-gpt\'s by {centring}')
    params = {k: v.detach().clone() for k, v in theirs.items()}
    update = Update(params, total=8)
    table = model.encoder.lut.double()
    gen = torch.Generator().manual_seed(7)
    worst = {'muon': 0.0, 'rotational': 0.0, 'adam': 0.0}
    for step in range(4):
        seq = torch.randint(0, 256, (2, 25), generator=gen)
        x, y = seq[:, :-1], seq[:, 1:]
        for o in (opt_muon, opt_rest):
            o.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        s = schedule(step, update.total)
        for group in opt_muon.param_groups:
            group['lr'] = PEAK_LR * s * MUON_RATIO
        for group in opt_rest.param_groups:
            group['lr'] = PEAK_LR * s
        opt_muon.step()
        opt_rest.step()
        leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        loss(c, leaves, table, x, y).backward()
        update(params, {k: v.grad for k, v in leaves.items()})
        for k, v in params.items():
            kind = role(k, v.shape)
            worst[kind] = max(worst[kind], ((v - theirs[k].detach()).abs().max() / v.abs().max()).item())
    if worst['muon'] > 1e-5 or worst['rotational'] > 1e-8 or worst['adam'] > 1e-8:
        raise SystemExit(f'updates differ from harmonic-gpt\'s: {worst}')
    return worst


# ---- the bank (Diakosmos.Model) ----
ALIGN = 64
STRIDE = 1344


def after(offset, extent):
    return ALIGN * ((offset + extent + ALIGN - 1) // ALIGN)


def bank_layout(c=FULL):
    """Each tensor's word offset: Diakosmos.Model's layout (checked against its proofs)."""
    P, HC, HW = 3 * STRIDE, c.heads * c.kan, c.heads * c.kan * c.head_width
    o = {}
    o['temporal.ln.weight'] = 0
    o['temporal.ln.bias'] = after(0, STRIDE)
    o['temporal.attn.qkv_proj.weight'] = after(o['temporal.ln.bias'], STRIDE)
    o['temporal.attn.qkv_proj.bias'] = after(o['temporal.attn.qkv_proj.weight'], P * STRIDE)
    o['temporal.attn.q_harmonic_weight'] = after(o['temporal.attn.qkv_proj.bias'], P)
    o['temporal.attn.k_harmonic_weight'] = after(o['temporal.attn.q_harmonic_weight'], HW)
    o['temporal.attn.spectral_coeffs'] = after(o['temporal.attn.k_harmonic_weight'], HW)
    o['temporal.attn._gate_logit'] = after(o['temporal.attn.spectral_coeffs'], HC)
    o['temporal.attn._harmonic_decay_offset'] = after(o['temporal.attn._gate_logit'], c.heads)
    o['temporal.attn.out_proj.weight'] = after(o['temporal.attn._harmonic_decay_offset'], c.kan)
    o['temporal.attn.out_proj.bias'] = after(o['temporal.attn.out_proj.weight'], STRIDE * STRIDE)
    o['spatial.theta'] = after(o['temporal.attn.out_proj.bias'], STRIDE)
    o['spatial.coupling'] = after(o['spatial.theta'], (GROUPS // 2) * c.width)
    o['gate_logit'] = after(o['spatial.coupling'], GROUPS)
    block = after(o['gate_logit'], GROUPS)
    final = c.layers * block
    head = after(after(final, STRIDE), STRIDE)
    table = after(head, 256 * STRIDE)
    words = 512 * ((table + 256 * STRIDE + 511) // 512)
    return {'block': o, 'block_words': block, 'ln_f.weight': final, 'ln_f.bias': after(final, STRIDE),
            'head.weight': head, 'table': table, 'words': words}


def padded_rows(t, rows, groups=1):
    """A [n][1330] (or [n]) tensor's rows placed in groups of `rows` padded rows of 1344 words."""
    import numpy as np
    t = t.detach().double().numpy()
    if t.ndim == 1:
        out = np.zeros(groups * rows)
        per = t.size // groups
        for g in range(groups):
            out[g * rows:g * rows + per] = t[g * per:(g + 1) * per]
        return out
    out = np.zeros((groups * rows, STRIDE))
    per = t.shape[0] // groups
    for g in range(groups):
        out[g * rows:g * rows + per, :t.shape[1]] = t[g * per:(g + 1) * per]
    return out


def write_bank(c, p, table):
    """The bank's binary32 words from the parameters and the wire table."""
    import numpy as np
    L = bank_layout(c)
    bank = np.zeros(L['words'], dtype=np.float32)

    def put(offset, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        bank[offset:offset + values.size] = values

    vectors = ('temporal.ln.weight', 'temporal.ln.bias', 'temporal.attn.out_proj.bias')
    for i in range(c.layers):
        base = i * L['block_words']
        for name, at in L['block'].items():
            t = p[f'blocks.{i}.{name}']
            if name in vectors:
                put(base + at, padded_rows(t, STRIDE))
            elif name == 'temporal.attn.qkv_proj.weight':
                put(base + at, padded_rows(t, STRIDE, 3))
            elif name == 'temporal.attn.qkv_proj.bias':
                put(base + at, padded_rows(t, STRIDE, 3))
            elif name == 'temporal.attn.out_proj.weight':
                put(base + at, padded_rows(t, STRIDE))
            else:
                put(base + at, t.detach().double().numpy())
    put(L['ln_f.weight'], padded_rows(p['ln_f.weight'], STRIDE))
    put(L['ln_f.bias'], padded_rows(p['ln_f.bias'], STRIDE))
    put(L['head.weight'], p['head.weight'].detach().double().numpy())
    put(L['table'], padded_rows(table, 256))
    return bank


def read_bank(c, bank):
    """The parameters and the wire table back from the bank (binary32)."""
    L = bank_layout(c)
    shapes = parameter_shapes(c)
    W = c.width * GROUPS
    p = {}

    def take(offset, count):
        return torch.from_numpy(bank[offset:offset + count].copy())

    for i in range(c.layers):
        base = i * L['block_words']
        for name, at in L['block'].items():
            key = f'blocks.{i}.{name}'
            shape = shapes[key]
            if name in ('temporal.ln.weight', 'temporal.ln.bias', 'temporal.attn.out_proj.bias'):
                p[key] = take(base + at, W)
            elif name == 'temporal.attn.qkv_proj.weight':
                m = take(base + at, 3 * STRIDE * STRIDE).view(3, STRIDE, STRIDE)[:, :W, :W]
                p[key] = m.reshape(3 * W, W).clone()
            elif name == 'temporal.attn.qkv_proj.bias':
                p[key] = take(base + at, 3 * STRIDE).view(3, STRIDE)[:, :W].reshape(-1).clone()
            elif name == 'temporal.attn.out_proj.weight':
                p[key] = take(base + at, STRIDE * STRIDE).view(STRIDE, STRIDE)[:W, :W].clone()
            else:
                p[key] = take(base + at, math.prod(shape)).view(shape).clone()
    p['ln_f.weight'] = take(L['ln_f.weight'], W)
    p['ln_f.bias'] = take(L['ln_f.bias'], W)
    p['head.weight'] = take(L['head.weight'], 256 * STRIDE).view(256, STRIDE).clone()
    table = take(L['table'], 256 * STRIDE).view(256, STRIDE)[:, :W].clone()
    return p, table


def score(c, p, table, windows, half=True, batch=4):
    """Every position's next-byte loss: windows [n][T + 1] -> [n][T] (binary32)."""
    out = []
    with torch.no_grad():
        for i in range(0, windows.shape[0], batch):
            w = windows[i:i + batch]
            logits = forward(c, p, table, w[:, :-1], half=half)
            out.append(torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256), w[:, 1:].reshape(-1), reduction='none').view(w.shape[0], -1))
            print(f'scored {min(i + batch, windows.shape[0])} of {windows.shape[0]} windows', flush=True)
    return torch.cat(out)


# ---- the optimizer's state as the device keeps it ----
# m and v as banks (Muon's momentum in m), and for each rotational row (a
# block's 4032 padded qkv rows, then the head's 256) its first norm |p0| and
# the running mean of its direction's squared norm: two planes of ROWS words.
QKV_ROWS = 3 * STRIDE


def row_count(c):
    return c.layers * QKV_ROWS + 256


def rotational_rows(c, name, n):
    """The padded row indices of a rotational tensor's n rows."""
    import numpy as np
    if name == 'head.weight':
        return np.arange(n) + c.layers * QKV_ROWS
    block = int(name.split('.')[1])
    j = np.arange(n)
    return block * QKV_ROWS + (j // (n // 3)) * STRIDE + j % (n // 3)


def state_files(c, params, upd):
    """The update's state as m, v (banks) and the rows' two planes."""
    import numpy as np
    R = row_count(c)
    m, v = {}, {}
    rows = np.zeros(2 * R)
    for name, p in params.items():
        st = upd.state[name]
        zero = torch.zeros_like(p)
        if role(name, p.shape) == 'muon':
            m[name], v[name] = st.get('momentum', zero), zero
        else:
            m[name], v[name] = st.get('m', zero), st.get('v', zero)
        if role(name, p.shape) == 'rotational' and 'norm' in st:
            at = rotational_rows(c, name, p.shape[0])
            rows[at] = st['norm'].reshape(-1).double().numpy()
            rows[R + at] = st['update'].reshape(-1).double().numpy()
    table = torch.zeros(256, c.width * GROUPS)
    return write_bank(c, m, table), write_bank(c, v, table), rows


def load_state(c, params, upd, state, step):
    """The update's state from state_files' m, v and rows (after `step` updates)."""
    import numpy as np
    L = bank_layout(c)
    R = row_count(c)
    m, _ = read_bank(c, state[:L['words']])
    v, _ = read_bank(c, state[L['words']:2 * L['words']])
    rows = state[2 * L['words']:2 * L['words'] + 2 * R].astype(np.float64)
    upd.step = step
    if step == 0:
        return
    for name, p in params.items():
        st = upd.state[name]
        kind = role(name, p.shape)
        if kind == 'muon':
            st['momentum'] = m[name].to(p.dtype)
            continue
        st['m'], st['v'] = m[name].to(p.dtype), v[name].to(p.dtype)
        if kind == 'rotational':
            at = rotational_rows(c, name, p.shape[0])
            shape = (p.shape[0],) + (1,) * (p.dim() - 1)
            st['norm'] = torch.from_numpy(rows[at]).to(p.dtype).view(shape)
            st['update'] = torch.from_numpy(rows[R + at]).to(p.dtype).view(shape)
            st['rotations'] = step


def report_change(name, d, r):
    import numpy as np
    scale = max(np.abs(r).max(), 1e-30)
    cos = float(d @ r / max(np.linalg.norm(d) * np.linalg.norm(r), 1e-300))
    print(f'{name:44s} |ref|max {scale:.3e}  max|diff|/|ref|max {np.abs(d - r).max() / scale:.3e}  cosine {cos:.8f}')


def tensor_ranges(c):
    L = bank_layout(c)
    out = []
    for i in range(c.layers):
        base = i * L['block_words']
        names = sorted(L['block'].items(), key=lambda kv: kv[1])
        for j, (name, at) in enumerate(names):
            end = names[j + 1][1] if j + 1 < len(names) else L['block_words']
            out.append((f'blocks.{i}.{name}', base + at, base + end))
    out += [('ln_f.weight', L['ln_f.weight'], L['ln_f.bias']), ('ln_f.bias', L['ln_f.bias'], L['head.weight']),
            ('head.weight', L['head.weight'], L['table'])]
    return out


# ---- a training checkpoint (Diakosmos.Build.CheckpointTransfer) ----
# master, m, v, the rows' two planes, the schedule (16 binary32 words an
# update: OptimizerSM86's constants row), the learner's state (u32 word 0:
# the updates made; 1024 bytes)
SCHEDULE_CAPACITY = 24576
SCHEDULE_ROW = 16
STATE_BYTES = 1024


def schedule_rows(total, peak=PEAK_LR, warmup=WARMUP, min_fraction=MIN_FRACTION):
    """Every update's scalars, computed in binary64 as Update computes them, rounded once.
    The run's schedule by default; a fine-tune's with its own peak and warmup."""
    import numpy as np
    if total > SCHEDULE_CAPACITY:
        raise SystemExit(f'{total} updates: the schedule holds {SCHEDULE_CAPACITY}')
    rows = np.zeros((SCHEDULE_CAPACITY, SCHEDULE_ROW))
    for u in range(total):
        s = schedule(u, total, warmup, min_fraction)
        lr = peak * s
        t = u + 1
        bc1, bc2 = 1 - BETA1 ** t, 1 - BETA2 ** t
        root = math.sqrt(bc2)
        rows[u, :9] = (lr, peak * MUON_RATIO * s, lr * root / bc1, ADAM_EPS * root, -root / bc1, ADAM_EPS * root,
                       math.sqrt(2 * lr * WEIGHT_DECAY * (1 - BETA1) / (1 + BETA1)), 1 / (1 - 0.9 ** t),
                       1.0 if u == 0 else 0.0)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('table')
    t.add_argument('output', type=Path)
    k = sub.add_parser('check')
    k.add_argument('--harmonic-gpt', type=Path)
    k.add_argument('--full', action='store_true')
    ck = sub.add_parser('checkpoint')
    ck.add_argument('--seed', type=int, default=0)
    ck.add_argument('--perturb', type=float, default=0.0,
                    help='add perturb x max(std, 0.1) x N(0, 1) to every tensor: a stress test, larger activations')
    ck.add_argument('output', type=Path)
    wi = sub.add_parser('windows')
    wi.add_argument('--bytes', type=Path, required=True)
    wi.add_argument('--count', type=int, required=True)
    wi.add_argument('output', type=Path)
    sc = sub.add_parser('score')
    sc.add_argument('--payload', type=Path, required=True)
    sc.add_argument('--windows', type=Path, required=True)
    sc.add_argument('--exact', action='store_true')
    sc.add_argument('output', type=Path)
    gr = sub.add_parser('gradient')
    gr.add_argument('--payload', type=Path, required=True)
    gr.add_argument('--windows', type=Path, required=True)
    gr.add_argument('--exact', action='store_true')
    gr.add_argument('output', type=Path)
    cg = sub.add_parser('compare-gradient')
    cg.add_argument('device', type=Path, help='the device gradient bank (binary32 words)')
    cg.add_argument('reference', type=Path)
    up = sub.add_parser('update')
    up.add_argument('--payload', type=Path, required=True)
    up.add_argument('--gradient', type=Path, required=True, help='a gradient bank (the mean over the update)')
    up.add_argument('--state', type=Path, help='the state after STEP updates (state_files: m, v, rows)')
    up.add_argument('--step', type=int, default=0, help='updates already made')
    up.add_argument('--total', type=int, required=True, help='the run\'s updates (the schedule\'s length)')
    up.add_argument('--exact', action='store_true', help='Newton-Schulz without the bfloat16 rounding')
    up.add_argument('output', type=Path, help='the payload after the update')
    up.add_argument('state_output', type=Path)
    cu = sub.add_parser('compare-update')
    cu.add_argument('--before', type=Path, required=True, help='the payload before the update')
    cu.add_argument('device', type=Path)
    cu.add_argument('reference', type=Path)
    cu.add_argument('--device-state', type=Path)
    cu.add_argument('--reference-state', type=Path)
    tc = sub.add_parser('training-checkpoint')
    tc.add_argument('--payload', type=Path, required=True, help='a weights payload (the master)')
    tc.add_argument('--state', type=Path, help='an update\'s state (m, v, rows) to start from')
    tc.add_argument('--step', type=int, default=0, help='the updates made')
    tc.add_argument('--total', type=int, required=True, help='the run\'s updates')
    tc.add_argument('--peak-lr', type=float, default=PEAK_LR, help='the schedule\'s peak (a fine-tune\'s)')
    tc.add_argument('--warmup', type=int, default=WARMUP, help='its linear warmup in updates')
    tc.add_argument('output', type=Path)
    tp = sub.add_parser('training-parts')
    tp.add_argument('checkpoint', type=Path, help='a training checkpoint (its envelope\'s header, then the payload)')
    tp.add_argument('payload', type=Path, help='its master, as a weights payload')
    tp.add_argument('state', type=Path, help='its m, v and rows, as update writes a state')
    gn = sub.add_parser('generate')
    gn.add_argument('--payload', type=Path, required=True)
    gn.add_argument('--prompt', required=True)
    gn.add_argument('--count', type=int, default=32)
    cm = sub.add_parser('compare')
    cm.add_argument('device', type=Path)
    cm.add_argument('reference', type=Path)
    args = p.parse_args()
    import numpy as np
    if args.command == 'checkpoint':
        params = initialize(FULL, args.seed)
        if args.perturb:
            gen = torch.Generator().manual_seed(args.seed + 1)
            for name, t in params.items():
                scale = max(t.std().item() if t.numel() > 1 else 0.0, 0.1)
                params[name] = t + args.perturb * scale * torch.randn(t.shape, generator=gen)
        bank = write_bank(FULL, params, wire_table(FULL.harmonics))
        tmp = args.output.with_name(args.output.name + '.partial')
        bank.astype('<f4').tofile(tmp)
        tmp.replace(args.output)
        import hashlib
        print(args.output, bank.size, 'words', hashlib.sha256(args.output.read_bytes()).hexdigest())
        return
    if args.command == 'windows':
        data = np.fromfile(args.bytes, dtype=np.uint8)
        T = FULL_CONTEXT
        if args.count * T + 1 > data.size:
            raise SystemExit('too few bytes for the windows asked for')
        out = np.stack([data[k * T:k * T + T + 1] for k in range(args.count)]).astype('<u4')
        out.tofile(args.output)
        print(args.output, out.shape)
        return
    if args.command == 'score':
        bank = np.fromfile(args.payload, dtype='<f4')
        if bank.size != bank_layout(FULL)['words']:
            raise SystemExit(f'the payload is {bank.size} words, the bank {bank_layout(FULL)["words"]}')
        params, table = read_bank(FULL, bank)
        windows = torch.from_numpy(np.fromfile(args.windows, dtype='<u4').astype(np.int64)).view(-1, FULL_CONTEXT + 1)
        losses = score(FULL, params, table, windows, half=not args.exact)
        losses.numpy().astype('<f4').tofile(args.output)
        print(f'{args.output}: mean {losses.mean().item():.6f} nats, {losses.mean().item() / math.log(2):.6f} bits a byte')
        return
    if args.command == 'gradient':
        bank = np.fromfile(args.payload, dtype='<f4')
        params, table = read_bank(FULL, bank)
        windows = torch.from_numpy(np.fromfile(args.windows, dtype='<u4').astype(np.int64)).view(-1, FULL_CONTEXT + 1)
        for t in params.values():
            t.requires_grad_(True)
        total = None
        for i in range(0, windows.shape[0], 4):
            w = windows[i:i + 4]
            logits = forward(FULL, params, table, w[:, :-1], half=not args.exact)
            part = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), w[:, 1:].reshape(-1), reduction='sum')
            (part / windows[:, 1:].numel()).backward()
            total = part.item() if total is None else total + part.item()
            print(f'pulled back {min(i + 4, windows.shape[0])} of {windows.shape[0]} windows', flush=True)
        grads = {k: v.grad for k, v in params.items()}
        out = write_bank(FULL, grads, torch.zeros_like(table))
        out.astype('<f4').tofile(args.output)
        print(f'{args.output}: mean loss {total / windows[:, 1:].numel():.6f}, gradient norm '
              f'{math.sqrt(sum(float((g.double() ** 2).sum()) for g in grads.values())):.6e}')
        return
    if args.command == 'update':
        L = bank_layout(FULL)
        bank = np.fromfile(args.payload, dtype='<f4')
        params, table = read_bank(FULL, bank)
        grads, _ = read_bank(FULL, np.fromfile(args.gradient, dtype='<f4'))
        params = {k: v.double() for k, v in params.items()}
        grads = {k: v.double() for k, v in grads.items()}
        upd = Update(params, args.total, ns=newton_schulz if args.exact else newton_schulz_half)
        if args.state:
            load_state(FULL, params, upd, np.fromfile(args.state, dtype='<f4'), args.step)
        elif args.step:
            raise SystemExit('a state is needed after the first update')
        norm = upd(params, grads)
        out = write_bank(FULL, params, table)
        out.astype('<f4').tofile(args.output)
        m, v, rows = state_files(FULL, params, upd)
        np.concatenate([m, v, rows]).astype('<f4').tofile(args.state_output)
        print(f'update {upd.step} of {args.total}: gradient norm {norm:.6e}, schedule {schedule(upd.step - 1, args.total):.6f}')
        return
    if args.command == 'generate':
        params, table = read_bank(FULL, np.fromfile(args.payload, dtype='<f4'))
        ids = list(args.prompt.encode('utf-8'))
        with torch.no_grad():
            for _ in range(args.count):
                x = torch.tensor([ids[-FULL_CONTEXT:]], dtype=torch.long)
                ids.append(int(forward(FULL, params, table, x, half=True)[0, -1].argmax()))
        print(json.dumps(bytes(ids[len(args.prompt.encode('utf-8')):]).decode('utf-8', errors='replace')))
        return
    if args.command == 'training-checkpoint':
        L = bank_layout(FULL)
        W, R = L['words'], row_count(FULL)
        master = np.fromfile(args.payload, dtype='<f4')
        if master.size != W:
            raise SystemExit(f'the payload is {master.size} words, the bank {W}')
        state = np.fromfile(args.state, dtype='<f4') if args.state else np.zeros(2 * W + 2 * R, dtype='<f4')
        if state.size != 2 * W + 2 * R:
            raise SystemExit(f'the state is {state.size} words, not {2 * W + 2 * R}')
        if args.step and not args.state:
            raise SystemExit('a state is needed after the first update')
        learner = np.zeros(STATE_BYTES // 4, dtype='<u4')
        learner[0] = args.step
        tmp = args.output.with_name(args.output.name + '.partial')
        with open(tmp, 'wb') as out:
            out.write(master.astype('<f4').tobytes())
            out.write(state.astype('<f4').tobytes())
            out.write(schedule_rows(args.total, args.peak_lr, args.warmup).astype('<f4').tobytes())
            out.write(learner.tobytes())
        tmp.replace(args.output)
        print(f'{args.output}: {args.output.stat().st_size} bytes, update {args.step} of {args.total}')
        return
    if args.command == 'training-parts':
        L = bank_layout(FULL)
        W, R = L['words'], row_count(FULL)
        raw = np.fromfile(args.checkpoint, dtype=np.uint8)
        header = int(raw[16:24].view('<u8')[0])
        words = raw[header:].view('<f4')
        if words.size * 4 != 3 * W * 4 + 2 * R * 4 + SCHEDULE_CAPACITY * SCHEDULE_ROW * 4 + STATE_BYTES:
            raise SystemExit(f'{args.checkpoint}: a payload of {words.size * 4} bytes is not a training checkpoint\'s')
        words[:W].tofile(args.payload)
        words[W:3 * W + 2 * R].tofile(args.state)
        print(f'{args.checkpoint}: header {header} bytes; updates made {int(raw[header:].view("<u4")[(3 * W + 2 * R + SCHEDULE_CAPACITY * SCHEDULE_ROW)])}')
        return
    if args.command == 'compare-update':
        L = bank_layout(FULL)
        before = np.fromfile(args.before, dtype='<f4').astype(np.float64)
        device = np.fromfile(args.device, dtype='<f4').astype(np.float64)
        reference = np.fromfile(args.reference, dtype='<f4').astype(np.float64)
        print('each tensor\'s change')
        for name, lo, hi in tensor_ranges(FULL):
            report_change(name, device[lo:hi] - before[lo:hi], reference[lo:hi] - before[lo:hi])
        table = slice(L['table'], L['words'])
        if np.abs(device[table] - before[table]).max() != 0:
            raise SystemExit('the device changed the wire table')
        if args.device_state and args.reference_state:
            ds = np.fromfile(args.device_state, dtype='<f4').astype(np.float64)
            rs = np.fromfile(args.reference_state, dtype='<f4').astype(np.float64)
            W, R = L['words'], row_count(FULL)
            for label, base in (('m', 0), ('v', W)):
                for name, lo, hi in tensor_ranges(FULL):
                    report_change(f'{label} {name}', ds[base + lo:base + hi], rs[base + lo:base + hi])
            report_change('rows |p0|', ds[2 * W:2 * W + R], rs[2 * W:2 * W + R])
            report_change('rows running mean', ds[2 * W + R:2 * W + 2 * R], rs[2 * W + R:2 * W + 2 * R])
        return
    if args.command == 'compare-gradient':
        device = np.fromfile(args.device, dtype='<f4').astype(np.float64)
        reference = np.fromfile(args.reference, dtype='<f4').astype(np.float64)
        L = bank_layout(FULL)
        if device.size < L['table'] or reference.size < L['table']:
            raise SystemExit('a gradient bank is short')
        def report(name, lo, hi):
            d, r = device[lo:hi], reference[lo:hi]
            scale = max(np.abs(r).max(), 1e-30)
            cos = float(d @ r / max(np.linalg.norm(d) * np.linalg.norm(r), 1e-300))
            print(f'{name:44s} |ref|max {scale:.3e}  max|diff|/|ref|max {np.abs(d - r).max() / scale:.3e}  cosine {cos:.8f}')
        for i in range(FULL.layers):
            base = i * L['block_words']
            names = sorted(L['block'].items(), key=lambda kv: kv[1])
            for j, (name, at) in enumerate(names):
                end = names[j + 1][1] if j + 1 < len(names) else L['block_words']
                report(f'blocks.{i}.{name}', base + at, base + end)
        report('ln_f.weight', L['ln_f.weight'], L['ln_f.bias'])
        report('ln_f.bias', L['ln_f.bias'], L['head.weight'])
        report('head.weight', L['head.weight'], L['table'])
        return
    if args.command == 'compare':
        raw = args.device.read_bytes()
        device = np.frombuffer(raw[:len(raw) - 160], dtype='<f4').astype(np.float64)
        reference = np.fromfile(args.reference, dtype='<f4').astype(np.float64)
        if device.size != reference.size:
            raise SystemExit(f'{device.size} device losses, {reference.size} reference')
        d = np.abs(device - reference)
        print(f'{device.size} positions: device mean {device.mean():.6f}, reference {reference.mean():.6f} nats; '
              f'|difference| max {d.max():.3e}, mean {d.mean():.3e}, at the 99.9th percentile {np.quantile(d, 0.999):.3e}')
        return
    if args.command == 'table':
        table = wire_table(FULL.harmonics)
        tmp = args.output.with_name(args.output.name + '.partial')
        table.numpy().astype('<f4').tofile(tmp)
        tmp.replace(args.output)
        import hashlib
        print(args.output, table.shape, hashlib.sha256(args.output.read_bytes()).hexdigest())
        return
    small = Config(harmonics=2, layers=2, heads=7, kan=3)
    print(f'gradient: worst relative difference from central differences {check_gradient(small):.2e}', flush=True)
    check_causality(small)
    print('causality: every position independent of later bytes', flush=True)
    if args.harmonic_gpt:
        table_difference, logit_difference = check_against_harmonic_gpt(Config(harmonics=3, layers=3, kan=4), args.harmonic_gpt)
        print(f'harmonic-gpt: wire table within {table_difference:.2e}, logits within {logit_difference:.2e}', flush=True)
        worst = check_update_against_harmonic_gpt(Config(harmonics=3, layers=3, kan=4), args.harmonic_gpt)
        print('harmonic-gpt: four updates within ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items())
              + ' (largest difference over largest entry)', flush=True)
    if args.full:
        import time
        count = sum(math.prod(s) for s in parameter_shapes(FULL).values())
        print(f'parameters: {count:,}', flush=True)
        params = initialize(FULL)
        table = wire_table(FULL.harmonics)
        seq = torch.randint(0, 256, (1, 2049), generator=torch.Generator().manual_seed(5))
        start = time.time()
        with torch.no_grad():
            value = loss(FULL, params, table, seq[:, :-1], seq[:, 1:]).item()
        print(f'100M forward, 2,048 bytes: loss {value:.4f} ({value / math.log(2):.4f} bits a byte) '
              f'in {time.time() - start:.1f} s', flush=True)


if __name__ == '__main__':
    main()
