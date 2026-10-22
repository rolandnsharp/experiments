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
GMIN = 0.0005          # Ambrosio's decay floor: |c| <= e^-gmin
RADIUS = 0.9995        # Manfred's projection radius
RATE = 16000
F_LOW, F_HIGH = 60.0, 7000.0
# the continuous-time variant: r = exp(-dt (1/TAU_MAX + 1/tau)), theta = 2 pi dt f
TAU_MAX = 4.0                     # the decay floor as a time: nothing rings longer than 4 s
TAU_LOW, TAU_HIGH = 1e-4, 2.0     # initial decay times, log-spaced over the bank
Q_INIT = 2.0                      # initial f = Q / (pi tau): a 2 s pole at 0.3 Hz, a 0.1 ms pole at 6.4 kHz


# ---- parameters ----
def make_params(c, variant, seed=0, dtype=torch.float32):
    p = O.initialize(c, seed, torch.float64)
    H, L, dh = c.heads, c.kan, c.head_width
    g = torch.Generator().manual_seed(seed + 7)
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
    return {k: v.to(dtype) for k, v in p.items()}


def pole(p, a, variant):
    """(log r, theta) per [H][L]."""
    if variant == 'real':
        logit = p[a + '_gate_logit'][:, None] + p[a + '_harmonic_decay_offset'][None, :]
        return F.logsigmoid(logit), torch.zeros_like(logit)
    if variant == 'polar':
        return -(GMIN + torch.exp(p[a + 'pole_decay_rho'])), p[a + 'pole_omega']
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


# ---- the recurrence, chunked ----
def powers(logr, theta, n):
    """c^j for j = 0 .. n-1: ([H][L][n] real, imaginary)."""
    j = torch.arange(n, dtype=logr.dtype, device=logr.device)
    mag = torch.exp(logr[..., None] * j)
    ang = theta[..., None] * j
    return mag * torch.cos(ang), mag * torch.sin(ang)


def complex_scan(ur, ui, logr, theta):
    """S_n = c S_{n-1} + u_n for every n, u [B][N][P] (real, imaginary), c = exp(logr + i theta) per pole [P].
    Chunks of CHUNK positions are solved in closed form; the chunk-end states are one scan of 1/CHUNK the length."""
    B, N, P = ur.shape
    C = CHUNK
    pad = (-N) % C
    if pad:
        ur = F.pad(ur, (0, 0, 0, pad))
        ui = F.pad(ui, (0, 0, 0, pad))
    n = ur.shape[1] // C
    ur, ui = ur.reshape(B, n, C, P), ui.reshape(B, n, C, P)
    pre, pim = powers(logr, theta, C + 1)                   # [P][C+1]
    t = torch.arange(C, device=ur.device)
    lag = (t[:, None] - t[None, :]).clamp(min=0)
    causal = (t[:, None] >= t[None, :]).to(ur.dtype)
    Pre, Pim = pre[:, lag] * causal, pim[:, lag] * causal   # [P][C][C]
    wr = torch.einsum('bnsp,pts->bntp', ur, Pre) - torch.einsum('bnsp,pts->bntp', ui, Pim)
    wi = torch.einsum('bnsp,pts->bntp', ur, Pim) + torch.einsum('bnsp,pts->bntp', ui, Pre)
    if n > 1:
        er, ei = pre[:, C - 1 - t], pim[:, C - 1 - t]       # c^(C-1-s)
        endr = torch.einsum('bnsp,ps->bnp', ur, er) - torch.einsum('bnsp,ps->bnp', ui, ei)
        endi = torch.einsum('bnsp,ps->bnp', ur, ei) + torch.einsum('bnsp,ps->bnp', ui, er)
        cr, ci = complex_scan(endr[:, :-1], endi[:, :-1], logr * C, theta * C)
        cr, ci = F.pad(cr, (0, 0, 1, 0)), F.pad(ci, (0, 0, 1, 0))
        sr, si = pre[:, 1:C + 1], pim[:, 1:C + 1]           # c^(t+1), [P][C]
        wr = wr + sr.T[None, None] * cr[:, :, None] - si.T[None, None] * ci[:, :, None]
        wi = wi + sr.T[None, None] * ci[:, :, None] + si.T[None, None] * cr[:, :, None]
    wr, wi = wr.reshape(B, n * C, P), wi.reshape(B, n * C, P)
    return wr[:, :N], wi[:, :N]


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
        qi = q_im.view(B, n, C, H, L)
        A = A + torch.einsum('bnthl,bnshl,hlts->bnhts', qi, kc, Pim)
    read = torch.einsum('bnhts,bnshd->bnthd', A, vc)
    if n > 1:
        # the state at each chunk end: S_i = c^C S_{i-1} + sum_s c^(C-1-s) k_s v_s, one complex scan over the chunks
        end_re, end_im = pre[:, :, C - 1 - t], pim[:, :, C - 1 - t]      # [H][L][C]
        inj_re = torch.einsum('bnshl,hls,bnshd->bnhld', kc, end_re, vc).reshape(B, n, H * L * dh)
        inj_im = torch.einsum('bnshl,hls,bnshd->bnhld', kc, end_im, vc).reshape(B, n, H * L * dh)
        lr = (logr * C)[:, :, None].expand(H, L, dh).reshape(-1)
        th = (theta * C)[:, :, None].expand(H, L, dh).reshape(-1)
        Sre, Sim = complex_scan(inj_re[:, :-1], inj_im[:, :-1], lr, th)
        Sre = F.pad(Sre, (0, 0, 1, 0)).view(B, n, H, L, dh)             # the state entering each chunk
        Sim = F.pad(Sim, (0, 0, 1, 0)).view(B, n, H, L, dh)
        # read_t += Re( conj(q~_t) c^(t+1) S_prev ) = sum_l (q_re W_re + q_im W_im), W = c^(t+1) S_prev
        step_re, step_im = pre[:, :, 1:C + 1], pim[:, :, 1:C + 1]          # c^(t+1), [H][L][C]
        carry = (torch.einsum('bnthl,hlt,bnhld->bnthd', qc, step_re, Sre)
                 - torch.einsum('bnthl,hlt,bnhld->bnthd', qc, step_im, Sim))
        if q_im is not None:
            carry = carry + (torch.einsum('bnthl,hlt,bnhld->bnthd', qi, step_re, Sim)
                             + torch.einsum('bnthl,hlt,bnhld->bnthd', qi, step_im, Sre))
        read = read + carry
    return read.reshape(B, T, H, dh)


def block(c, p, prefix, z, variant):
    """The oracle's block with the recurrence above."""
    B, T, D = z.shape
    H, L, dh, W = c.heads, c.kan, c.head_width, c.width
    a = prefix + 'temporal.attn.'
    u = O.layer_norm(z, p[prefix + 'temporal.ln.weight'], p[prefix + 'temporal.ln.bias'])
    qkv = (O.product(u, p[a + 'qkv_proj.weight'], False) + p[a + 'qkv_proj.bias']).view(B, T, 3, H, dh)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    qh = torch.einsum('bthd,hld->bthl', q, p[a + 'q_harmonic_weight'])
    kh = torch.einsum('bthd,hld->bthl', k, p[a + 'k_harmonic_weight']) * p[a + 'spectral_coeffs']
    logr, theta = pole(p, a, variant)
    read = recurrence(qh, kh, v, logr, theta).reshape(B, T, D)
    temporal = O.product(read, p[a + 'out_proj.weight'], False) + p[a + 'out_proj.bias']
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
    return z + (gate * temporal.view(B, T, O.GROUPS, W) + (1 - gate) * spatial).reshape(B, T, D)


def forward(c, p, table, x, variant, checkpoint=False):
    z = table[x]
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
    return O.product(torch.cat([f, cos, length], dim=-1), p['head.weight'], False)


def loss_of(c, p, table, w, variant, checkpoint=False):
    logits = forward(c, p, table, w[:, :-1], variant, checkpoint)
    return F.cross_entropy(logits.reshape(-1, 256), w[:, 1:].reshape(-1))


# ---- generation: one position at a time, the recurrence as written ----
class Sampler:
    def __init__(self, c, p, table, variant):
        self.c, self.p, self.table, self.variant = c, p, table, variant
        H, L, dh = c.heads, c.kan, c.head_width
        self.S = [(torch.zeros(H, L, dh, device=table.device), torch.zeros(H, L, dh, device=table.device)) for _ in range(c.layers)]
        self.poles = [pole(p, f'blocks.{i}.temporal.attn.', variant) for i in range(c.layers)]
        self.sum = torch.zeros(O.GROUPS, c.width, device=table.device)
        self.count = 0

    @torch.no_grad()
    def logits(self, byte):
        c, p = self.c, self.p
        H, L, dh, W = c.heads, c.kan, c.head_width, c.width
        z = self.table[byte]
        for i in range(c.layers):
            prefix, a = f'blocks.{i}.', f'blocks.{i}.temporal.attn.'
            u = O.layer_norm(z, p[prefix + 'temporal.ln.weight'], p[prefix + 'temporal.ln.bias'])
            qkv = (u @ p[a + 'qkv_proj.weight'].T + p[a + 'qkv_proj.bias']).view(3, H, dh)
            q, k, v = qkv[0], qkv[1], qkv[2]
            qh = torch.einsum('hd,hld->hl', q, p[a + 'q_harmonic_weight'])
            kh = torch.einsum('hd,hld->hl', k, p[a + 'k_harmonic_weight']) * p[a + 'spectral_coeffs']
            logr, theta = self.poles[i]
            cre, cim = torch.exp(logr) * torch.cos(theta), torch.exp(logr) * torch.sin(theta)
            Sre, Sim = self.S[i]
            inj = kh[:, :, None] * v[:, None, :]
            Sre, Sim = cre[:, :, None] * Sre - cim[:, :, None] * Sim + inj, cre[:, :, None] * Sim + cim[:, :, None] * Sre
            self.S[i] = (Sre, Sim)
            read = torch.einsum('hl,hld->hd', qh[:, :L], Sre)
            if qh.shape[1] > L:
                read = read + torch.einsum('hl,hld->hd', qh[:, L:], Sim)
            temporal = read.reshape(-1) @ p[a + 'out_proj.weight'].T + p[a + 'out_proj.bias']
            n = O.normalize(z.view(O.GROUPS, W))
            pairs = O.GROUPS // 2
            th = c.dt * p[prefix + 'spatial.theta']
            cos, sin = torch.cos(th) - 1, torch.sin(th)
            first, second = n[0:2 * pairs:2], n[1:2 * pairs:2]
            rotated = torch.stack((cos * first - sin * second, sin * first + cos * second), dim=1).reshape(2 * pairs, W)
            rotated = torch.cat((rotated, n.new_zeros(O.GROUPS - 2 * pairs, W)), dim=0)
            spatial = p[prefix + 'spatial.coupling'][:, None] * rotated
            gate = torch.sigmoid(p[prefix + 'gate_logit'])[:, None]
            z = z + (gate * temporal.view(O.GROUPS, W) + (1 - gate) * spatial).reshape(-1)
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
        return np.array(out, dtype=np.uint8)


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
            l = F.cross_entropy(logits.reshape(-1, 256), w[:, 1:].reshape(-1), reduction='sum')
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
    x = torch.randint(0, 256, (2, 5 * CHUNK))
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
    y = torch.randint(0, 256, (2, 5 * CHUNK + 1))
    grads = []
    for ck in (False, True):
        q = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
        loss_of(c, q, table, y, 'polar', checkpoint=ck).backward()
        grads.append(torch.cat([q[k].grad.flatten() for k in sorted(q)]))
    print(f'checkpointed backward vs plain: max |diff| {(grads[0] - grads[1]).abs().max().item():.3e}')
    # 4. gradients of the pole parameters against central differences
    for variant in ('ct', 'polar', 'cartesian', 'real'):
        p = make_params(c, variant, 0, torch.float64)
        y = torch.randint(0, 256, (2, 5 * CHUNK + 1))
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
