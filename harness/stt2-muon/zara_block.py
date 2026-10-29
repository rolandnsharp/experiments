"""The frame-rate block of the oscillator recogniser: a bank of damped complex oscillators
(continuous-time poles) between two linear maps, with a multiplicative gate, and its optimizer.

Per block, for frames t at FRAME_RATE Hz, input z_t [D], N complex states:

    u_t = LayerNorm(z_t)
    h_t = c * h_{t-1} + gamma * (B u_t)            c = exp(-dt / tau + i 2 pi f dt), per state
    y_t = C [Re h_t ; Im h_t] + d * u_t
    z_t <- z_t + W_o (y_t * silu(W_g u_t))

tau (decay time) and f (frequency) are learned per state: tau = 1 / (1 / TAU_MAX + exp(-log_tau)),
so |c| < 1 for every parameter value, and gamma = sqrt(1 - |c|^2) keeps a state's response to
white input at unit power whatever its decay. B is [2N][D] (real and imaginary input rows),
C is [D][2N]. The recurrence is diagonal and linear, so the whole sequence is computed in
parallel during training (complex_scan) and one frame at a time when streaming.

Prior art: diagonal linear recurrences with complex eigenvalues (S4D, Gu et al. 2022; LRU,
Orvieto et al. 2023, which also motivates gamma), gated outputs (GLU, Dauphin et al. 2017;
Mamba, Gu and Dao 2023), and the damped-rotation oscillator banks of Roland Sharp's audio work.

Optimizer: AdamW (Loshchilov and Hutter 2019), linear warmup then cosine decay to 10% of the peak;
weight decay only on matrices; the poles take a lower learning rate, as in LRU. Optionally Muon
(Jordan et al. 2024; Liu et al. 2025, "Muon is Scalable for LLM Training") on the blocks' hidden
matrices: Nesterov momentum, then the update orthogonalised by a quintic Newton-Schulz iteration
and scaled to AdamW's update size (0.2 sqrt(max(rows, cols))), so both share one learning rate;
AdamW keeps the input projection, the output head and every vector.
"""
import math

import torch
import torch.nn.functional as F

CHUNK = 64
FRAME_RATE = 50.0
TAU_MAX = 8.0                    # s: the slowest a state can decay
TAU_LOW, TAU_HIGH = 0.02, 4.0    # s: initial decay times, log-spaced
F_HIGH = 5.0                     # Hz: initial frequencies uniform in [0, F_HIGH] (syllable rates)


# ---- the linear recurrence, exactly, in log depth ----
def powers(logr, theta, n):
    """c^j for j = 0 .. n-1, c = exp(logr + i theta): ([P][n] real, imaginary)."""
    j = torch.arange(n, dtype=logr.dtype, device=logr.device)
    mag = torch.exp(logr[..., None] * j)
    ang = theta[..., None] * j
    return mag * torch.cos(ang), mag * torch.sin(ang)


def complex_scan(ur, ui, logr, theta):
    """S_n = c S_{n-1} + u_n for every n, u [B][N][P] (real, imaginary), c = exp(logr + i theta) per state [P].
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


# ---- the block ----
def pole(p, b):
    """Block b's states: log |c| and arg c per frame, [N] each."""
    logr = -(1.0 / TAU_MAX + torch.exp(-p[b + 'pole.log_tau'])) / FRAME_RATE
    return logr, 2 * math.pi * p[b + 'pole.freq_hz'] / FRAME_RATE


def block(p, b, z):
    """z [B][T][D] -> z + the block's update; T a multiple of CHUNK is fastest but not required."""
    u = F.layer_norm(z, z.shape[-1:], p[b + 'ln.weight'], p[b + 'ln.bias'])
    logr, theta = pole(p, b)
    gamma = torch.sqrt(-torch.expm1(2 * logr))                 # sqrt(1 - |c|^2)
    bu = F.linear(u, p[b + 'in.weight'])                       # [B][T][2N]
    N = logr.numel()
    hr, hi = complex_scan(bu[..., :N] * gamma, bu[..., N:] * gamma, logr, theta)
    y = F.linear(torch.cat([hr, hi], -1), p[b + 'out.weight']) + p[b + 'skip'] * u
    y = y * F.silu(F.linear(u, p[b + 'gate.weight']))
    return z + F.linear(y, p[b + 'proj.weight']) + p[b + 'proj.bias']


def block_stream(p, b, z):
    """The same block one frame at a time (what a streaming decoder does); for checks."""
    u = F.layer_norm(z, z.shape[-1:], p[b + 'ln.weight'], p[b + 'ln.bias'])
    logr, theta = pole(p, b)
    cr, ci = torch.exp(logr) * torch.cos(theta), torch.exp(logr) * torch.sin(theta)
    gamma = torch.sqrt(-torch.expm1(2 * logr))
    N = logr.numel()
    hr = torch.zeros(z.shape[0], N, dtype=z.dtype, device=z.device)
    hi = torch.zeros_like(hr)
    out = []
    for t in range(z.shape[1]):
        bu = F.linear(u[:, t], p[b + 'in.weight'])
        xr, xi = bu[:, :N] * gamma, bu[:, N:] * gamma
        hr, hi = cr * hr - ci * hi + xr, cr * hi + ci * hr + xi
        y = F.linear(torch.cat([hr, hi], -1), p[b + 'out.weight']) + p[b + 'skip'] * u[:, t]
        y = y * F.silu(F.linear(u[:, t], p[b + 'gate.weight']))
        out.append(z[:, t] + F.linear(y, p[b + 'proj.weight']) + p[b + 'proj.bias'])
    return torch.stack(out, 1)


def make_block(p, b, d, n, g, dtype=torch.float64):
    """Initial parameters of a block with width d and n complex states into p, under prefix b."""
    def uniform(rows, cols, scale=1.0):
        return (torch.rand((rows, cols), generator=g, dtype=dtype) * 2 - 1) * scale / math.sqrt(cols)
    p[b + 'ln.weight'], p[b + 'ln.bias'] = torch.ones(d, dtype=dtype), torch.zeros(d, dtype=dtype)
    tau = torch.exp(torch.linspace(math.log(TAU_LOW), math.log(TAU_HIGH), n, dtype=dtype))
    p[b + 'pole.log_tau'] = -torch.log(1.0 / tau - 1.0 / TAU_MAX)  # so pole() gives back tau
    p[b + 'pole.freq_hz'] = torch.rand(n, generator=g, dtype=dtype) * F_HIGH
    p[b + 'in.weight'] = uniform(2 * n, d)
    p[b + 'out.weight'] = uniform(d, 2 * n)
    p[b + 'skip'] = torch.ones(d, dtype=dtype)
    p[b + 'gate.weight'] = uniform(d, d)
    p[b + 'proj.weight'] = uniform(d, d, 0.5)
    p[b + 'proj.bias'] = torch.zeros(d, dtype=dtype)


def describe_poles(p, b):
    logr, theta = pole(p, b)
    tau = -1.0 / (logr * FRAME_RATE)
    f = theta * FRAME_RATE / (2 * math.pi)
    q = lambda t: ' '.join(f'{v:.3g}' for v in torch.quantile(t.detach().double().cpu(), torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)).tolist())
    return f'decay s [min median max] {q(tau)}, Hz {q(f)}'


# ---- the optimizer ----
NS_COEFFS = (3.4445, -4.7750, 2.0315)   # quintic Newton-Schulz (Jordan et al. 2024): pushes singular values towards 1


def orthogonalize(g, steps=5):
    """Approximately U V^T for g = U S V^T: five steps bring most singular values into about 0.7-1.2 (very small ones stay smaller)."""
    a, b, c = NS_COEFFS
    x = g / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """Muon for matrices: Nesterov momentum, orthogonalised, scaled to AdamW's update RMS, decoupled weight decay."""
    def __init__(self, params, lr, momentum=0.95, weight_decay=0.01, steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, steps=steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for v in g['params']:
                if v.grad is None:
                    continue
                buf = self.state[v].setdefault('momentum', torch.zeros_like(v))
                buf.mul_(g['momentum']).add_(v.grad)
                u = orthogonalize(v.grad.add(buf, alpha=g['momentum']), g['steps'])
                v.mul_(1 - g['lr'] * g['weight_decay'])
                v.add_(u, alpha=-g['lr'] * 0.2 * math.sqrt(max(v.shape)))


class Optimizer:
    """AdamW (or Muon on the blocks' matrices) over a dict of tensors, warmup then cosine to 10%;
    returns the gradient norm (before clipping)."""
    def __init__(self, p, total, lr=1.5e-3, warmup=1000, weight_decay=0.01, pole_lr=0.5, clip=1.0, muon=False):
        hidden = {k for k, v in p.items() if muon and v.dim() == 2 and k.startswith('blocks.')}
        mats = [v for k, v in p.items() if v.dim() == 2 and k not in hidden]
        poles = [v for k, v in p.items() if '.pole.' in k]
        rest = [v for k, v in p.items() if v.dim() != 2 and '.pole.' not in k]
        self.params = list(p.values())
        self.opt = torch.optim.AdamW([{'params': mats, 'weight_decay': weight_decay, 'scale': 1.0},
                                      {'params': rest, 'weight_decay': 0.0, 'scale': 1.0},
                                      {'params': poles, 'weight_decay': 0.0, 'scale': pole_lr}],
                                     lr=lr, betas=(0.9, 0.98), eps=1e-8)
        self.opts = [self.opt]
        if hidden:
            self.opts.append(Muon([{'params': [p[k] for k in sorted(hidden)], 'scale': 1.0}], lr=lr, weight_decay=weight_decay))
        self.lr, self.total, self.warmup, self.clip, self.step_n = lr, total, min(warmup, max(1, total // 10)), clip, 0

    def rate(self, step):
        if step < self.warmup:
            return self.lr * (step + 1) / self.warmup
        x = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, x))))

    def __call__(self):
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        r = self.rate(self.step_n)
        for o in self.opts:
            for g in o.param_groups:
                g['lr'] = r * g['scale']
            o.step()
            o.zero_grad(set_to_none=True)
        self.step_n += 1
        return norm.item()

    def state_dict(self):
        s = {'opt': self.opt.state_dict(), 'step': self.step_n}
        if len(self.opts) > 1:
            s['muon'] = self.opts[1].state_dict()
        return s

    def load_state_dict(self, s):
        self.opt.load_state_dict(s['opt'])
        if len(self.opts) > 1:
            self.opts[1].load_state_dict(s['muon'])
        self.step_n = s['step']
