"""Oscillator-first speech recogniser: a bank of damped complex oscillators at the
sample rate, a Diakosmos-shaped stack of complex-pole blocks at the frame rate,
a CTC head over characters. No attention, no convolutions, matrices only at
the frame rate.

    python zara_stt.py train --front osc|mel --data DIR --train NAMES --test NAME --out DIR
        [--extra-test NAMES --init PARAMS.pt --updates N --eval-every N ...]
    python zara_stt.py check

NAMES is one corpus name or several joined by commas (train-clean-100,train-clean-360).
Parameters are saved to OUT/params.pt at every evaluation and OUT/best.pt at the best
subset WER on --test, so a crash loses at most --eval-every updates; --init warm-starts
from a saved file (also how a fine-tune on a new corpus begins). --extra-test corpora
are scored on a subset at each evaluation and whole at the end (result.json "tests").

Data: NAME.bytes (mu-law, one byte per 16 kHz sample, utterances concatenated)
and NAME.index.json ({id, offset, length, text}), as pack_libri.py and
kokoro_pairs.py write them.

Front ends:
  osc  P complex poles z_t = c z_{t-1} + x_t, c = exp(-dt/tau + 2 pi i f dt), tau and f
       learned (log tau, log Hz), f log-spaced 50 Hz .. 7 kHz, Q = 8 at the start.
       Every FRAME samples the features are Re z, Im z (both times 1 - |c|, unit gain
       at resonance) and log mean |z|^2 over the frame: 3P numbers, no matrix.
  mel  log energies of MELS triangular mel filters over a 25 ms Hann STFT, hop FRAME:
       the conventional front end, same stack behind it.
Stack: layer-normed features -> linear -> LAYERS blocks of zara_poles.block with the
continuous-time ('ct') poles at the frame rate (decays 20 ms .. 4 s) -> layer norm ->
linear to the characters. CTC over blank + ' + space + A..Z.
--mlp H adds a gated MLP after every block (ln -> silu(W1 u) * W3 u -> W2, hidden H, a
learned sigmoid gate on the branch): the Diakosmos block has no pointwise nonlinearity,
and round 2 (Oct 22) plateaued near 80% WER without one.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # round 1 died of fragmentation, not size
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import mulaw  # noqa: E402
import zara_poles as Z  # noqa: E402

RATE = 16000
FRAME = 320                     # 20 ms
FRAME_RATE = RATE // FRAME      # 50 Hz
CHUNK = 64
ALPHABET = "'" + ' ' + ''.join(chr(ord('A') + i) for i in range(26))   # index 0 is the CTC blank
BLANK = 0
# the frame-rate stack's poles: zara_poles reads these module globals in pole() and make_params()
Z.RATE, Z.TAU_MAX, Z.TAU_LOW, Z.TAU_HIGH, Z.Q_INIT = FRAME_RATE, 8.0, 0.02, 4.0, 0.5
# the front bank's initial poles
F_LOW, F_HIGH, Q_FRONT, TAU_FRONT_MAX = 50.0, 7000.0, 8.0, 1.0
MELS, NFFT, WIN = 40, 512, 400


def encode_text(s):
    return [ALPHABET.index(ch) + 1 for ch in s.upper() if ch in ALPHABET]


def decode_ids(ids):
    return ''.join(ALPHABET[i - 1] for i in ids if i > 0)


# the exact log-depth linear recurrence lives in zara_poles (shared with the continuation model)
scan = Z.complex_scan


# ---- front ends ----
def front_pole(p):
    logr = -(1.0 / TAU_FRONT_MAX + torch.exp(-p['front.pole_decay_log_tau'])) / RATE
    return logr, 2 * math.pi / RATE * torch.exp(p['front.pole_freq_log_hz'])


def front_osc(p, x):
    """x [B][T] float -> features [B][T/FRAME][3P]."""
    B, T = x.shape
    assert T % FRAME == 0
    logr, theta = front_pole(p)
    zr, zi = scan(x[..., None].expand(B, T, logr.numel()), torch.zeros(B, T, logr.numel(), dtype=x.dtype, device=x.device), logr, theta)
    gain = (1 - torch.exp(logr))[None, None]
    zr, zi = zr * gain, zi * gain
    nf = T // FRAME
    energy = (zr * zr + zi * zi).view(B, nf, FRAME, -1).mean(2)
    return torch.cat([zr[:, FRAME - 1::FRAME], zi[:, FRAME - 1::FRAME], torch.log(energy + 1e-8)], dim=-1)


def mel_matrix(device, dtype):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(20), mel(7600), MELS + 2))
    bins = np.floor((NFFT + 1) * pts / RATE).astype(int)
    w = np.zeros((MELS, NFFT // 2 + 1))
    for m in range(MELS):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        for k in range(lo, mid):
            w[m, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            w[m, k] = (hi - k) / max(hi - mid, 1)
    return torch.tensor(w, dtype=dtype, device=device)


def front_mel(p, x):
    B, T = x.shape
    win = torch.hann_window(WIN, device=x.device, dtype=x.dtype)
    spec = torch.stft(F.pad(x, (0, FRAME)), NFFT, hop_length=FRAME, win_length=WIN, window=win, center=True, return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2                  # [B][F][frames]
    e = torch.einsum('mf,bft->btm', mel_matrix(x.device, x.dtype), power)
    return torch.log(e + 1e-8)[:, :T // FRAME]


# ---- the model ----
def make_params(c, front, poles, seed=0, dtype=torch.float32, mlp=0, lookahead=0):
    p = Z.make_params(c, 'ct', seed, torch.float64)
    del p['head.weight']
    g = torch.Generator().manual_seed(seed + 11)
    for i in range(c.layers if mlp else 0):
        b = f'blocks.{i}.mlp.'
        p[b + 'ln.weight'], p[b + 'ln.bias'] = torch.ones(c.d, dtype=torch.float64), torch.zeros(c.d, dtype=torch.float64)
        p[b + 'w1.weight'] = (torch.rand((mlp, c.d), generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(c.d)
        p[b + 'w3.weight'] = (torch.rand((mlp, c.d), generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(c.d)
        p[b + 'w2.weight'] = (torch.rand((c.d, mlp), generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(mlp)
        p[b + 'w2.bias'] = torch.zeros(c.d, dtype=torch.float64)
        p[b + 'gate_logit'] = torch.zeros(1, dtype=torch.float64)
    if front == 'osc':
        f = torch.logspace(math.log10(F_LOW), math.log10(F_HIGH), poles, dtype=torch.float64)
        tau = Q_FRONT / (math.pi * f)
        p['front.pole_decay_log_tau'] = torch.log(tau)
        p['front.pole_freq_log_hz'] = torch.log(f)
        nin = 3 * poles
    else:
        nin = MELS
    p['front.ln.weight'], p['front.ln.bias'] = torch.ones(nin, dtype=torch.float64), torch.zeros(nin, dtype=torch.float64)
    p['front.probe.weight'] = (torch.rand((c.d, nin), generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(nin)
    p['front.probe.bias'] = torch.zeros(c.d, dtype=torch.float64)
    if lookahead:
        # DeepSpeech 2's row convolution: frame t's head input mixes the final states of frames
        # t .. t + lookahead per channel; it starts as the identity (weight 1 on frame t, 0 ahead)
        # stored flat: the optimizer treats every 2-D tensor as a row-normed "rotational" matrix
        # (and centre_rotational would centre its rows), which this filter is not; 1-D gets Adam
        w = torch.zeros((lookahead + 1) * c.d, dtype=torch.float64); w[:c.d] = 1.0
        p['lookahead.weight'] = w
    p['head.weight'] = (torch.rand((len(ALPHABET) + 1, c.d), generator=g, dtype=torch.float64) * 2 - 1) / math.sqrt(c.d)
    p['head.bias'] = torch.zeros(len(ALPHABET) + 1, dtype=torch.float64)
    p = O.centre_rotational(p)
    return {k: v.to(dtype) for k, v in p.items()}


def mlp_step(p, b, z):
    """z + gate * W2 (silu(W1 u) * W3 u), u = ln(z); the rotational optimizer fixes row norms, so the gate sets the scale."""
    u = O.layer_norm(z, p[b + 'ln.weight'], p[b + 'ln.bias'])
    h = F.silu(O.product(u, p[b + 'w1.weight'], False)) * O.product(u, p[b + 'w3.weight'], False)
    return z + torch.sigmoid(p[b + 'gate_logit']) * (O.product(h, p[b + 'w2.weight'], False) + p[b + 'w2.bias'])


def forward(c, p, x, front, checkpoint=False, mask_gen=None):
    """x [B][T] float samples -> log-probabilities [B][T/FRAME][classes]. mask_gen: a torch
    Generator, and the features are masked (training with --augment only)."""
    feats = front_osc(p, x) if front == 'osc' else front_mel(p, x)
    if mask_gen is not None:
        feats = mask_features(feats, mask_gen)
    nf = feats.shape[1]
    feats = F.pad(feats, (0, 0, 0, (-nf) % CHUNK))          # the frame-rate recurrence works in chunks of CHUNK frames
    u = O.layer_norm(feats, p['front.ln.weight'], p['front.ln.bias'])
    z = O.product(u, p['front.probe.weight'], False) + p['front.probe.bias']
    for i in range(c.layers):
        if checkpoint and torch.is_grad_enabled():
            z = torch.utils.checkpoint.checkpoint(Z.block, c, p, f'blocks.{i}.', z, 'ct', use_reentrant=False)
        else:
            z = Z.block(c, p, f'blocks.{i}.', z, 'ct')
        if f'blocks.{i}.mlp.w1.weight' in p:
            z = mlp_step(p, f'blocks.{i}.mlp.', z)
    f = O.layer_norm(z, p['ln_f.weight'], p['ln_f.bias'])
    if 'lookahead.weight' in p:
        w = p['lookahead.weight'].view(-1, f.shape[-1])            # [d+1][D]
        fp = F.pad(f, (0, 0, 0, w.shape[0] - 1))                   # frames past the end see zeros
        f = sum(fp[:, j:j + f.shape[1]] * w[j] for j in range(w.shape[0]))
    return F.log_softmax(O.product(f, p['head.weight'], False) + p['head.bias'], dim=-1)[:, :nf]


def macs_per_second(c, front, poles, mlp=0):
    """Multiply-adds per second of audio, by part (a complex multiply-add counted as 4)."""
    D, H, L, dh = c.d, c.heads, c.kan, c.head_width
    if front == 'osc':
        bank = 4 * poles * RATE
        nin = 3 * poles
    else:
        bank = (NFFT * math.log2(NFFT) / 2 + MELS * (NFFT // 2 + 1)) * FRAME_RATE
        nin = MELS
    proj = nin * D * FRAME_RATE
    per_block = (D * 3 * D + 2 * H * dh * L + D * D) * FRAME_RATE + 4 * H * L * dh * FRAME_RATE
    per_block += 3 * D * mlp * FRAME_RATE if mlp else 0
    head = D * (len(ALPHABET) + 1) * FRAME_RATE
    return {'front': bank, 'projection': proj, 'blocks': per_block * c.layers, 'head': head,
            'total': bank + proj + per_block * c.layers + head}


# ---- data ----
class Corpus:
    def __init__(self, d, names, max_seconds=16.0):
        """`names`: one corpus name or several joined by commas; each keeps its own byte array."""
        self.name, self.bytes, self.items = names, [], []
        for k, name in enumerate(names.split(',')):
            self.bytes.append(np.memmap(d / f'{name}.bytes', dtype=np.uint8, mode='r'))   # 960 h is 53 GB: page it in, never load it
            for u in json.loads((d / f'{name}.index.json').read_text()):
                if u['length'] <= max_seconds * RATE and encode_text(u['text']):
                    u['src'] = k
                    self.items.append(u)
        self.hours = sum(u['length'] for u in self.items) / RATE / 3600

    def audio(self, u):
        b = self.bytes[u['src']]
        return mulaw.decode(np.asarray(b[u['offset']:u['offset'] + u['length']]))

    def batch(self, items, device, rng=None, as_numpy=False):
        """rng given: every utterance is augmented (augment_audio) before normalisation."""
        audio = [self.audio(u) for u in items]
        if rng is not None:
            audio = [augment_audio(a, rng) for a in audio]
        T = -(-max(a.size for a in audio) // FRAME) * FRAME
        x = np.zeros((len(items), T), dtype=np.float32)
        for i, a in enumerate(audio):
            x[i, :a.size] = a / (np.sqrt((a ** 2).mean()) + 1e-5) * 0.1
        targets = [encode_text(u['text']) for u in items]
        if as_numpy:
            return (x, np.array([-(-a.size // FRAME) for a in audio], dtype=np.int64),
                    np.array([t for ts in targets for t in ts], dtype=np.int64), np.array([len(t) for t in targets], dtype=np.int64))
        return (torch.from_numpy(x).to(device),
                torch.tensor([-(-a.size // FRAME) for a in audio], dtype=torch.long),
                torch.tensor([t for ts in targets for t in ts], dtype=torch.long),
                torch.tensor([len(t) for t in targets], dtype=torch.long))

    def batches(self, seconds, seed):
        """Batches of about `seconds` of audio, lengths kept similar, order shuffled per epoch."""
        rng = np.random.default_rng(seed)
        order = sorted(range(len(self.items)), key=lambda i: self.items[i]['length'])
        groups, cur, tot = [], [], 0
        for i in order:
            n = -(-self.items[i]['length'] // FRAME) * FRAME
            if cur and (len(cur) + 1) * max(tot, n) > seconds * RATE:
                groups.append(cur)
                cur, tot = [], 0
            cur.append(i)
            tot = max(tot, n)
        if cur:
            groups.append(cur)
        while True:
            for k in rng.permutation(len(groups)):
                yield [self.items[i] for i in groups[k]]


# ---- augmentation (--augment) ----
def coloured_noise(n, rng):
    """White, pink or brown noise of unit RMS, shaped in the frequency domain."""
    kind = rng.integers(3)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.maximum(np.arange(spec.size), 1.0)
    spec = spec / (f ** (0.5 * kind))             # 0 white, 1/f pink, 1/f^2 brown (power)
    x = np.fft.irfft(spec, n)
    return x / (np.sqrt((x ** 2).mean()) + 1e-12)


def augment_audio(a, rng):
    """One utterance: speed 0.9/1.0/1.1, then (each with probability 0.5) a synthetic room
    (exponentially decaying noise impulse response, RT60 0.1-0.6 s), a spectral tilt (a
    first-order filter, coefficient -0.6..0.6), and noise at 5-35 dB SNR."""
    a = a.astype(np.float64)
    sp = rng.choice([0.9, 1.0, 1.1])
    if sp != 1.0:                                  # faster speech: fewer samples
        n = int(round(a.size / sp))
        a = np.interp(np.arange(n) * sp, np.arange(a.size), a)
    if rng.random() < 0.5:
        rt60 = rng.uniform(0.1, 0.6)
        n = int(rt60 * RATE)
        t = np.arange(n) / RATE
        h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
        h[0] = 1.0 / rng.uniform(0.3, 1.0)        # the direct path, 0-10 dB above the reverb
        a = np.fft.irfft(np.fft.rfft(a, a.size + n) * np.fft.rfft(h, a.size + n), a.size + n)[:a.size]
    if rng.random() < 0.5:
        c = rng.uniform(-0.6, 0.6)
        a = np.concatenate([[a[0]], a[1:] - c * a[:-1]])
    if rng.random() < 0.5:
        snr = rng.uniform(5, 35)
        p = np.sqrt((a ** 2).mean()) + 1e-12
        a = a + coloured_noise(a.size, rng) * p * 10 ** (-snr / 20)
    return a.astype(np.float32)


def mask_features(feats, gen, time_masks=2, time_width=10, chan_masks=2, chan_frac=0.1):
    """SpecAugment-style: zero `time_masks` spans of up to `time_width` frames and `chan_masks`
    bands of up to `chan_frac` of the feature channels, per utterance. feats [B][T][C]."""
    B, T, C = feats.shape
    keep = torch.ones(B, T, C, dtype=feats.dtype, device=feats.device)
    for b in range(B):
        for _ in range(time_masks):
            w = int(torch.randint(0, time_width + 1, (1,), generator=gen)); t0 = int(torch.randint(0, max(T - w, 1), (1,), generator=gen))
            keep[b, t0:t0 + w] = 0
        for _ in range(chan_masks):
            w = int(torch.randint(0, max(int(C * chan_frac), 1) + 1, (1,), generator=gen)); c0 = int(torch.randint(0, max(C - w, 1), (1,), generator=gen))
            keep[b, :, c0:c0 + w] = 0
    return feats * keep


# ---- parallel batch preparation (--workers) ----
_W_CORPUS = None


def _w_batch(job):
    """In a worker process: one batch's numpy arrays, augmented with its own seed."""
    items, seed = job
    x, lengths, targets, tlen = _W_CORPUS.batch(items, None, np.random.default_rng(seed) if seed is not None else None, as_numpy=True)
    return x, lengths, targets, tlen


def parallel_batches(corpus, gen, workers, augment, seed):
    """Prepare batches in `workers` processes ahead of the GPU (fork: the memory-mapped corpus is shared)."""
    import multiprocessing as mp
    global _W_CORPUS
    _W_CORPUS = corpus
    import threading
    ctx = mp.get_context('fork')
    pool = ctx.Pool(workers)
    ahead = threading.Semaphore(2 * workers)      # at most 2 x workers batches prepared and not yet used
    stop = threading.Event()

    def jobs():                                   # Pool.imap pulls jobs eagerly from its own thread: bound it here
        for k in range(10 ** 12):
            ahead.acquire()
            if stop.is_set():
                return
            yield next(gen), ((seed * 1000003 + k) if augment else None)

    results = pool.imap(_w_batch, jobs(), chunksize=1)

    def feed():
        for r in results:
            ahead.release()
            yield r
    def close():                                  # unblock the feeder thread, then end the workers
        stop.set()
        ahead.release()
        pool.terminate()
        pool.join()
    return feed(), close


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def greedy(logp, length):
    ids = logp[:length].argmax(-1).tolist()
    out, last = [], BLANK
    for i in ids:
        if i != last and i != BLANK:
            out.append(i)
        last = i
    return decode_ids(out)


def evaluate(c, p, corpus, front, device, seconds=160, limit=None):
    words = chars = werr = cerr = 0
    rows = []
    items = corpus.items[:limit] if limit else corpus.items
    with torch.no_grad():
        groups = []
        order = sorted(range(len(items)), key=lambda i: items[i]['length'])
        cur, tot = [], 0
        for i in order:
            n = -(-items[i]['length'] // FRAME) * FRAME
            if cur and (len(cur) + 1) * max(tot, n) > seconds * RATE:
                groups.append(cur)
                cur, tot = [], 0
            cur.append(i)
            tot = max(tot, n)
        if cur:
            groups.append(cur)
        for g in groups:
            batch = [items[i] for i in g]
            x, lengths, _, _ = corpus.batch(batch, device)
            logp = forward(c, p, x, front)
            for k, u in enumerate(batch):
                hyp, ref = greedy(logp[k].cpu(), int(lengths[k])), u['text'].upper()
                werr += edit_distance(hyp.split(), ref.split())
                words += len(ref.split())
                cerr += edit_distance(hyp, ref)
                chars += len(ref)
                if len(rows) < 8:
                    rows.append({'id': u['id'], 'ref': ref, 'hyp': hyp})
    return {'wer': werr / max(words, 1), 'cer': cerr / max(chars, 1), 'utterances': len(items), 'examples': rows}


def describe_front(p):
    logr, theta = front_pole(p)
    hz, ms = theta * RATE / (2 * math.pi), -1000 / RATE / logr
    return f'front poles hz {hz.min():.3g}-{hz.max():.3g}, decay ms {ms.min():.3g}/{ms.median():.3g}/{ms.max():.3g}'


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = O.Config(harmonics=args.harmonics, layers=args.layers, heads=7, kan=8, dt=0.3)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / 'train.log', 'a')

    def say(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()
    train_set, test_set = Corpus(args.data, args.train, args.max_seconds), Corpus(args.data, args.test, 40.0)   # the test set is scored whole
    extra = [Corpus(args.data, n, 40.0) for n in args.extra_test.split(',')] if args.extra_test else []
    if args.test_limit:
        for e in [test_set] + extra:
            e.items = e.items[:args.test_limit]
    p = {k: v.to(device) for k, v in make_params(c, args.front, args.poles, args.seed, mlp=args.mlp, lookahead=args.lookahead).items()}
    if args.init:
        saved = torch.load(args.init, map_location=device)
        new = set(p) - set(saved)                                  # parameters the checkpoint predates keep their init
        assert set(saved) <= set(p) and new <= {'lookahead.weight'}, (set(saved) ^ set(p))
        p = {k: (saved[k].to(device, p[k].dtype) if k in saved else p[k]) for k in p}
        say(f'initialised from {args.init}')
    count = sum(v.numel() for v in p.values())
    macs = macs_per_second(c, args.front, args.poles, args.mlp)
    say(f'front {args.front} poles {args.poles} layers {c.layers} width {c.d} mlp {args.mlp} params {count} device {device} '
        f'train {train_set.name} {train_set.hours:.2f} h ({len(train_set.items)} utt) test {test_set.name} {test_set.hours:.2f} h ({len(test_set.items)} utt) '
        + ''.join(f'extra {e.name} {e.hours:.2f} h ({len(e.items)} utt) ' for e in extra)
        + f'batch {args.batch_seconds} s updates {args.updates}')
    say('multiply-adds per second of audio: ' + ', '.join(f'{k} {v / 1e6:.1f} M' for k, v in macs.items()))
    if args.front == 'osc':
        say('initial ' + describe_front(p))
    say('initial frame-rate ' + Z.describe_poles(p, c, 'ct'))
    upd = O.Update(p, args.updates)
    gen = train_set.batches(args.batch_seconds, args.seed)
    t0 = time.time()
    best = float('inf')

    def save(name):
        torch.save({k: v.detach().cpu() for k, v in p.items()}, args.out / name)
    arng = np.random.default_rng(args.seed + 101) if args.augment else None
    mgen = torch.Generator().manual_seed(args.seed + 102) if args.augment else None
    feed, close_feed = parallel_batches(train_set, gen, args.workers, args.augment, args.seed + 101) if args.workers > 0 else (None, None)
    for step in range(args.updates):
        if feed is not None:
            xn, ln, tn, tln = next(feed)
            x, lengths, targets, tlen = torch.from_numpy(xn).to(device), torch.from_numpy(ln), torch.from_numpy(tn), torch.from_numpy(tln)
        else:
            x, lengths, targets, tlen = train_set.batch(next(gen), device, arng)
        for v in p.values():
            v.requires_grad_(True)
        logp = forward(c, p, x, args.front, checkpoint=args.checkpoint, mask_gen=mgen)
        l = F.ctc_loss(logp.transpose(0, 1), targets, lengths, tlen, blank=BLANK, reduction='mean', zero_infinity=True)
        l.backward()
        grads = {k: v.grad for k, v in p.items()}
        with torch.no_grad():
            p = {k: v.detach() for k, v in p.items()}
            norm = upd(p, grads)
        if step % 20 == 0 or step == args.updates - 1:
            say(f'step {step:6d}  ctc loss {l.item():.3f}  gradnorm {norm:.3f}  {x.shape[0]} utt x {x.shape[1] / RATE:.1f} s  {time.time() - t0:.0f} s'
                + (f'  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB' if device.type == 'cuda' else ''))
        if (step + 1) % args.eval_every == 0 or step == args.updates - 1:
            r = evaluate(c, p, test_set, args.front, device, limit=args.eval_utterances)
            say(f'step {step:6d}  test ({r["utterances"]} utt) WER {r["wer"]:.3f} CER {r["cer"]:.3f}  e.g. "{r["examples"][0]["hyp"]}" for "{r["examples"][0]["ref"]}"')
            for e in extra:
                re_ = evaluate(c, p, e, args.front, device, limit=args.eval_utterances)
                say(f'step {step:6d}  {e.name} ({re_["utterances"]} utt) WER {re_["wer"]:.3f} CER {re_["cer"]:.3f}  e.g. "{re_["examples"][0]["hyp"]}" for "{re_["examples"][0]["ref"]}"')
            save('params.pt')
            if r['wer'] < best:
                best = r['wer']
                save('best.pt')
    save('params.pt')
    if close_feed is not None:                    # the workers would otherwise keep the process alive after training
        close_feed()
    r = evaluate(c, p, test_set, args.front, device)
    say(f'final test {test_set.name} ({r["utterances"]} utt) WER {r["wer"]:.3f} CER {r["cer"]:.3f}')
    tests = {}
    for e in extra:
        tests[e.name] = evaluate(c, p, e, args.front, device)
        say(f'final {e.name} ({tests[e.name]["utterances"]} utt) WER {tests[e.name]["wer"]:.3f} CER {tests[e.name]["cer"]:.3f}')
    if args.front == 'osc':
        say('learned ' + describe_front(p))
    say('learned frame-rate ' + Z.describe_poles(p, c, 'ct'))
    result = {'front': args.front, 'poles': args.poles, 'layers': c.layers, 'mlp': args.mlp, 'lookahead': args.lookahead, 'augment': args.augment, 'params': count, 'updates': args.updates,
              'batchSeconds': args.batch_seconds, 'seconds': time.time() - t0, 'macsPerSecond': macs,
              'test': r, 'tests': tests, 'train': train_set.name, 'trainHours': train_set.hours, 'init': str(args.init or ''),
              'framePoles': [Z.poles_of(p, i, 'ct') for i in range(c.layers)]}
    if args.front == 'osc':
        logr, theta = front_pole(p)
        result['frontPoles'] = {'hz': (theta * RATE / (2 * math.pi)).tolist(), 'decayMs': (-1000 / RATE / logr).tolist()}
    (args.out / 'result.json').write_text(json.dumps(result, indent=1) + '\n')
    say('done')


def check():
    torch.manual_seed(0)
    dt = torch.float64
    # 1. the log-depth scan equals the plain loop, for lengths that do and do not divide the chunk
    for N in (CHUNK * 3, CHUNK * 70 + 5):
        P = 5
        logr = -torch.rand(P, dtype=dt) * 0.01
        theta = torch.rand(P, dtype=dt)
        ur, ui = torch.randn(2, N, P, dtype=dt), torch.randn(2, N, P, dtype=dt)
        sr, si = scan(ur, ui, logr, theta)
        cr, ci = torch.exp(logr) * torch.cos(theta), torch.exp(logr) * torch.sin(theta)
        Sr, Si = torch.zeros(2, P, dtype=dt), torch.zeros(2, P, dtype=dt)
        worst = 0.0
        for t in range(N):
            Sr, Si = cr * Sr - ci * Si + ur[:, t], cr * Si + ci * Sr + ui[:, t]
            worst = max(worst, (Sr - sr[:, t]).abs().max().item(), (Si - si[:, t]).abs().max().item())
        print(f'scan vs loop, N={N}: max |diff| {worst:.3e} (states up to {sr.abs().max().item():.2f})')
    # 2. the whole model runs, both fronts, and its gradients reach the front poles
    c = O.Config(harmonics=4, layers=2, heads=7, kan=8, dt=0.3)
    x = torch.randn(2, FRAME * 40, dtype=dt) * 0.1
    for front in ('osc', 'mel'):
        p = make_params(c, front, 16, 0, dt)
        for v in p.values():
            v.requires_grad_(True)
        logp = forward(c, p, x, front)
        print(f'{front}: log-probs {tuple(logp.shape)}, rows sum to {logp.exp().sum(-1).mean().item():.6f}')
        targets = torch.tensor(encode_text('HELLO WORLD'))
        l = F.ctc_loss(logp.transpose(0, 1), targets.repeat(2), torch.tensor([40, 40]), torch.tensor([11, 11]), blank=BLANK)
        l.backward()
        if front == 'osc':
            for name in ('front.pole_decay_log_tau', 'front.pole_freq_log_hz'):
                t = p[name]
                j = 3
                eps = 1e-6
                base = t.detach().view(-1)[j].item()
                t.detach().view(-1)[j] = base + eps
                with torch.no_grad():
                    lp = F.ctc_loss(forward(c, {k: v.detach() for k, v in p.items()}, x, front).transpose(0, 1), targets.repeat(2), torch.tensor([40, 40]), torch.tensor([11, 11]), blank=BLANK).item()
                t.detach().view(-1)[j] = base - eps
                with torch.no_grad():
                    lm = F.ctc_loss(forward(c, {k: v.detach() for k, v in p.items()}, x, front).transpose(0, 1), targets.repeat(2), torch.tensor([40, 40]), torch.tensor([11, 11]), blank=BLANK).item()
                t.detach().view(-1)[j] = base
                fd, an = (lp - lm) / (2 * eps), t.grad.view(-1)[j].item()
                print(f'{name}: analytic {an:.6e} finite difference {fd:.6e}')
            print('initial ' + describe_front({k: v.detach() for k, v in p.items()}))
    print('frame-rate ' + Z.describe_poles(make_params(c, 'osc', 16, 0, dt), c, 'ct'))
    print('greedy decode of a blank-padded sequence:', repr(greedy(F.one_hot(torch.tensor([0, 9, 9, 0, 6, 13, 13, 0]), 29).double().log().clamp(min=-30), 8)))
    print('multiply-adds per second (4 layers, 64 poles):', macs_per_second(O.Config(harmonics=32, layers=4, heads=7, kan=8, dt=0.3), 'osc', 64))
    print('multiply-adds per second (4 layers, mel):', macs_per_second(O.Config(harmonics=32, layers=4, heads=7, kan=8, dt=0.3), 'mel', 64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--front', required=True, choices=['osc', 'mel'])
    t.add_argument('--data', type=Path, required=True)
    t.add_argument('--train', required=True, help='corpus name(s), e.g. train-clean-100,train-clean-360')
    t.add_argument('--test', required=True, help='the corpus the subset WER and best.pt follow')
    t.add_argument('--extra-test', default='', help='more corpora to score, comma-separated')
    t.add_argument('--init', type=Path, default=None, help='warm-start from a saved params.pt')
    t.add_argument('--lookahead', type=int, default=0, help='frames of future context for each output (row convolution; 20 ms each)')
    t.add_argument('--workers', type=int, default=0, help='processes preparing batches ahead of the GPU (0: in the training loop)')
    t.add_argument('--augment', action='store_true', help='speed, room, tilt and noise on the audio; masks on the features (training only)')
    t.add_argument('--test-limit', type=int, default=0, help='score only the first N utterances of each test corpus (smoke tests)')
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--poles', type=int, default=64)
    t.add_argument('--layers', type=int, default=4)
    t.add_argument('--harmonics', type=int, default=32)
    t.add_argument('--mlp', type=int, default=0, help='hidden width of a gated MLP after every block (0: none, the Diakosmos shape)')
    t.add_argument('--batch-seconds', type=float, default=120.0)
    t.add_argument('--max-seconds', type=float, default=16.0, help='training utterances longer than this are skipped')
    t.add_argument('--updates', type=int, default=3000)
    t.add_argument('--eval-every', type=int, default=500)
    t.add_argument('--eval-utterances', type=int, default=200)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--checkpoint', action='store_true')
    sub.add_parser('check')
    args = ap.parse_args()
    if args.command == 'check':
        check()
    else:
        train(args)


if __name__ == '__main__':
    main()
