#!/usr/bin/env python3
"""blah-core-http/1 scoring endpoint for a field_poles.py checkpoint (byte-level).

    core_server.py --run DIR [--port 7990] [--device cuda] [--max-seq-len 2048] [--ternary]

DIR holds result.json (variant, layers, harmonics, and ternary for a --ternary
run) and params.pt from field_poles.py. --ternary scores a ternary run whose
result.json predates that key. A token is a byte; BOS is byte 10 (newline), prepended so
position 0 has a target. Spans are found in byte space exactly as the
protocol says (common prefix for choice_set, common suffix for suffix_set,
strict prefix for greedy_pair); sequences longer than --max-seq-len keep
their last bytes. Losses are natural-log means over the continuation's own
bytes. GET /healthz, POST /v1/score, 503 when more than 4 calls are in flight.

POST / (or /v1/generate) answers the evals.blah.dev inference call, {prompt, max_tokens,
temperature, top_p} -> {text, model, usage}: the prompt's bytes after BOS, then a newline, are fed
one at a time through the model's own recurrence (zara_poles.Sampler) and bytes are sampled until a
blank line after some text, the end-of-document marker, or min(max_tokens, --max-new-bytes) bytes.
Temperature 0 is greedy; 0 < top_p < 1 samples from the nucleus. A token is a byte.
"""
import argparse
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diakosmos_oracle as O  # noqa: E402
import zara_poles  # noqa: E402
from zara_poles import CHUNK, forward  # noqa: E402

PROTOCOL = 'blah-core-http/1'
BOS = 10
EOT = b'<|endoftext|>'


def common_prefix(seqs):
    n = min(len(s) for s in seqs)
    i = 0
    while i < n and all(s[i] == seqs[0][i] for s in seqs):
        i += 1
    return i


def common_suffix(seqs):
    n = min(len(s) for s in seqs)
    i = 0
    while i < n and all(s[-1 - i] == seqs[0][-1 - i] for s in seqs):
        i += 1
    return i


class Scorer:
    def __init__(self, run, device, max_seq_len, ternary=False):
        meta = json.loads((run / 'result.json').read_text())
        # the forward the run was trained with: ternary projections when it said so
        zara_poles.TERNARY = ternary or meta.get('ternary', False)
        zara_poles.ACT8 = meta.get('act8', False)
        zara_poles.POLE_SCALE = meta.get('poleScale', 1.0)
        self.c = O.Config(harmonics=meta['harmonics'], layers=meta['layers'], heads=7, kan=8, dt=0.3)
        self.variant = meta['variant']
        self.device = torch.device(device)
        self.p = {k: v.to(self.device) for k, v in torch.load(run / 'params.pt', map_location='cpu').items()}
        self.table = O.wire_table(self.c.harmonics).to(self.device)
        self.max_seq_len = max_seq_len
        self.revision = f"{run.name}-{hashlib.sha256((run / 'params.pt').read_bytes()).hexdigest()[:12]}"
        self.lock = threading.Semaphore(4)
        self.positions = 0          # byte positions forwarded so far, for the compute account
        self.calls = 0

    @staticmethod
    def tokenize(prompt):
        return [BOS] + list(prompt.encode('utf-8'))

    def crop(self, seqs, starts, ends):
        out, s2, e2 = [], [], []
        for seq, s, e in zip(seqs, starts, ends):
            cut = max(0, len(seq) - self.max_seq_len)
            if s - cut < 1:
                raise ValueError('the answer span exceeds the context length')
            out.append(seq[cut:])
            s2.append(s - cut)
            e2.append(e - cut)
        return out, s2, e2

    @torch.no_grad()
    def losses(self, seqs):
        """Per-position losses [n][L-1] (loss[t] predicts token t+1) and argmax predictions [n][L-1]."""
        L = max(len(s) for s in seqs)
        # the chunked recurrence wants an input length that is a multiple of CHUNK;
        # the model is causal, so padding after the last byte changes nothing before it
        Lp = -(-(L - 1) // CHUNK) * CHUNK + 1
        ids = torch.full((len(seqs), Lp), BOS, dtype=torch.int64)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        ids = ids.to(self.device)
        self.positions += int(ids.numel() - ids.shape[0])
        logits = forward(self.c, self.p, self.table, ids[:, :-1], self.variant)
        loss = F.cross_entropy(logits.reshape(-1, 256), ids[:, 1:].reshape(-1), reduction='none').view(len(seqs), Lp - 1)
        return loss[:, :L - 1].float().cpu(), logits.argmax(-1)[:, :L - 1].cpu()

    def score(self, example):
        kind, prompts = example['kind'], example['prompts']
        seqs = [self.tokenize(p) for p in prompts]
        if kind == 'choice_set':
            if len(seqs) < 2:
                raise ValueError('choice_set needs >= 2 prompts')
            # byte vocabulary: an option may be exactly the common prefix (BPE would end it with a
            # different token); every option then scores from one byte before the shortest ends
            start = min(common_prefix(seqs), min(len(s) for s in seqs) - 1)
            starts, ends = [start] * len(seqs), [len(s) for s in seqs]
        elif kind == 'suffix_set':
            if len(seqs) < 2:
                raise ValueError('suffix_set needs >= 2 prompts')
            k = common_suffix(seqs)
            starts, ends = [max(len(s) - k, 1) for s in seqs], [len(s) for s in seqs]
        elif kind == 'greedy_pair':
            if len(seqs) != 2 or seqs[1][:len(seqs[0])] != seqs[0] or len(seqs[1]) <= len(seqs[0]):
                raise ValueError('greedy_pair needs [without, with] with without a strict prefix')
            seqs, starts, ends = [seqs[1]], [len(seqs[0])], [len(seqs[1])]
        else:
            raise ValueError(f'unknown kind {kind}')
        if any(s >= e for s, e in zip(starts, ends)):
            raise ValueError('empty answer span')
        seqs, starts, ends = self.crop(seqs, starts, ends)
        loss, pred = self.losses(seqs)
        if kind == 'greedy_pair':
            s, e = starts[0], ends[0]
            match = all(int(pred[0, t - 1]) == seqs[0][t] for t in range(s, e))
            return {'kind': kind, 'match': bool(match)}
        means = [float(loss[i, s - 1:e - 1].mean()) for i, (s, e) in enumerate(zip(starts, ends))]
        return {'kind': kind, 'mean_losses': means}


def generate(scorer, prompt, max_new, temperature, top_p, generator=None):
    """(text, prompt bytes, generated bytes): the model's continuation of the prompt, see the module note."""
    s = zara_poles.Sampler(scorer.c, scorer.p, scorer.table, scorer.variant)
    fed = [BOS] + list(prompt.encode('utf-8')) + [10]
    with torch.no_grad():
        for b in fed:
            logits = s.logits(b)
        out = bytearray()
        for _ in range(max_new):
            if temperature <= 0:
                nxt = int(torch.argmax(logits))
            else:
                probs = torch.softmax(logits.float() / temperature, dim=-1)
                if 0 < top_p < 1:
                    sp, idx = torch.sort(probs, descending=True)
                    keep = torch.cumsum(sp, 0) - sp < top_p
                    probs = torch.zeros_like(probs).scatter(0, idx[keep], sp[keep])
                nxt = int(torch.multinomial(probs / probs.sum(), 1, generator=generator))
            out.append(nxt)
            if out.endswith(EOT):
                del out[-len(EOT):]
                break
            if out.endswith(b'\n\n') and out.strip():
                break
            logits = s.logits(nxt)
    scorer.positions += len(fed) + len(out)
    return out.decode('utf-8', errors='replace').strip(), len(fed), len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--port', type=int, default=7990)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-seq-len', type=int, default=2048)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--ternary', action='store_true', help='a field_poles.py --ternary run whose result.json does not say so')
    ap.add_argument('--max-new-bytes', type=int, default=512, help='cap on generated bytes a call (default 512)')
    ap.add_argument('--name', default=None, help='the model name the generation route reports (default the run revision)')
    args = ap.parse_args()
    scorer = Scorer(args.run, args.device, args.max_seq_len, args.ternary)
    if args.selftest:
        ex = [{'kind': 'choice_set', 'prompts': ['Q: What is 2+2?\nAnswer: 4', 'Q: What is 2+2?\nAnswer: banana']},
              {'kind': 'greedy_pair', 'prompts': ['the cat sat on the', 'the cat sat on the mat']},
              {'kind': 'suffix_set', 'prompts': ['The trophy did not fit in the suitcase because the trophy was too big.',
                                                 'The trophy did not fit in the suitcase because the suitcase was too big.']}]
        print(json.dumps({'revision': scorer.revision, 'results': [scorer.score(e) for e in ex]}, indent=1))
        return

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % a))

        def _json(self, status, body):
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/healthz':
                self._json(200, {'status': 'ok', 'protocol': PROTOCOL, 'model_revision': scorer.revision})
            else:
                self._json(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path in ('/', '/v1/generate'):
                return self.generate()
            if self.path != '/v1/score':
                return self._json(404, {'error': 'not_found'})
            if not scorer.lock.acquire(blocking=False):
                return self._json(503, {'error': 'saturated'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                request = json.loads(self.rfile.read(length))
                examples = request['examples']
                if not 1 <= len(examples) <= 64:
                    raise ValueError('1..64 examples per call')
                self._json(200, {'results': [scorer.score(e) for e in examples]})
                scorer.calls += 1
                sys.stderr.write(f'account: call {scorer.calls} positions {scorer.positions}\n')
            except Exception as err:
                try:
                    shape = [(e.get('kind'), [len(p.encode('utf-8')) for p in e.get('prompts', [])]) for e in examples]
                except Exception:
                    shape = '?'
                sys.stderr.write(f'error: {err} on {shape}\n')
                self._json(400, {'error': str(err)})
            finally:
                scorer.lock.release()

        def generate(self):
            if not scorer.lock.acquire(blocking=False):
                return self._json(503, {'error': 'saturated'})
            try:
                request = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                prompt = request['prompt']
                max_new = max(1, min(int(request.get('max_tokens') or args.max_new_bytes), args.max_new_bytes))
                text, n_in, n_out = generate(scorer, prompt, max_new, float(request.get('temperature') or 0),
                                             float(request.get('top_p') or 0))
                self._json(200, {'text': text, 'model': args.name or scorer.revision,
                                 'usage': {'prompt_tokens': n_in, 'completion_tokens': n_out}})
                sys.stderr.write(f'generate: {n_in} prompt bytes, {n_out} generated\n')
            except Exception as err:
                sys.stderr.write(f'generate error: {err}\n')
                self._json(400, {'error': str(err)})
            finally:
                scorer.lock.release()

    print(f'serving {scorer.revision} on port {args.port} ({args.device})', flush=True)
    ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
