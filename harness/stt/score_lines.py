"""score_lines.py REFS HYPS: WER/CER of hypothesis lines against reference lines."""
import sys
def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
R = [l.rstrip('\n') for l in open(sys.argv[1])]; H = [l.rstrip('\n') for l in open(sys.argv[2])]
assert len(R) == len(H), (len(R), len(H))
W = C = we = ce = 0
for r, h in zip(R, H):
    we += ed(h.split(), r.split()); W += len(r.split()); ce += ed(h, r); C += len(r)
print(f'WER {we / W:.3f} CER {ce / C:.3f} over {len(R)} utterances')
