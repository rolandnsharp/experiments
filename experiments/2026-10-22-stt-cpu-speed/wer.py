"""wer.py REFS HYPDIR: word and character error rate of HYPDIR/<id>.wav.txt against REFS (id text per line)."""
import re, sys
from pathlib import Path
def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
norm = lambda s: re.sub(r"[^A-Z' ]", ' ', s.upper()).split()
W = C = we = ce = 0
for line in open(sys.argv[1]):
    uid, ref = line.rstrip('\n').split(' ', 1)
    hyp = Path(sys.argv[2], uid + '.wav.txt').read_text() if Path(sys.argv[2], uid + '.wav.txt').exists() else ''
    r, h = norm(ref), norm(hyp)
    we += ed(h, r); W += len(r); ce += ed(' '.join(h), ' '.join(r)); C += len(' '.join(r))
print(f'WER {we / W:.3f} CER {ce / C:.3f} over {W} words')
