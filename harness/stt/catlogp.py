"""catlogp.py DIR: DIR/<id>.lp files (float32 [frames][29]) in list order -> DIR/all.logp for --decode-logp."""
import struct, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1]); ids = (d / 'list.txt').read_text().split()
with open(d / 'all.logp', 'wb') as f:
    f.write(struct.pack('<i', len(ids)))
    for i in ids:
        a = np.fromfile(d / (i + '.lp'), dtype=np.float32).reshape(-1, 29)
        f.write(struct.pack('<i', a.shape[0])); f.write(a.tobytes())
print(d, len(ids))
