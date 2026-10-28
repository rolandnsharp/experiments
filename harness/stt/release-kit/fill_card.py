"""fill_card.py TEMPLATE NUMBERS.json OUT: the Hugging Face card with this release's numbers."""
import json, sys
tpl, nums, out = sys.argv[1:4]
n = json.load(open(nums))
n['results_rows'] = '\n'.join('| ' + ' | '.join(r) + ' |' for r in n['results_rows'])
s = open(tpl).read().format_map(n)
assert '{' not in s.split('---', 2)[2], 'unfilled placeholder'
open(out, 'w').write(s)
