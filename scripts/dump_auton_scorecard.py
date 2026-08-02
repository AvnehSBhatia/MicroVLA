#!/usr/bin/env python3
import json, glob
from pathlib import Path
rows=[]
for f in sorted(glob.glob('eval_results/auton/**/*results.json', recursive=True)):
    n=f.split('eval_results/auton/')[-1].split('/')[0]
    d=json.load(open(f)); i=d.get('intermediates') or {}
    rows.append((n, float(d.get('mean_success') or 0), float(i.get('grip_close_rate') or 0), float(i.get('eef_obj_dist_min') or 99)))
rows.sort(key=lambda r: (r[1], -r[3]), reverse=True)
Path('eval_results/auton/SCORECARD.txt').write_text('
'.join('%s succ=%.2f grip=%.2f eef=%.3f'%x for x in rows)+ '
')
print('wrote', len(rows), 'rows')
