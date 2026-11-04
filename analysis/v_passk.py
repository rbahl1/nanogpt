import json, sys, numpy as np
sys.path.insert(0, '/home/user/nanoGPT')
import stats
d=json.load(open('/home/user/nanoGPT/results/passk.json'))
C={c['label']:c for c in d['checkpoints']}
doms=np.array([p['domain'] for p in C['sft']['passk']['problems']])
def per(l,k):
    return np.array([stats.pass_at_k(16,p['correct'],k) for p in C[l]['passk']['problems']])
for l in C:
    assert [p['prompt'] for p in C[l]['passk']['problems']]==[p['prompt'] for p in C['sft']['passk']['problems']]
    print(l, [round(per(l,k).mean()*100,1) for k in (1,2,4,8,16)], 'n', len(doms))
for l in C:
    if l=='sft': continue
    for k in (1,16):
        diff,lo,hi=stats.paired_bootstrap(per('sft',k),per(l,k),strata=doms)
        print(l,k,f'{diff*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}]')
