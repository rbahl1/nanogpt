import json, collections, sys, re, math
sys.path.insert(0, '/home/user/nanoGPT')
from formal import nat, linear
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
labels = ['sft'] + [f'{a}_s{s}' for a in ['rlhf','ppochecker','rft','sftplus'] for s in (1,2,3)]
ANS = re.compile(r'^(-?\d+) := (.*)$')
def truth(r):
    return nat.value_of(r['prompt']) if r['domain']=='nat' else linear.solution_of(r['prompt'])
def ent(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
for dom in ['nat','linear']:
    print(f'\n== {dom}: per checkpoint value distribution ==')
    T = collections.Counter(truth(r) for r in sft if r['domain']==dom)
    print('truth: distinct', len(T), 'entropy %.2f'%ent(T), 'top', T.most_common(5))
    for lab in labels:
        recs=[r for r in cks[lab]['records'] if r['domain']==dom]
        vals=[]; 
        for r in recs:
            m=ANS.match(r['answer']); vals.append(int(m.group(1)) if m else None)
        c=collections.Counter(v for v in vals if v is not None)
        top=c.most_common(5)
        top5share=sum(v for _,v in top)/len(recs)
        # accuracy split by truth==0 / small
        t=[truth(r) for r in recs]
        z=[i for i in range(len(recs)) if t[i]==0]
        zacc=sum(recs[i]['lean_ok'] for i in z)
        pred0=sum(v==0 for v in vals)
        signed=[v-tt for v,tt in zip(vals,t) if v is not None and v!=tt]
        mean_signed=sum(signed)/len(signed)
        print(f'{lab:14s} acc={sum(r["lean_ok"] for r in recs):3d} distinct={len(c):3d} H={ent(c):.2f} top5share={top5share*100:.0f}% top={top} pred0={pred0} (truth0 n={len(z)}, solved {zacc}) mean signed err={mean_signed:+.1f}')
