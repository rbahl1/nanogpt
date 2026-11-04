import sys, json, collections, re
sys.path.insert(0,'/home/user/nanoGPT')
from formal import nat, linear
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
ANS = re.compile(r'^(-?\d+) := (.*)$')
def val(a):
    m=ANS.match(a); return int(m.group(1)) if m else None
def truth(r): return nat.value_of(r['prompt']) if r['domain']=='nat' else linear.solution_of(r['prompt'])
def nops(p): return len(re.findall(r' [-+*/%] ', p))
for dom in ['nat','linear']:
    print('=====',dom)
    for arm in ['rlhf','ppochecker','rft','sftplus']:
        L=collections.Counter(); G=collections.Counter(); byops=collections.defaultdict(lambda:[0,0])
        for s in (1,2,3):
            R=cks[f'{arm}_s{s}']['records']
            for a,b in zip(sft,R):
                if a['domain']!=dom: continue
                t=truth(a); vb=val(b['answer']); va=val(a['answer'])
                if a['lean_ok'] and not b['lean_ok']:
                    L['arm says 0' if vb==0 else 'arm says 1' if vb==1 else 'off by 1' if vb is not None and abs(vb-t)==1 else 'malformed' if vb is None else 'other']+=1
                    byops[nops(a['prompt'])][0]+=1
                if b['lean_ok'] and not a['lean_ok']:
                    G['truth 0' if t==0 else 'truth 1' if t==1 else 'SFT off by 1' if va is not None and abs(va-t)==1 else 'other']+=1
                    byops[nops(a['prompt'])][1]+=1
        print(f'{arm:10s} losses {dict(L)} | gains {dict(G)} | by #ops (loss,gain) {dict(sorted(byops.items()))}')
# SFT accuracy by #ops
for dom in ['nat','linear']:
    c=collections.defaultdict(lambda:[0,0])
    for r in sft:
        if r['domain']==dom: c[nops(r['prompt'])][0]+=r['lean_ok']; c[nops(r['prompt'])][1]+=1
    print(dom,'SFT acc by #ops', {k:f'{v[0]}/{v[1]}' for k,v in sorted(c.items())})
