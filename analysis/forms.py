import sys, os, json, pickle, re, collections
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import numpy as np
d=json.load(open('results/eval.json')); C={c['label']:c['records'] for c in d['checkpoints']}
P=pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','rb'))
mu,sd=P['mu'],P['sd']
def form(r):
    a=r['answer']
    if r['domain'] in ('nat','linear'):
        m=re.match(r'\s*(-?\d+)\s*:=\s*by\s+(\w+)',a); return ('num', m.group(2)) if m else ('other',a[:15])
    return a.split()[0] if a.split() else ''
for dom in ('prop','fol','nat','linear'):
    print('==',dom)
    for lab in ('sft','rlhf_s1','rlhf_s2','rlhf_s3','ppochecker_s1','sftplus_s1'):
        rs=[r for r in C[lab] if r['domain']==dom]
        cnt=collections.Counter(str(form(r)) for r in rs).most_common(5)
        L=np.mean([len(r['answer']) for r in rs])
        print(f"  {lab:14s} len {L:5.1f} acc {np.mean([r['lean_ok'] for r in rs]):.3f} RM {np.mean([(P['raw'][(r['prompt'],r['answer'])]-mu)/sd for r in rs if r['status']=='finished']):+.3f}  ", cnt)
# examples: rlhf_s1 changed, RM up, correct->wrong
s0=C['sft']; s1=C['rlhf_s1']
ex=[(r0,r1,(P['raw'][(r1['prompt'],r1['answer'])]-P['raw'][(r0['prompt'],r0['answer'])])/sd) for r0,r1 in zip(s0,s1) if r0['answer']!=r1['answer'] and r0['lean_ok'] and not r1['lean_ok'] and r1['status']=='finished']
ex.sort(key=lambda t:-t[2])
for r0,r1,dd in ex[:8]: print(f"{r0['domain']:6s} dRM {dd:+.2f} | {r0['prompt'][-70:]} | SFT: {r0['answer']} | RLHF: {r1['answer']}")
