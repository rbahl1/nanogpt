import sys, os, json, glob, pickle
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import numpy as np
from scipy import stats
import formal, rlhf
from config import RLHFConfig
f=glob.glob('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/sftbase_K2_0,*.pkl')[0]
B=pickle.load(open(f,'rb'))
for d in ('prop','fol','nat','linear'):
    a=[];b=[]
    for s in B:
        it=np.array(B[s]['iter']); ok=np.array(B[s]['ok'],float); dm=np.array(B[s]['domain'])
        a.append(ok[(it<30)&(dm==d)]); b.append(ok[(it>=120)&(dm==d)])
    a=np.concatenate(a); b=np.concatenate(b)
    print(f'SFT T=1 on rollout prompts, {d}: first30 {100*a.mean():.1f} (n {len(a)}) last30 {100*b.mean():.1f} (n {len(b)})  diff {100*(b.mean()-a.mean()):+.1f}')
for seed in (1,2,3):
    cfg=RLHFConfig(**json.load(open(f'results/logs/rlhf_s{seed}.json'))['config'])
    sch=rlhf.rollout_problems(cfg, formal.resolve(cfg.domains))
    L=np.array([[np.mean([len(p.prompt) for p in it if p.domain==d]) for d in ('prop','fol','nat','linear')] for it in sch])
    x=np.arange(150)
    print(f'seed {seed}: mean prompt chars first30 / last30 by domain: ' + ' '.join(f"{d} {L[:30,j].mean():.1f}->{L[-30:,j].mean():.1f}" for j,d in enumerate(('prop','fol','nat','linear'))) + '  slope p: ' + ' '.join(f"{stats.linregress(x,L[:,j]).pvalue:.2g}" for j in range(4)))
