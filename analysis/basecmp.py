import json, pickle, glob, numpy as np
from scipy import stats
f=glob.glob('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/sftbase_K2_0,*.pkl')[0]
B=pickle.load(open(f,'rb'))
S=json.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/series.json'))
mu,sd=0.009395301342010498,2.7437539100646973
doms=['prop','fol','nat','linear']
win={'first30':range(0,30),'last30':range(120,150)}
res={}
for seed in sorted(B):
    b=B[seed]; it=np.array(b['iter']); ok=np.array(b['ok'],float); raw=np.array(b['raw'],float); nt=np.array(b['ntok'],float); dm=np.array(b['domain'])
    base={}
    for i in sorted(set(it)):
        m=it==i
        base[i]={'acc':ok[m].mean(),'rm':np.nanmean(raw[m]),'tok':np.nanmean(nt[m]), **{'acc_'+d:ok[m&(dm==d)].mean() for d in doms}}
    res[seed]=base
    print(f'seed {seed}: SFT T=1 accuracy on rollout prompts, first30 {100*np.mean([base[i]["acc"] for i in win["first30"]]):.2f}, last30 {100*np.mean([base[i]["acc"] for i in win["last30"]]):.2f}')
print()
print('Excess of the PPO policy rollouts over the SFT policy on the same prompts (SFT: 2 samples/prompt). Pooled over 3 seeds; SE from per-iteration differences (90 iterations per window).')
for arm in ('rlhf','ppochecker'):
    for key,lk,mul in (('acc','accuracy',100),('acc_prop','acc_prop',100),('acc_fol','acc_fol',100),('acc_nat','acc_nat',100),('acc_linear','acc_linear',100),('rm','rm_score',1/sd),('tok','answer_tokens',1)):
        out=[]
        for w,rg in win.items():
            D=np.array([S[f'{arm}_s{s}'][lk][i]-res[s][i][key] for s in (1,2,3) for i in rg])
            out.append((w,D.mean(),D.std(ddof=1)/np.sqrt(len(D)),D))
        dd=out[1][3]-out[0][3]; se=np.sqrt(out[0][2]**2+out[1][2]**2); ch=out[1][1]-out[0][1]
        lab=key if key!='rm' else 'rm (SD units)'
        print(f"  {arm:10s} {lab:13s} first30 {mul*out[0][1]:+7.3f} ±{1.96*mul*out[0][2]:.3f}   last30 {mul*out[1][1]:+7.3f} ±{1.96*mul*out[1][2]:.3f} (p={2*stats.norm.sf(abs(out[1][1]/out[1][2])):.3g})   change {mul*ch:+7.3f} ±{1.96*mul*se:.3f}")
# SFT baseline own drift (difficulty of prompts) first vs last
A=np.array([res[s][i]['acc'] for s in (1,2,3) for i in win['first30']]); Bb=np.array([res[s][i]['acc'] for s in (1,2,3) for i in win['last30']])
print('SFT baseline first30 vs last30 prompts: ', 100*A.mean(), 100*Bb.mean(), '±', 196*np.sqrt(A.var(ddof=1)/90+Bb.var(ddof=1)/90))
Ar=np.array([res[s][i]['rm'] for s in (1,2,3) for i in win['first30']]); Br=np.array([res[s][i]['rm'] for s in (1,2,3) for i in win['last30']])
print('SFT baseline RM (SD units) first30 vs last30 prompts: ', ((Ar-mu)/sd).mean(), ((Br-mu)/sd).mean())
