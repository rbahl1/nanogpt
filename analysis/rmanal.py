import sys, os, json, pickle
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import numpy as np
from reward import auc
from scipy import stats
P=pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','rb'))
raw,mu,sd=P['raw'],P['mu'],P['sd']
d=json.load(open('results/eval.json'))
C={c['label']:c for c in d['checkpoints']}
labs=[c['label'] for c in d['checkpoints']]
doms=['prop','fol','nat','linear']
rng=np.random.default_rng(0)
def arr(lab):
    recs=C[lab]['records']
    fin=np.array([r['status']=='finished' for r in recs])
    s=np.array([ (raw[(r['prompt'],r['answer'])]-mu)/sd if r['status']=='finished' else np.nan for r in recs])
    ok=np.array([bool(r['lean_ok']) for r in recs])
    dom=np.array([r['domain'] for r in recs])
    ans=[r['answer'] for r in recs]
    return s,ok,dom,fin,ans
A={l:arr(l) for l in labs}
s0,ok0,dom,fin0,ans0=A['sft']
print("label  meanRM(norm, finished) [95% CI problem-bootstrap]  Lean acc  dRM vs SFT [CI]  dAcc  AUC(lean)  by-domain meanRM")
B=2000
idx=rng.integers(0,2000,(B,2000))
rows=[]
for l in labs:
    s,ok,dm,fin,ans=A[l]
    m=np.nanmean(s)
    bs=np.array([np.nanmean(s[i]) for i in idx[:500]])
    lo,hi=np.percentile(bs,[2.5,97.5])
    both=fin&fin0
    dd=s[both]-s0[both]
    bd=np.array([dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(1000)])
    a=auc(s[fin],ok[fin])
    bydom={k:np.nanmean(s[dm==k]) for k in doms}
    aucdom={k:auc(s[(dm==k)&fin],ok[(dm==k)&fin]) for k in doms}
    changed=np.array([x!=y for x,y in zip(ans,ans0)])
    rows.append((l,m,lo,hi,ok.mean(),dd.mean(),*np.percentile(bd,[2.5,97.5]),a,bydom,aucdom,changed.mean()))
    print(f"{l:14s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]  acc {100*ok.mean():5.1f}  dRM {dd.mean():+.4f} [{np.percentile(bd,2.5):+.4f},{np.percentile(bd,97.5):+.4f}]  dAcc {100*(ok.mean()-ok0.mean()):+5.1f}  AUC {a:.3f}  changed {100*changed.mean():.1f}%  "+" ".join(f"{k} {v:+.3f}" for k,v in bydom.items())+ "  AUCdom "+" ".join(f"{k} {v:.3f}" for k,v in aucdom.items()))
# correlation across checkpoints
L=[r for r in rows if r[0]!='pretrain']
x=np.array([r[1] for r in L]); y=np.array([r[4] for r in L])
print('Pearson meanRM vs acc over 13 post-pretrain ckpts', stats.pearsonr(x,y), 'Spearman', stats.spearmanr(x,y))
x2=np.array([r[1] for r in rows]); y2=np.array([r[4] for r in rows])
print('incl pretrain', stats.pearsonr(x2,y2), stats.spearmanr(x2,y2))
# arm means
for arm in ['rlhf','ppochecker','rft','sftplus']:
    ls=[f'{arm}_s{i}' for i in (1,2,3)]
    print(arm, 'mean RM', np.mean([np.nanmean(A[l][0]) for l in ls]), 'sd', np.std([np.nanmean(A[l][0]) for l in ls],ddof=1), 'acc', np.mean([A[l][1].mean() for l in ls]))
# pooled paired dRM vs SFT across seeds with problem bootstrap
print('pooled paired dRM (3 seeds) vs SFT, problem bootstrap')
for arm in ['rlhf','ppochecker','rft','sftplus']:
    ls=[f'{arm}_s{i}' for i in (1,2,3)]
    D=np.stack([np.where(A[l][3]&fin0, A[l][0]-s0, np.nan) for l in ls])  # 3x2000
    per=np.nanmean(D,axis=0)
    ok_arm=np.stack([A[l][1] for l in ls]).mean(0)
    dacc=ok_arm-ok0
    bsd=[];bsa=[]
    for _ in range(2000):
        i=rng.integers(0,2000,2000); bsd.append(np.nanmean(per[i])); bsa.append(dacc[i].mean())
    print(f"  {arm:10s} dRM {np.nanmean(per):+.4f} [{np.percentile(bsd,2.5):+.4f},{np.percentile(bsd,97.5):+.4f}]  dAcc {100*dacc.mean():+.2f} [{100*np.percentile(bsa,2.5):+.2f},{100*np.percentile(bsa,97.5):+.2f}]")
    for k in doms:
        m=dom==k
        print(f"     {k:7s} dRM {np.nanmean(per[m]):+.4f}  dAcc {100*dacc[m].mean():+.2f}")
# SFT test AUC with bootstrap CI
s,ok,dm,fin,_=A['sft']
ba=[]
for _ in range(2000):
    i=rng.integers(0,2000,2000); ba.append(auc(s[i][fin[i]],ok[i][fin[i]]))
print('SFT test AUC', auc(s[fin],ok[fin]), np.percentile(ba,[2.5,97.5]))
for k in doms:
    m=(dm==k)&fin
    bb=[]
    ii=np.where(m)[0]
    for _ in range(1000):
        j=rng.choice(ii,len(ii)); bb.append(auc(s[j],ok[j]))
    print('  ',k, auc(s[m],ok[m]), np.percentile(bb,[2.5,97.5]), 'n',m.sum(),'pos',ok[m].sum())
# within-domain pooled AUC (average of domain AUCs) vs pooled
print('SFT: mean RM correct', np.nanmean(s[ok]), 'incorrect', np.nanmean(s[~ok]))
# among answers changed by rlhf vs sft: does RM prefer the rlhf answer?
print('Changed-answer analysis (arm answer != SFT answer):')
for l in labs[2:]:
    s1,ok1,_,fin1,ans1=A[l]
    ch=np.array([a!=b for a,b in zip(ans1,ans0)])&fin1&fin0
    up=(s1-s0)[ch]
    gain=(ok1&~ok0)[ch].sum(); loss=(~ok1&ok0)[ch].sum()
    # among changed, P(RM higher | correctness improved), etc
    # RM's pairwise accuracy on changed pairs where exactly one is correct
    one=ch&(ok1!=ok0)
    rmright=((s1>s0)==(ok1&~ok0))[one].mean() if one.any() else np.nan
    print(f"  {l:14s} changed {ch.sum():4d}  RM higher for new {100*(up>0).mean():5.1f}%  mean dRM {up.mean():+.3f}  new-correct {gain} new-wrong {loss}  RM pairwise acc on {one.sum()} decisive pairs {100*rmright:.1f}%")
pickle.dump(rows, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmrows.pkl','wb'))
