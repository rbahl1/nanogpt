import json, numpy as np
from scipy import stats
R='/home/user/nanoGPT/results/logs/'
runs=[f'{a}_s{s}' for a in ('rlhf','ppochecker') for s in (1,2,3)]
def ols(y):
    x=np.arange(1,len(y)+1); r=stats.linregress(x,y); return r.slope*149, r.stderr*149, r.pvalue
def nw_slope(y, L=5):
    # OLS slope with Newey-West HAC SE
    x=np.arange(1,len(y)+1,dtype=float); X=np.c_[np.ones_like(x),x]
    b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
    XtXi=np.linalg.inv(X.T@X); S=np.zeros((2,2))
    for l in range(L+1):
        w=1 if l==0 else 1-l/(L+1)
        G=sum(np.outer(X[t]*e[t],X[t-l]*e[t-l]) for t in range(l,len(y)))
        S+= w*(G if l==0 else G+G.T)
    V=XtXi@S@XtXi
    return b[1]*149, np.sqrt(V[1,1])*149
out={}
for r in runs:
    d=json.load(open(R+r+'.json'))
    it=d['iterations']; n=len(it)
    S={k:np.array([e[k] for e in it],float) for k in ('score','rm_score','kl','accuracy','answer_tokens','seconds','finished')}
    for dom in ('prop','fol','nat','linear'):
        S['acc_'+dom]=np.array([e['accuracy_by_domain'][dom] for e in it])
    # rm normalised
    S['rm_norm']=(S['rm_score']-d['reward_mean'])/d['reward_std']
    out[r]=S
    print(f"== {r}: n_iter {n}, reward_mean {d['reward_mean']:.4f} std {d['reward_std']:.4f}, finished min {S['finished'].min()}")
    for k in ('score','rm_score','rm_norm','kl','accuracy','acc_prop','acc_fol','acc_nat','acc_linear','answer_tokens'):
        y=S[k]; a,b=y[:30],y[-30:]
        if k.startswith('acc'):
            # binomial pooled; per-domain each iteration has 16 samples
            m=64 if k=='accuracy' else 16
            ka,kb=a*m,b*m; pa,pb=ka.sum()/(30*m), kb.sum()/(30*m)
            se=np.sqrt(pa*(1-pa)/(30*m)+pb*(1-pb)/(30*m))
            z=(pb-pa)/se; p=2*stats.norm.sf(abs(z))
            # also iteration-variance based
            se2=np.sqrt(a.var(ddof=1)/30+b.var(ddof=1)/30)
            sl,sse,slp=ols(y)
            print(f"  {k:13s} first30 {100*pa:6.2f} last30 {100*pb:6.2f}  diff {100*(pb-pa):+6.2f} ±{100*1.96*se:.2f} (binom, p={p:.3f}; iter-var se {100*se2:.2f})  OLS slope over 150 it {100*sl:+.2f} ±{100*1.96*sse:.2f} p={slp:.3f}")
        else:
            se=np.sqrt(a.var(ddof=1)/30+b.var(ddof=1)/30); t=(b.mean()-a.mean())/se; p=2*stats.t.sf(abs(t),58)
            sl,sse,slp=ols(y); nws,nwse=nw_slope(y)
            print(f"  {k:13s} first30 {a.mean():+8.4f} last30 {b.mean():+8.4f}  diff {b.mean()-a.mean():+8.4f} ±{1.96*se:.4f} (p={p:.3g})  OLS slope*149 {sl:+.4f} ±{1.96*sse:.4f} p={slp:.3g}  NW ±{1.96*nwse:.4f}")
    print(f"  seconds total {S['seconds'].sum():.0f}  mean/iter {S['seconds'].mean():.2f}")
json.dump({r:{k:v.tolist() for k,v in S.items()} for r,S in out.items()}, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/series.json','w'))
