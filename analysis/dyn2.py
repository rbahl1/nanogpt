import json, numpy as np
from scipy import stats
S=json.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/series.json'))
S={r:{k:np.array(v) for k,v in s.items()} for r,s in S.items()}
beta=0.1
def fl(y): a,b=y[:30],y[-30:]; return a.mean(),b.mean(),b.mean()-a.mean(),np.sqrt(a.var(ddof=1)/30+b.var(ddof=1)/30)
print('Pooled over 3 seeds (first 30 vs last 30 iterations; SE from iteration-to-iteration variance, 90 iterations per window)')
for arm in ('rlhf','ppochecker'):
    for k in ('score','rm_score','rm_norm','kl','obj','accuracy','acc_prop','acc_fol','acc_nat','acc_linear','answer_tokens'):
        ys=[]
        for s in (1,2,3):
            X=S[f'{arm}_s{s}']
            y=X['score']-beta*X['kl'] if k=='obj' else X[k]
            ys.append(y)
        A=np.concatenate([y[:30] for y in ys]); B=np.concatenate([y[-30:] for y in ys])
        se=np.sqrt(A.var(ddof=1)/90+B.var(ddof=1)/90); d=B.mean()-A.mean(); p=2*stats.norm.sf(abs(d/se))
        # slope pooled: regress on iteration with seed fixed effects
        x=np.tile(np.arange(1,151),3); yy=np.concatenate([y-y.mean() for y in ys]); r=stats.linregress(x,yy)
        mul=100 if k.startswith('acc') else 1
        print(f"  {arm:10s} {k:13s} first30 {mul*A.mean():+9.4f} last30 {mul*B.mean():+9.4f} diff {mul*d:+8.4f} ±{mul*1.96*se:.4f} p={p:.3g} | slope x149 {mul*r.slope*149:+.4f} ±{mul*1.96*r.stderr*149:.4f} p={r.pvalue:.3g}")
print()
print('Paired rlhf_sX - ppochecker_sX per iteration (same rollout prompts each iteration), last 30 and first 30 iterations')
for k in ('rm_score','rm_norm','accuracy','kl','answer_tokens'):
    for w,sl in (('first30',slice(0,30)),('last30',slice(120,150)),('all150',slice(0,150))):
        D=np.concatenate([S[f'rlhf_s{s}'][k][sl]-S[f'ppochecker_s{s}'][k][sl] for s in (1,2,3)])
        se=D.std(ddof=1)/np.sqrt(len(D)); mul=100 if k=='accuracy' else 1
        print(f"  {k:13s} {w}: {mul*D.mean():+.4f} ±{mul*1.96*se:.4f} (p={2*stats.norm.sf(abs(D.mean()/se)):.3g})")
print()
# correlation between per-iteration rm_score and accuracy (within run)
for r,X in S.items():
    c=stats.pearsonr(X['rm_score'],X['accuracy'])
    print(f"  {r}: per-iteration corr(rm_score, accuracy) = {c[0]:+.3f} (p={c[1]:.3g}); corr(score-kl term) ; corr(rm, answer_tokens) {stats.pearsonr(X['rm_score'],X['answer_tokens'])[0]:+.3f}")
