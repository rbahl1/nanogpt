import json, pickle, numpy as np
B=pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rftbase.pkl','rb'))
D=('prop','fol','nat','linear')
def se_strat(okp,domp,fn):
    v=0
    for d in D:
        x=fn(okp[domp==d]); v+=(x.var(ddof=1)/len(x))/16
    return np.sqrt(v)
tot={'acc':[], 'yield':[], 'se_acc':[], 'se_y':[]}
for s in (1,2,3):
    L=json.load(open(f'/home/user/nanoGPT/results/logs/rft_s{s}.json'))['rounds']
    for r in (0,1):
        b=B[s][r]; lg=L[r]
        sa_l=np.mean([lg['sample_accuracy'][d] for d in D]); y_l=np.mean([lg['yield'][d] for d in D])
        sa_b=np.mean([b['sample_acc'][d] for d in D]); y_b=np.mean([b['yield'][d] for d in D])
        sea=se_strat(b['okp'],b['domp'],lambda x:x.mean(1))*np.sqrt(2); sey=se_strat(b['okp'],b['domp'],lambda x:x.any(1).astype(float))*np.sqrt(2)
        who='SFT (logged)' if r==0 else 'RFT round-1 policy (logged)'
        print(f"rft_s{s} round {r+1}: {who}: sample acc {100*sa_l:.1f} yield {100*y_l:.1f} | SFT re-sampled on same prompts: sample acc {100*sa_b:.1f} yield {100*y_b:.1f} | diff acc {100*(sa_l-sa_b):+.1f} ±{196*sea:.1f}  yield {100*(y_l-y_b):+.1f} ±{196*sey:.1f}   by domain yield diff "+' '.join(f"{d} {100*(lg['yield'][d]-b['yield'][d]):+.1f}" for d in D))
        if r==1:
            tot['acc'].append(sa_l-sa_b); tot['yield'].append(y_l-y_b); tot['se_acc'].append(sea); tot['se_y'].append(sey)
a=np.mean(tot['acc']); sa=np.sqrt(np.sum(np.square(tot['se_acc'])))/3
y=np.mean(tot['yield']); sy=np.sqrt(np.sum(np.square(tot['se_y'])))/3
print(f"pooled round 2 (policy after round 1 minus SFT on the same prompts): sample acc {100*a:+.2f} ±{196*sa:.2f}; yield {100*y:+.2f} ±{196*sy:.2f}")
