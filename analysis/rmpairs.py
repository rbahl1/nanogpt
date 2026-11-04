import sys, os, json, pickle
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import numpy as np, torch; torch.set_num_threads(1)
from common import load_checkpoint
import reward, formal
rm, tok, ck = load_checkpoint('out/reward.pt','cpu',kind='reward'); rm.eval()
pairs=[json.loads(l) for l in open('out/preferences.jsonl')]
vf=ck['run_config']['val_frac']
ho=np.array([reward.held_out(p['prompt'],vf) for p in pairs])
print('pairs', len(pairs), 'held out', ho.sum(), 'prompts', len({p['prompt'] for p,h in zip(pairs,ho) if h}), 'train prompts', len({p['prompt'] for p,h in zip(pairs,ho) if not h}))
items=sorted({(p['prompt'],p[k]) for p in pairs for k in ('chosen','rejected')})
s=reward.score(rm,tok,items,'cpu',batch_size=256); S=dict(zip(items,s.tolist()))
diff=np.array([S[(p['prompt'],p['chosen'])]-S[(p['prompt'],p['rejected'])] for p in pairs])
right=(diff>0)+0.5*(diff==0)
loss=np.log1p(np.exp(-diff))
dom=np.array([p['domain'] for p in pairs])
# type of pair: correct-vs-incorrect or both-correct (shorter wins)
okc=np.array([formal.verify(p['prompt'],p['chosen']) for p in pairs]); okr=np.array([formal.verify(p['prompt'],p['rejected']) for p in pairs])
typ=np.where(okc&~okr,'right>wrong',np.where(okc&okr,'shorter','other'))
rng=np.random.default_rng(0)
prom=np.array([p['prompt'] for p in pairs])
for name,m in (('train',~ho),('held-out',ho)):
    print(f"{name}: n {m.sum()} acc {100*right[m].mean():.1f} loss {loss[m].mean():.3f}")
    up=np.unique(prom[m]); groups={u:np.where((prom==u)&m)[0] for u in up}
    if name=='held-out':
        bs=[]
        for _ in range(2000):
            pick=rng.choice(up,len(up)); ii=np.concatenate([groups[u] for u in pick]); bs.append(right[ii].mean())
        print('   prompt-cluster bootstrap 95% CI', np.percentile(bs,[2.5,97.5])*100)
    for k in ('prop','fol','nat','linear'):
        mm=m&(dom==k); print(f"   {k}: n {mm.sum()} acc {100*right[mm].mean():.1f}")
    for t in ('right>wrong','shorter','other'):
        mm=m&(typ==t); print(f"   {t}: n {mm.sum()} acc {100*right[mm].mean():.1f}" if mm.any() else f"   {t}: 0")
    for k in ('prop','fol','nat','linear'):
        for t in ('right>wrong','shorter'):
            mm=m&(dom==k)&(typ==t)
            if mm.any(): print(f"     {k} {t}: n {mm.sum()} acc {100*right[mm].mean():.1f}")
