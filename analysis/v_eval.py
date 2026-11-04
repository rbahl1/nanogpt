import json, sys, numpy as np, math
sys.path.insert(0, '/home/user/nanoGPT')
import stats
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
E = {e['label']: e for e in d['checkpoints']}
doms = np.array([r['domain'] for r in E['sft']['records']])
ok = {l: np.array([bool(r['lean_ok']) for r in e['records']]) for l, e in E.items()}
for l in ok: print(l, round(ok[l].mean()*100, 2), E[l]['status_counts'])
tot=fin=fp=fn=0; decls=set()
for l,e in E.items():
    for r in e['records']:
        tot+=1
        if r['status']=='finished':
            fin+=1; decls.add((r['prompt'], r['answer']))
        if r['checker_ok'] and not r['lean_ok']: fp+=1
        if (not r['checker_ok']) and r['lean_ok']: fn+=1
print('total', tot, 'finished', fin, 'distinct finished (prompt,answer)', len(decls), 'fp', fp, 'fn', fn)
base = ok['sft'].astype(float)
def pool(a): return np.mean([ok[f'{a}_s{i}'] for i in (1,2,3)], axis=0)
for a in ['rlhf','ppochecker','rft','sftplus']:
    P = pool(a)
    for dom in ['prop','fol','nat','linear','all']:
        m = np.ones(len(doms),bool) if dom=='all' else doms==dom
        st = doms[m] if dom=='all' else None
        diff, lo, hi = stats.paired_bootstrap(base[m], P[m], strata=st)
        p = stats.paired_z_test(base[m], P[m], st)
        print(a, dom, f'{diff*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] p={p:.5g}')
    # differing answers and flips
    for i in (1,2,3):
        l=f'{a}_s{i}'
        diffans = sum(1 for x,y in zip(E['sft']['records'],E[l]['records']) if x['answer']!=y['answer'])
        so = int(np.sum(ok['sft'] & ~ok[l])); ao = int(np.sum(~ok['sft'] & ok[l]))
        print('  ', l, 'diff answers', diffans, 'sft-only', so, 'arm-only', ao)
# pairwise pooled
pairs=[('rlhf','ppochecker'),('ppochecker','sftplus'),('rlhf','sftplus'),('rft','sftplus')]
for a,b in pairs:
    A,B=pool(a),pool(b)
    diff,lo,hi=stats.paired_bootstrap(A,B,strata=doms); p=stats.paired_z_test(A,B,doms)
    print(f'{b}-{a}: {diff*100:+.3f} [{lo*100:+.3f},{hi*100:+.3f}] p={p:.3g}')
# bonferroni count 16 tests
print('bonferroni threshold', 0.05/16)
