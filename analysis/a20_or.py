import sys, json, collections
sys.path.insert(0,'/home/user/nanoGPT')
from formal import prop
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft=cks['sft']['records']
idx=[i for i,r in enumerate(sft) if r['domain']=='prop' and prop.goal_of(r['prompt'])[0]=='or']
groups={'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
# which disjunct is provable? use the solver on each side
def side_ok(i, side):
    g=prop.goal_of(sft[i]['prompt']); sub=g[1] if side=='l' else g[2]
    return prop._tautology(sub) and prop.solve(prop.unfold(sub))[0] is not None
prov={i:(side_ok(i,'l'), side_ok(i,'r')) for i in idx}
print('or-goals', len(idx), 'left provable', sum(p[0] for p in prov.values()), 'right provable', sum(p[1] for p in prov.values()), 'both', sum(p[0] and p[1] for p in prov.values()))
for g,labs in groups.items():
    c=collections.Counter(); n=0; ok=0; wrongside=0; inl=0
    for lab in labs:
        for i in idx:
            a=cks[lab]['records'][i]['answer']; n+=1; ok+=cks[lab]['records'][i]['lean_ok']
            side='l' if a.startswith('Or.inl') else 'r' if a.startswith('Or.inr') else None
            if side: 
                inl += side=='l'
                if not prov[i][0 if side=='l' else 1]: wrongside+=1
            c[side]+=1
    print(f'{g:10s} acc {ok/n*100:.1f}%  Or.inl {c["l"]/n*100:.0f}% Or.inr {c["r"]/n*100:.0f}% other {c[None]/n*100:.0f}%  picked an unprovable disjunct {wrongside/n*100:.1f}%')
