import json, sys, collections, re, difflib
sys.path.insert(0,'/home/user/nanoGPT')
from formal import prop
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
groups={}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
def top(prompt):
    g = prop.goal_of(prompt)
    return g[0] if isinstance(g, tuple) else str(g)
print(collections.Counter(top(r['prompt']) for r in sft if r['domain']=='prop'))
def toks(a): return re.findall(r'h\d+|\.\d|\.\w+|[A-Za-z_.]+|\S', a)
def diffkind(a, b):
    ta, tb = toks(a), toks(b)
    if ta[:1] != tb[:1] and ta[0] in ('Or.inl','Or.inr') and tb[0] in ('Or.inl','Or.inr'): return 'other disjunct'
    sm = difflib.SequenceMatcher(a=ta, b=tb)
    ops=[o for o in sm.get_opcodes() if o[0]!='equal']
    nchanged = sum(max(o[2]-o[1], o[4]-o[3]) for o in ops)
    if nchanged<=2: return 'small edit (<=2 tokens)'
    return 'different structure'
# goal-type breakdown of losses/gains
tp = {i: top(r['prompt']) for i,r in enumerate(sft) if r['domain']=='prop'}
print('SFT acc by goal top:', {k:(sum(sft[i]['lean_ok'] for i in tp if tp[i]==k), sum(1 for i in tp if tp[i]==k)) for k in set(tp.values())})
for g,labs in groups.items():
    L=collections.Counter(); G=collections.Counter(); dk=collections.Counter(); dkg=collections.Counter()
    for lab in labs:
        R=cks[lab]['records']
        for i in tp:
            a,b=sft[i],R[i]
            if a['lean_ok'] and not b['lean_ok']: L[tp[i]]+=1; dk[diffkind(a['answer'],b['answer'])]+=1
            if b['lean_ok'] and not a['lean_ok']: G[tp[i]]+=1; dkg[diffkind(a['answer'],b['answer'])]+=1
    print(g, 'losses by goal', dict(L), 'gains', dict(G))
    print('   loss diff kinds', dict(dk), '| gain diff kinds', dict(dkg))
