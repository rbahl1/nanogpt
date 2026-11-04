import json, collections, sys, re
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
groups = {'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
def head(a):
    m = re.match(r'(by \w+|fun ⟨|fun \w+|⟨|[A-Za-z_.]+|\S)', a)
    return m.group(1) if m else a[:5]
for dom in ['prop','fol']:
    print('==',dom)
    for g,labs in groups.items():
        c=collections.Counter(); ok=collections.Counter(); n=0
        for lab in labs:
            for r in cks[lab]['records']:
                if r['domain']!=dom: continue
                h=head(r['answer']); c[h]+=1; ok[h]+=r['lean_ok']; n+=1
        print(f'{g:10s}', ', '.join(f'{h!r}:{v/n*100:.1f}% (acc {ok[h]/v*100:.0f})' for h,v in c.most_common(9)))
# sample of 'other' answers in sft prop
import random
random.seed(0)
oth=[r for r in cks['sft']['records'] if r['domain']=='prop' and not r['answer'].startswith(('fun','by'))]
for r in random.sample(oth,8): print(r['prompt'],'||',r['answer'], r['lean_ok'])
