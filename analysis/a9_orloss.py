import json, sys, collections, re
sys.path.insert(0,'/home/user/nanoGPT')
from formal import prop
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
def top(prompt):
    g = prop.goal_of(prompt); return g[0]
idx=[i for i,r in enumerate(sft) if r['domain']=='prop' and top(r['prompt'])=='or']
def path(a):
    # sequence of Or.inl/Or.inr at front
    return tuple(re.findall(r'Or\.in[lr]', a.split('fun')[0]))
for arm in ['rlhf','rft','ppochecker','sftplus']:
    print('=====',arm)
    pc=collections.Counter(); shown=0
    for s in (1,2,3):
        R=cks[f'{arm}_s{s}']['records']
        for i in idx:
            a,b=sft[i],R[i]
            if a['lean_ok'] and not b['lean_ok']:
                pa, pb = path(a['answer']), path(b['answer'])
                kind = 'same disjunct path' if pa==pb else 'different disjunct path'
                pc[kind]+=1
                if shown<6 and s==1:
                    print(a['prompt']); print('   SFT:', a['answer']); print('   ARM:', b['answer']); shown+=1
            if b['lean_ok'] and not a['lean_ok']:
                pa, pb = path(a['answer']), path(b['answer'])
                pc['gain: '+('same' if pa==pb else 'different')]+=1
    print(pc)
# how often does each arm's answer path differ from SFT over all or-goal problems
for arm in ['rlhf','rft','ppochecker','sftplus']:
    n=dif=0
    for s in (1,2,3):
        R=cks[f'{arm}_s{s}']['records']
        for i in idx:
            n+=1; dif+= path(sft[i]['answer'])!=path(R[i]['answer'])
    print(arm,'or-goal disjunct path differs from SFT in %.1f%%'%(dif/n*100))
