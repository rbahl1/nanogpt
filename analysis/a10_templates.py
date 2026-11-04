import json, sys, collections, re
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
labs = ['pretrain','sft'] + [f'{a}_s{s}' for a in ['rlhf','ppochecker','rft','sftplus'] for s in (1,2,3)]
for dom in ['prop','fol']:
    print('=====',dom)
    sftc = collections.Counter(r['answer'] for r in cks['sft']['records'] if r['domain']==dom)
    print('SFT top answers:', sftc.most_common(8))
    for lab in labs:
        R=[r for r in cks[lab]['records'] if r['domain']==dom]
        c=collections.Counter(r['answer'] for r in R)
        top10=sum(v for _,v in c.most_common(10))
        rep=sum(v for v in c.values() if v>=5)  # answers used on >=5 prompts
        acc_rep=sum(r['lean_ok'] for r in R if c[r['answer']]>=5)
        print(f'{lab:14s} distinct={len(c):3d}/500 top10 share={top10/5:.1f}% answers-used-on>=5-prompts={rep/5:.1f}% (their acc {acc_rep/max(rep,1)*100:.0f}%) top3={c.most_common(3)}')
