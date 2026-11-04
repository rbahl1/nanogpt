import json, collections, sys, re
sys.path.insert(0, '/home/user/nanoGPT')
from formal import nat, linear
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
labels = ['pretrain','sft'] + [f'{a}_s{s}' for a in ['rlhf','ppochecker','rft','sftplus'] for s in (1,2,3)]
groups = {'pretrain':['pretrain'],'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
ANS = re.compile(r'^(-?\d+) := (.*)$')
def truth(r):
    return nat.value_of(r['prompt']) if r['domain']=='nat' else linear.solution_of(r['prompt'])
def nums_in_prompt(r):
    p = r['prompt']
    if r['domain']=='linear':
        p = p[len('example (x : Int) (h : '):]
    return [int(x) for x in re.findall(r'\d+', p)]
def classify(r):
    m = ANS.match(r['answer'])
    t = truth(r)
    if r['status']!='finished': return 'cutoff', None
    if not m: return 'malformed', None
    v = int(m.group(1)); pf = m.group(2)
    return ('ok' if r['lean_ok'] else ('rightval_badproof' if v==t else 'wrongval')), (v, t, pf)
for dom in ['nat','linear']:
    print(f'\n===== {dom} =====')
    for g, labs in groups.items():
        cnt = collections.Counter(); pf = collections.Counter(); err = []
        sign=0; copy=0; zero=0; wrong=0; absd=[]; vals=collections.Counter(); rel_close=0
        n=0
        for lab in labs:
            for r in cks[lab]['records']:
                if r['domain']!=dom: continue
                n+=1
                c, info = classify(r); cnt[c]+=1
                if info:
                    v,t,p = info; pf[p]+=1; vals[v]+=1
                    if c=='wrongval':
                        wrong+=1
                        absd.append(abs(v-t))
                        if v==-t and t!=0: sign+=1
                        if v in nums_in_prompt(r) : copy+=1
                        if v==0: zero+=1
        import statistics
        k=len(labs)
        med = statistics.median(absd) if absd else None
        within1 = sum(x<=1 for x in absd); within3=sum(x<=3 for x in absd)
        top = vals.most_common(3)
        print(f'{g:10s} n={n} ' + ' '.join(f'{c}={cnt[c]/n*100:.1f}%' for c in ['ok','wrongval','rightval_badproof','malformed','cutoff']))
        print(f'           wrongval: median|err|={med} |err|<=1:{within1/max(wrong,1)*100:.0f}% <=3:{within3/max(wrong,1)*100:.0f}% sign-flip:{sign/max(wrong,1)*100:.1f}% copies-prompt-number:{copy/max(wrong,1)*100:.1f}% zero:{zero/max(wrong,1)*100:.1f}%  distinct values:{len(vals)} top values {top}')
        print('           proof forms: ' + ', '.join(f'{p!r}:{c/n*100:.1f}%' for p,c in pf.most_common(6)))
