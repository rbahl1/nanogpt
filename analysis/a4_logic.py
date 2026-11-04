import json, collections, sys, re, statistics as st
sys.path.insert(0, '/home/user/nanoGPT')
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
groups = {'pretrain':['pretrain'],'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
def style(a):
    if a.startswith('by'): return 'tactic'
    if a.startswith('fun'): return 'fun'
    return 'other'
for dom in ['prop','fol']:
    print(f'\n===== {dom} =====')
    for g,labs in groups.items():
        n=0; cnt=collections.Counter(); sty=collections.Counter(); styok=collections.Counter()
        lens=[]; lens_ok=[]; lens_bad=[]
        for lab in labs:
            for r in cks[lab]['records']:
                if r['domain']!=dom: continue
                n+=1; cnt[r['status']]+=1
                s=style(r['answer']); sty[s]+=1; styok[s]+=r['lean_ok']
                lens.append(len(r['answer']))
                (lens_ok if r['lean_ok'] else lens_bad).append(len(r['answer']))
        print(f'{g:10s} status: ' + ' '.join(f'{k}={v/n*100:.1f}%' for k,v in cnt.items()) +
              ' | style: ' + ' '.join(f'{k}={v/n*100:.1f}% (acc {styok[k]/max(v,1)*100:.0f}%)' for k,v in sty.most_common()) +
              f' | len mean {st.mean(lens):.1f} median {st.median(lens)} ok-mean {st.mean(lens_ok):.1f} bad-mean {st.mean(lens_bad):.1f}')
# paired length: problems both SFT and arm solved
print('\nPaired length comparisons (arm - SFT), in characters')
for dom in ['prop','fol','nat','linear']:
    for g,labs in groups.items():
        if g=='sft': continue
        both=[]; allp=[]; shorter=longer=same=0
        for lab in labs:
            for a,b in zip(sft, cks[lab]['records']):
                if a['domain']!=dom: continue
                allp.append(len(b['answer'])-len(a['answer']))
                if a['lean_ok'] and b['lean_ok']:
                    both.append(len(b['answer'])-len(a['answer']))
                    if len(b['answer'])<len(a['answer']): shorter+=1
                    elif len(b['answer'])>len(a['answer']): longer+=1
                    else: same+=1
        print(f'{dom:6s} {g:10s} all problems: mean diff {st.mean(allp):+.2f}; both solved (n={len(both)}): mean diff {st.mean(both):+.2f}, shorter {shorter}, longer {longer}, same length {same}')
