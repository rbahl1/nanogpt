import sys, json, pickle, collections, math, statistics as st
L = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/logps.pkl','rb'))
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
arms=['rlhf','ppochecker','rft','sftplus']
doms=['prop','fol','nat','linear']
# SFT confidence in its greedy answer
conf=[math.exp(L['sft'][(r['prompt'],r['answer'])]) if r['status']=='finished' else 0 for r in sft]
bins=[(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1.01)]
print('SFT-solved problems by SFT greedy-answer probability; loss rate per arm (pooled seeds)')
for dom in doms+['all']:
    idx=[i for i,r in enumerate(sft) if (dom=='all' or r['domain']==dom) and r['lean_ok']]
    row=[]
    for lo,hi in bins:
        b=[i for i in idx if lo<=conf[i]<hi]
        rates=' '.join(f"{a}:{sum(not cks[f'{a}_s{s}']['records'][i]['lean_ok'] for s in (1,2,3) for i in b)/max(3*len(b),1)*100:.0f}%" for a in arms)
        row.append(f'p[{lo},{min(hi,1)}) n={len(b)}: {rates}')
    print(dom); [print('   ',x) for x in row]
# gains: SFT-failed problems; arm success rate by SFT prob of the (correct) arm answer
print('\nMedian SFT log-prob ratio log p_sft(SFT ans) - log p_sft(arm ans) on changed answers')
for a in arms:
    for dom in doms:
        cat=collections.defaultdict(list)
        for s in (1,2,3):
            for x,y in zip(sft, cks[f'{a}_s{s}']['records']):
                if x['domain']!=dom or x['answer']==y['answer'] or y['status']!='finished': continue
                k='loss' if x['lean_ok'] and not y['lean_ok'] else 'gain' if y['lean_ok'] and not x['lean_ok'] else 'other'
                cat[k].append(L['sft'][(x['prompt'],x['answer'])]-L['sft'][(y['prompt'],y['answer'])])
        print(f'{a:10s} {dom:6s}', ' '.join(f'{k}: n={len(v)} median {st.median(v):.2f} (<1 nat: {sum(t<1 for t in v)/len(v)*100:.0f}%)' for k,v in sorted(cat.items())))
# How far did each arm move: mean log p_arm(SFT answer) - log p_sft(SFT answer) on SFT-correct vs SFT-wrong
print('\nMean change in log-prob of SFT greedy answer (arm - SFT), SFT-correct vs SFT-wrong answers')
for a in arms:
    row=[]
    for dom in doms:
        rc=[L[f'{a}_s{s}'][(x['prompt'],x['answer'])]-L['sft'][(x['prompt'],x['answer'])] for s in (1,2,3) for x in sft if x['domain']==dom and x['lean_ok']]
        rw=[L[f'{a}_s{s}'][(x['prompt'],x['answer'])]-L['sft'][(x['prompt'],x['answer'])] for s in (1,2,3) for x in sft if x['domain']==dom and not x['lean_ok'] and x['status']=='finished']
        row.append(f'{dom}: correct {st.mean(rc):+.3f} wrong {st.mean(rw):+.3f}')
    print(f'{a:10s}', ' | '.join(row))
