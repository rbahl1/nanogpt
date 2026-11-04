import json, collections, sys
sys.path.insert(0, '/home/user/nanoGPT')
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
print(list(cks))
sft = cks['sft']['records']
doms = ['prop','fol','nat','linear']
# sanity: same order
for lab, c in cks.items():
    assert all(a['prompt']==b['prompt'] for a,b in zip(c['records'], sft)), lab
arms = ['rlhf','ppochecker','rft','sftplus']
def ok(r): return r['lean_ok']
print('\nPer-seed gains/losses vs SFT (Lean):')
print('label  dom  sft_acc arm_acc losses gains net same_answer%')
pooled = {}
for arm in arms:
    for dom in doms+['all']:
        L=G=0; same=0; n=0
        for s in (1,2,3):
            recs = cks[f'{arm}_s{s}']['records']
            l=g=sm=nn=0
            for a,b in zip(sft, recs):
                if dom!='all' and a['domain']!=dom: continue
                nn+=1
                if ok(a) and not ok(b): l+=1
                if ok(b) and not ok(a): g+=1
                if a['answer']==b['answer']: sm+=1
            L+=l; G+=g; same+=sm; n+=nn
            if dom=='all' or True:
                pass
        pooled[(arm,dom)] = (L,G,same/n)
        print(f'{arm:11s} {dom:6s} losses(3 seeds sum)={L:4d} gains={G:4d} net/seed={(G-L)/3:+.1f} ({(G-L)/3/(n/3)*100:+.2f} pts) same_answer={same/n*100:.1f}%')
# per seed for all
print()
for arm in arms:
    for s in (1,2,3):
        recs = cks[f'{arm}_s{s}']['records']
        row=[]
        for dom in doms+['all']:
            l=sum(1 for a,b in zip(sft,recs) if (dom=='all' or a['domain']==dom) and ok(a) and not ok(b))
            g=sum(1 for a,b in zip(sft,recs) if (dom=='all' or a['domain']==dom) and ok(b) and not ok(a))
            acc=sum(ok(b) for a,b in zip(sft,recs) if (dom=='all' or a['domain']==dom))
            row.append(f'{dom}:{acc} -{l}/+{g}')
        print(f'{arm}_s{s}', ' | '.join(row))
print('sft', {dom: sum(ok(a) for a in sft if a['domain']==dom) for dom in doms})
# consistency: problems lost in k of 3 seeds
print('\nLoss consistency (# SFT-solved problems lost in k seeds), gains likewise')
for arm in arms:
    for dom in doms+['all']:
        lc=collections.Counter(); gc=collections.Counter()
        for i,a in enumerate(sft):
            if dom!='all' and a['domain']!=dom: continue
            k=sum(not ok(cks[f'{arm}_s{s}']['records'][i]) for s in (1,2,3))
            kg=sum(ok(cks[f'{arm}_s{s}']['records'][i]) for s in (1,2,3))
            if ok(a): lc[k]+=1
            else: gc[kg]+=1
        print(f'{arm:11s} {dom:6s} lost in 1/2/3 seeds: {lc[1]}/{lc[2]}/{lc[3]}   gained in 1/2/3 seeds: {gc[1]}/{gc[2]}/{gc[3]}')
# seed-seed agreement of answers
print('\nAnswer identity between seeds of same arm and with SFT (all domains)')
for arm in arms:
    r=[cks[f'{arm}_s{s}']['records'] for s in (1,2,3)]
    for dom in doms:
        idx=[i for i,a in enumerate(sft) if a['domain']==dom]
        ss=sum(r[0][i]['answer']==r[1][i]['answer'] for i in idx)/len(idx)
        sa=sum(r[s][i]['answer']==sft[i]['answer'] for s in range(3) for i in idx)/len(idx)/3
        print(f'{arm:11s} {dom:6s} s1==s2 {ss*100:.1f}%  arm==sft {sa*100:.1f}%')
