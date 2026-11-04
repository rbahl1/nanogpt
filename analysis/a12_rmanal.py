import sys, json, pickle, statistics as st, collections
sys.path.insert(0,'/home/user/nanoGPT')
from formal import nat
from scipy.stats import spearmanr
from reward import auc as _auc
def roc_auc_score(y, s): return _auc(s, y)
z = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','rb'))
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
groups={'sft':['sft']}
for a in ['rlhf','ppochecker','rft','sftplus']: groups[a]=[f'{a}_s{s}' for s in (1,2,3)]
doms=['prop','fol','nat','linear']
print('Mean normalised RM score of greedy test answers (and AUC for Lean-correctness within the checkpoint)')
for g,labs in groups.items():
    row=[]
    for dom in doms+['all']:
        sc=[];lab_=[]
        for lab in labs:
            for r in cks[lab]['records']:
                if (dom=='all' or r['domain']==dom) and r['status']=='finished':
                    sc.append(z[(r['prompt'],r['answer'])]); lab_.append(r['lean_ok'])
        row.append(f'{dom} {st.mean(sc):+.3f} (AUC {roc_auc_score(lab_,sc):.2f})')
    print(f'{g:10s}', ' | '.join(row))
print('\nOn changed answers (arm answer != SFT answer): RM score arm - SFT, split by outcome')
for g,labs in groups.items():
    if g=='sft': continue
    for dom in doms:
        cat=collections.defaultdict(list)
        for lab in labs:
            for a,b in zip(sft, cks[lab]['records']):
                if a['domain']!=dom or a['answer']==b['answer'] or b['status']!='finished': continue
                k = 'loss' if a['lean_ok'] and not b['lean_ok'] else 'gain' if b['lean_ok'] and not a['lean_ok'] else 'both wrong' if not a['lean_ok'] else 'both right'
                cat[k].append(z[(b['prompt'],b['answer'])]-z[(a['prompt'],a['answer'])])
        print(f'{g:10s} {dom:6s}', ' '.join(f'{k}: n={len(v)} mean dRM {st.mean(v):+.2f} frac RM-up {sum(x>0 for x in v)/len(v)*100:.0f}%' for k,v in sorted(cat.items())))
# RM vs length correlation among SFT finished answers, per domain, and among correct ones
print('\nSpearman(RM score, answer length) on SFT greedy answers')
for dom in doms:
    R=[r for r in sft if r['domain']==dom]
    print(dom, 'all %.2f'%spearmanr([len(r['answer']) for r in R],[z[(r['prompt'],r['answer'])] for r in R])[0],
          'correct-only %.2f'%spearmanr([len(r['answer']) for r in R if r['lean_ok']],[z[(r['prompt'],r['answer'])] for r in R if r['lean_ok']])[0])
# nat: RM for true value vs 0 vs 1 vs SFT answer
R=[r for r in sft if r['domain']=='nat']
tv=[z[(r['prompt'],f"{nat.value_of(r['prompt'])} := by decide")] for r in R]
z0=[z[(r['prompt'],'0 := by decide')] for r in R]; z1=[z[(r['prompt'],'1 := by decide')] for r in R]
zs=[z[(r['prompt'],r['answer'])] for r in R]
print('\nnat: mean RM of true answer %.3f, of "0" %.3f, of "1" %.3f, of SFT greedy %.3f'%(st.mean(tv),st.mean(z0),st.mean(z1),st.mean(zs)))
nt=[i for i,r in enumerate(R) if nat.value_of(r['prompt'])!=0]
print('  on problems whose value != 0: RM prefers "0" over the true answer in %.0f%%; prefers "1" over true in %.0f%%; RM ranks true answer top among {true,0,1} in %.0f%%'%(
    sum(z0[i]>tv[i] for i in nt)/len(nt)*100, sum(z1[i]>tv[i] for i in [i for i in nt if nat.value_of(R[i]['prompt'])!=1])/len(nt)*100,
    sum(tv[i]>=max(z0[i],z1[i]) for i in nt)/len(nt)*100))
# does the RM prefer 0 on subtraction problems?
sub=[i for i in nt if '-' in R[i]['prompt']]
print('  value!=0 and has "-": n=%d, RM(0)>RM(true) %.0f%%'%(len(sub), sum(z0[i]>tv[i] for i in sub)/len(sub)*100))
nosub=[i for i in nt if '-' not in R[i]['prompt']]
print('  value!=0 and no "-": n=%d, RM(0)>RM(true) %.0f%%'%(len(nosub), sum(z0[i]>tv[i] for i in nosub)/len(nosub)*100))
