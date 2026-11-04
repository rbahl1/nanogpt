import sys, pickle, collections, re, statistics as st, json
sys.path.insert(0,'/home/user/nanoGPT')
from formal import nat, linear, prop
D = pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rft_s1_round1.pkl','rb'))
P = D['problems']; A=D['answers']; S=D['statuses']; V=D['verified']; K=D['kept']; G=D['greedy']; GO=D['greedy_ok']
k=4
doms=['prop','fol','nat','linear']
log=json.load(open('/home/user/nanoGPT/results/logs/rft_s1.json'))['rounds'][0]
print('logged yield', log['yield'], 'sample acc', log['sample_accuracy'])
for dom in doms:
    idx=[i for i,p in enumerate(P) if p[0]==dom]
    solved=[i for i in idx if K[i]]
    nver=[sum(V[i*k:(i+1)*k]) for i in idx]
    print(f'\n== {dom}: prompts {len(idx)}, yield {len(solved)/len(idx):.3f}, sample acc {sum(nver)/len(idx)/k:.3f}, greedy acc on these prompts {sum(GO[i] for i in idx)/len(idx):.3f}')
    # how many of kept prompts were solved by greedy SFT (easy) and verified count distribution
    c=collections.Counter(sum(V[i*k:(i+1)*k]) for i in solved)
    print('   kept prompts by #verified of 4:', dict(sorted(c.items())), ' greedy also correct on', sum(GO[i] for i in solved), 'of', len(solved))
    kept=[K[i][0] for i in solved]
    print('   kept == SFT greedy answer:', sum(K[i][0]==G[i] for i in solved), f'({sum(K[i][0]==G[i] for i in solved)/len(solved)*100:.0f}%)',
          '| kept == reference:', sum(K[i][0]==P[i][2] for i in solved),
          '| kept shorter than reference:', sum(len(K[i][0])<len(P[i][2]) for i in solved), 'longer:', sum(len(K[i][0])>len(P[i][2]) for i in solved))
    # among prompts with >=2 distinct verified answers, did shortest selection differ from the 'first' verified one?
    multi=[i for i in solved if len(set(a for a,v in zip(A[i*k:(i+1)*k],V[i*k:(i+1)*k]) if v))>1]
    print('   prompts with >=2 distinct verified answers:', len(multi), '; mean len kept', round(st.mean(len(x) for x in kept),1),
          'mean len of all verified samples', round(st.mean(len(a) for i in solved for a,v in zip(A[i*k:(i+1)*k],V[i*k:(i+1)*k]) if v),1),
          'mean len SFT greedy on all prompts', round(st.mean(len(G[i]) for i in idx),1), 'mean len reference', round(st.mean(len(P[i][2]) for i in idx),1))
    cc=collections.Counter(kept)
    print('   distinct kept answers', len(cc), 'top:', cc.most_common(6))
    if dom in ('nat','linear'):
        vals=[int(x.split(' :=')[0]) for x in kept]
        print('   kept values top:', collections.Counter(vals).most_common(8), ' proof forms', collections.Counter(x.split(':= ')[1] for x in kept))
        allv=[int(P[i][2].split(' :=')[0]) for i in idx]
        print('   all-prompt truth values top:', collections.Counter(allv).most_common(8))
        # share of value 0 in kept vs truth
        print('   share value 0: kept %.1f%% vs all prompts %.1f%%; value 1: kept %.1f%% vs %.1f%%'%(vals.count(0)/len(vals)*100, allv.count(0)/len(allv)*100, vals.count(1)/len(vals)*100, allv.count(1)/len(allv)*100))
        # samples proof forms
        print('   proof forms over all samples:', collections.Counter(a.split(':= ')[1] if ':= ' in a else 'malformed' for i in idx for a in A[i*k:(i+1)*k]))
    if dom=='prop':
        tops=collections.Counter(prop.goal_of(P[i][1])[0] for i in solved); tall=collections.Counter(prop.goal_of(P[i][1])[0] for i in idx)
        print('   goal types kept', dict(tops), 'all', dict(tall))
        sty=collections.Counter('tactic' if x.startswith('by') else 'term' for x in kept); print('   kept style', sty,
              'all samples style', collections.Counter('tactic' if a.startswith('by') else 'term' for i in idx for a in A[i*k:(i+1)*k]))
# composition of the training set
print('\ntraining-set composition by domain:', {dom: sum(1 for i,p in enumerate(P) if p[0]==dom and K[i]) for dom in doms})
