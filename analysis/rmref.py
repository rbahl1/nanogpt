import sys, os, json, pickle
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import numpy as np, torch; torch.set_num_threads(1)
import data, formal, reward
from common import load_checkpoint
from reward import auc
d=json.load(open('results/eval.json'))
probs=data.sample_problems(2000,'test',31337,formal.resolve('all'),unique=True)
recs=d['checkpoints'][1]['records']
assert all(p.prompt==r['prompt'] for p,r in zip(probs,recs)), 'order mismatch'
rm,tok,ck=load_checkpoint('out/reward.pt','cpu',kind='reward'); rm.eval()
mu,sd=ck['reward_mean'],ck['reward_std']
ref=((reward.score(rm,tok,[(p.prompt,p.reference) for p in probs],'cpu')-mu)/sd).numpy()
P=pickle.load(open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','rb'))
sft=np.array([(P['raw'][(r['prompt'],r['answer'])]-mu)/sd for r in recs])
ok=np.array([r['lean_ok'] for r in recs]); dom=np.array([r['domain'] for r in recs])
same=np.array([p.reference==r['answer'] for p,r in zip(probs,recs)])
print('mean RM of reference proofs', ref.mean(), 'SFT greedy', sft.mean(), 'SFT correct', sft[ok].mean(), 'SFT wrong', sft[~ok].mean())
print('SFT greedy answer == reference:', same.mean(), 'of correct:', same[ok].mean())
m=~ok
print('Reference vs SFT wrong answer (same prompt): RM prefers reference in', ((ref>sft)[m]).mean(), 'n', m.sum())
for k in ('prop','fol','nat','linear'):
    mm=m&(dom==k); print('  ',k, ((ref>sft)[mm]).mean(), mm.sum(), 'mean ref', ref[dom==k].mean())
