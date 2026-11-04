import sys, os, json, pickle, time
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import torch; torch.set_num_threads(1)
from common import load_checkpoint
import reward
rm, tok, ck = load_checkpoint('out/reward.pt','cpu',kind='reward'); rm.eval()
mu, sd = ck['reward_mean'], ck['reward_std']
d=json.load(open('results/eval.json'))
items=set()
for c in d['checkpoints']:
    for r in c['records']:
        if r['status']=='finished': items.add((r['prompt'], r['answer']))
items=sorted(items); print(len(items), flush=True)
t=time.time()
s=reward.score(rm, tok, items, 'cpu', batch_size=256)
print('scored', time.time()-t, flush=True)
raw=dict(zip(items, s.tolist()))
# also references? test problems reference proofs: regenerate via data
pickle.dump({'raw':raw,'mu':mu,'sd':sd}, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','wb'))
