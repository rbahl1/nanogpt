import sys, json, pickle, time, os
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import torch; torch.set_num_threads(1)
from common import load_checkpoint
import reward
rm, tok, ck = load_checkpoint('out/reward.pt','cpu',kind='reward')
print({k:ck[k] for k in ('reward_mean','reward_std','val_accuracy','auc') if k in ck})
d = json.load(open('results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
labs = ['sft'] + [f'{a}_s{s}' for a in ['rlhf','ppochecker','rft','sftplus'] for s in (1,2,3)]
items=set()
for lab in labs:
    for r in cks[lab]['records']:
        if r['status']=='finished': items.add((r['prompt'], r['answer']))
# also extra: nat prompts with '0 := by decide' and the true answer
from formal import nat
for r in cks['sft']['records']:
    if r['domain']=='nat':
        items.add((r['prompt'], '0 := by decide')); items.add((r['prompt'], '1 := by decide'))
        items.add((r['prompt'], f"{nat.value_of(r['prompt'])} := by decide"))
items=sorted(items)
print(len(items), flush=True)
t=time.time()
s = reward.score(rm, tok, items, 'cpu', batch_size=256)
print('scored', time.time()-t)
z = ((s - ck['reward_mean'])/ck['reward_std']).tolist()
pickle.dump(dict(zip(items, z)), open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rmscores.pkl','wb'))
