# SFT policy sampled on the RFT round-2 prompts (4 samples each, T=1): did round 1 change the sampled accuracy?
import sys, os, json, time, pickle
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import torch; torch.set_num_threads(1)
import numpy as np
import formal, rft
from common import load_checkpoint, generate_answers
from config import RFTConfig
pol,tok,_=load_checkpoint('out/sft.pt','cpu',kind='policy'); pol.eval()
out={}
for seed in (1,2,3):
    d=json.load(open(f'results/logs/rft_s{seed}.json'))
    cfg=RFTConfig(**d['config'])
    rounds,_=rft.round_problems(cfg, formal.resolve(cfg.domains))
    res={}
    for r in (0,1):
        probs=rounds[r]
        flat=[p for p in probs for _ in range(4)]
        g=torch.Generator().manual_seed(5000+seed*10+r)
        t=time.time()
        ans,st=generate_answers(pol,tok,flat,max_new_tokens=200,greedy=False,temperature=1.0,generator=g,batch_size=256,statuses=True)
        ok=np.array([s=='finished' and formal.verify(p.prompt,a) for p,a,s in zip(flat,ans,st)])
        dom=np.array([p.domain for p in flat])
        okp=ok.reshape(-1,4); domp=dom[::4]
        res[r]={'sample_acc':{k:ok[dom==k].mean() for k in ('prop','fol','nat','linear')},'yield':{k:okp[domp==k].any(1).mean() for k in ('prop','fol','nat','linear')},'okp':okp,'domp':domp}
        print(seed, 'round', r+1, 'SFT on these prompts: sample acc', {k:round(v,3) for k,v in res[r]['sample_acc'].items()}, 'yield', {k:round(v,3) for k,v in res[r]['yield'].items()}, f'{time.time()-t:.0f}s', flush=True)
    out[seed]=res
pickle.dump(out, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rftbase.pkl','wb'))
