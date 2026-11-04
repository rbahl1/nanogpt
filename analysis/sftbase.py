# SFT policy sampled (T=1) on the exact PPO rollout prompts of each seed: a per-iteration baseline
import sys, os, json, time, pickle
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import torch; torch.set_num_threads(1)
import numpy as np
import data, formal, rlhf, reward
from common import load_checkpoint, generate_answers
from config import RLHFConfig
K=int(sys.argv[1]) if len(sys.argv)>1 else 2
iters=sys.argv[2] if len(sys.argv)>2 else 'all'
pol,tok,_=load_checkpoint('out/sft.pt','cpu',kind='policy'); pol.eval()
rm,_,ck=load_checkpoint('out/reward.pt','cpu',kind='reward'); rm.eval()
mu,sd=ck['reward_mean'],ck['reward_std']
out={}
for seed in (1,2,3):
    d=json.load(open(f'results/logs/rlhf_s{seed}.json'))
    cfg=RLHFConfig(**d['config'])
    sch=rlhf.rollout_problems(cfg, formal.resolve(cfg.domains))
    its=range(150) if iters=='all' else [int(x) for x in iters.split(',')]
    flat=[(i,p) for i in its for p in sch[i] for _ in range(K)]
    g=torch.Generator().manual_seed(1000+seed)
    t=time.time()
    ans,st=generate_answers(pol,tok,[p for _,p in flat],max_new_tokens=200,greedy=False,temperature=1.0,generator=g,batch_size=256,statuses=True)
    ok=[s=='finished' and formal.verify(p.prompt,a) for (_,p),a,s in zip(flat,ans,st)]
    fin_items=[(p.prompt,a) for (_,p),a,s in zip(flat,ans,st) if s=='finished']
    sc=reward.score(rm,tok,fin_items,'cpu',batch_size=256).tolist()
    it_sc=iter(sc)
    raw=[next(it_sc) if s=='finished' else np.nan for s in st]
    ntok=[len(tok.encode(a))+1 if s=='finished' else np.nan for a,s in zip(ans,st)]
    out[seed]={'iter':[i for i,_ in flat],'domain':[p.domain for _,p in flat],'ok':ok,'status':st,'raw':raw,'ntok':ntok}
    print(seed, len(flat), 'samples', time.time()-t, 's', 'acc', np.mean(ok), flush=True)
pickle.dump(out, open(f'/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/sftbase_K{K}_{iters}.pkl','wb'))
