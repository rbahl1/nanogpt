import sys, json, pickle, os, time
sys.path.insert(0,'/home/user/nanoGPT'); os.chdir('/home/user/nanoGPT')
import torch; torch.set_num_threads(1)
import torch.nn.functional as F
from common import load_checkpoint, pad_batch
d = json.load(open('results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
labs = ['sft'] + [f'{a}_s{s}' for a in ['rlhf','ppochecker','rft','sftplus'] for s in (1,2,3)]
items=set()
for lab in labs:
    for r in cks[lab]['records']:
        if r['status']=='finished': items.add((r['prompt'], r['answer']))
items=sorted(items); print(len(items), flush=True)
@torch.no_grad()
def logps(model, tok, items, bs=256):
    seqs=[tok.example_ids(p,a) for p,a in items]
    order=sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out=[0.0]*len(seqs)
    for s in range(0,len(order),bs):
        ch=order[s:s+bs]
        x,mask=pad_batch([seqs[i][0][:-1] for i in ch])
        logits,_=model(x, attention_mask=mask)
        lp=F.log_softmax(logits.float(),-1)
        for j,i in enumerate(ch):
            ids,n=seqs[i]
            tgt=torch.tensor(ids[n:])            # answer tokens + eos
            pos=torch.arange(n-1, len(ids)-1)
            out[i]=lp[j,pos,tgt].sum().item()
    return out
res={}
for lab in labs:
    t=time.time()
    m,tok,_=load_checkpoint(f'out/{lab}.pt','cpu',kind='policy'); m.eval()
    res[lab]=dict(zip(items, logps(m,tok,items)))
    print(lab, time.time()-t, flush=True)
pickle.dump(res, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/logps.pkl','wb'))
