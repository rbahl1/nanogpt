# Regenerate RFT round-1 samples/selection for seed 1 (read-only w.r.t. project files)
import sys, json, time, pickle
sys.path.insert(0, '/home/user/nanoGPT')
import os; os.chdir('/home/user/nanoGPT')
import torch
torch.set_num_threads(1)
import formal, data, rft
from common import generate_answers, load_checkpoint, seed_everything
from config import RFTConfig, parse_config
config, preset = parse_config(RFTConfig, ['--preset','small','--seed','1','--out','/dev/null'])
seed_everything(config.seed)
device='cpu'
domains = formal.resolve(config.domains)
policy, tokenizer, sft = load_checkpoint(config.sft, device, kind="policy")
policy.eval()
rounds, _ = rft.round_problems(config, domains)
problems = rounds[0]
k = config.samples_per_prompt
sample_gen = torch.Generator(device=device).manual_seed(config.seed)
flat = [p for p in problems for _ in range(k)]
t=time.time()
answers, statuses = generate_answers(policy, tokenizer, flat, max_new_tokens=config.max_new_tokens, greedy=False,
    temperature=config.temperature, generator=sample_gen, batch_size=128, statuses=True)
print('sampled', time.time()-t, flush=True)
kept_all=[]
for i,p in enumerate(problems):
    a, s = answers[i*k:(i+1)*k], statuses[i*k:(i+1)*k]
    kept = rft.select_verified(p.prompt, a, s, config.keep_per_prompt)
    kept_all.append(kept)
print('examples', sum(len(x) for x in kept_all), flush=True)
t=time.time()
greedy = generate_answers(policy, tokenizer, problems, max_new_tokens=config.max_new_tokens, greedy=True, batch_size=64)
print('greedy', time.time()-t, flush=True)
out = {'problems': [(p.domain, p.prompt, p.reference) for p in problems], 'answers': answers, 'statuses': statuses,
       'verified': [formal.verify(p.prompt, a) and s=='finished' for p,a,s in zip(flat, answers, statuses)],
       'kept': kept_all, 'greedy': greedy, 'greedy_ok': [formal.verify(p.prompt, g) for p,g in zip(problems, greedy)]}
pickle.dump(out, open('/tmp/claude-0/-home-user-aegis/0fa20d90-65dd-5ded-9124-755795554ac2/scratchpad/analysis/rft_s1_round1.pkl','wb'))
print('done')
