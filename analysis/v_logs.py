import json, re, numpy as np
R='/home/user/nanoGPT/results/logs/'
sftval=None
for a in ['rlhf','ppochecker']:
    for s in (1,2,3):
        d=json.load(open(f'{R}{a}_s{s}.json'))
        it=d['iterations']
        acc=np.array([x['accuracy'] for x in it])*100; sc=np.array([x['score'] for x in it]); kl=np.array([x['kl'] for x in it])
        slope=np.polyfit(np.arange(len(acc)),acc,1)[0]*150
        ev=d['evals']; v=[e['results']['all']['accuracy']*100 for e in ev]
        e0=ev[0]['results']; eN=ev[-1]['results']
        lin=eN['linear']['correct']-e0['linear']['correct']
        other={k:eN[k]['correct']-e0[k]['correct'] for k in ['prop','fol','nat']}
        best=int(np.argmax(v))
        print(f'{a}_s{s}: rollout acc first30 {acc[:30].mean():.1f} last30 {acc[-30:].mean():.1f} slope/150 {slope:+.1f}; score first30 {sc[:30].mean():+.3f} last30 {sc[-30:].mean():+.3f}; kl last30 {kl[-30:].mean():.3f}; val {v[0]:.1f}->{v[-1]:.1f} max {max(v):.1f} at it {ev[best]["iteration"]}; n evals {len(ev)}; lin gain {lin} other {other}')
        if sftval is None: sftval={k:(e0[k]['correct'],e0[k]['accuracy']) for k in e0}
print('sft val', sftval)
for s in (1,2,3):
    d=json.load(open(f'{R}rft_s{s}.json'))
    r=d['rounds'][-1]['val']
    print('rft', s, [x['examples'] for x in d['rounds']], [x['steps'] for x in d['rounds']], 'final val', round(r['all']['accuracy']*100,1), 'lin gain', r['linear']['correct']-3, {k:r[k]['correct']-sftval[k][0] for k in ['prop','fol','nat']})
for s in (1,2,3):
    t=open(f'{R}sftplus_s{s}.log').read().strip().splitlines()
    last=[l for l in t if 'val acc' in l][-1]
    m=re.findall(r'(\w+) [\d.]+ \((\d+)/(\d+)\)', last)
    print('sftplus', s, m)
