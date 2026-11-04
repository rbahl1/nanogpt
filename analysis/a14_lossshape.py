import sys, json, collections, re, statistics as st
d = json.load(open('/home/user/nanoGPT/results/eval.json'))
cks = {c['label']: c for c in d['checkpoints']}
sft = cks['sft']['records']
def binders(a):
    m = re.match(r'(?:by intro |fun )(.*?)=>', a)
    if not m: return 0
    s=m.group(1)
    # count top-level binders: names or ⟨..⟩ groups
    depth=0; n=0; cur=''
    for ch in s:
        if ch=='⟨': depth+=1
        if ch=='⟩': depth-=1
        if ch==' ' and depth==0:
            if cur: n+=1
            cur=''
        else: cur+=ch
    if cur.strip(): n+=1
    return n
def unused_binder(a): return a.startswith('fun _')
print('On changed answers: length diff (arm-SFT) and top-level binder count diff')
for arm in ['rlhf','ppochecker','rft','sftplus']:
    for dom in ['prop','fol']:
        out=[]
        for kind in ['loss','gain','both wrong']:
            dl=[]; db=[]; sh=0; fewer=0; more=0; dropus=0
            for s in (1,2,3):
                for a,b in zip(sft, cks[f'{arm}_s{s}']['records']):
                    if a['domain']!=dom or a['answer']==b['answer']: continue
                    k = 'loss' if a['lean_ok'] and not b['lean_ok'] else 'gain' if b['lean_ok'] and not a['lean_ok'] else 'both wrong' if not a['lean_ok'] else 'both right'
                    if k!=kind: continue
                    dl.append(len(b['answer'])-len(a['answer']))
                    x=binders(b['answer'])-binders(a['answer']); db.append(x)
                    fewer+=x<0; more+=x>0; sh+=len(b['answer'])<len(a['answer'])
                    dropus += unused_binder(a['answer']) and not unused_binder(b['answer'])
            n=len(dl)
            out.append(f'{kind}: n={n} dLen {st.mean(dl):+.1f} shorter {sh/n*100:.0f}% fewer-binders {fewer/n*100:.0f}% more-binders {more/n*100:.0f}% dropped-leading-fun_ {dropus}')
        print(f'{arm:10s} {dom:5s}', ' || '.join(out))
