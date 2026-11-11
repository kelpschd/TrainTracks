"""
Length-matched comparison of particle track speed, cotracking vs never-cotracking.

Reads <cotracking_dir>/summary/track_speed.csv (from summarize_cotracking_velocity.py)
and writes <cotracking_dir>/summary/speed_length_matched.csv.

Why: cotracking tracks are about twice as long as never-cotracking tracks, and a
track needs >= 3 contact frames to qualify, so the raw speed comparison is partly
selection. Estimators (px/frame, cotracking minus never, 95% CI by resampling movies):
  A raw
  B matched within n_steps bins
  C matched within movie x n_steps bin (removes sg100/sg1035 imbalance too)
  D regression with movie fixed effects + n_steps bins
  E like C, but cotracking tracks contribute only their speed OUTSIDE episodes
    (is the faster population fast at baseline, or only while cotracking?)

Usage: python length_matched_speed.py /path/to/tracking_batch/cotracking
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
pd.set_option('display.width',200)
root=Path(sys.argv[1]).expanduser().resolve()/'summary'
ts=pd.read_csv(root/'track_speed.csv')
ts['group']=ts.movie.str.split('_Well').str[0]
BINS=[1,3,5,8,12,100]   # n_steps bins: 2-3,4-5,6-8,9-12,13+
ts['lbin']=pd.cut(ts.n_steps,BINS,labels=False)
rng=np.random.default_rng(0)

def raw(d,y):  return d[d.cotracks==1][y].mean()-d[d.cotracks==0][y].mean()

def matched(d,y,keys):
    """ATT: within cells defined by keys, cotrack-mean minus never-mean, weighted by #cotracking tracks."""
    g=d.groupby(keys+['cotracks'])[y].agg(['mean','size']).unstack('cotracks')
    g=g.dropna(subset=[('mean',0),('mean',1)])
    if g.empty: return np.nan
    w=g[('size',1)]
    return ((g[('mean',1)]-g[('mean',0)])*w).sum()/w.sum()

def ols(d,y):
    """coef on cotracks with movie FE + n_steps-bin dummies."""
    X=[d.cotracks.to_numpy(float)[:,None]]
    X.append(pd.get_dummies(d.mv,dtype=float).to_numpy())
    X.append(pd.get_dummies(d.lbin,drop_first=True,dtype=float).to_numpy())
    X=np.hstack(X); b=np.linalg.lstsq(X,d[y].to_numpy(float),rcond=None)[0]
    return b[0]

def boot(d,y,fn,n=1500):
    movies=d.movie.unique(); parts={m:g for m,g in d.groupby('movie')}
    out=[]
    for _ in range(n):
        pick=rng.choice(movies,len(movies))
        s=pd.concat([parts[m].assign(mv=f'{m}#{i}') for i,m in enumerate(pick)],ignore_index=True)
        if s.cotracks.nunique()<2: continue
        try: out.append(fn(s,y))
        except Exception: pass
    out=np.array(out); return np.nanpercentile(out,[2.5,97.5])

def run(o):
    d=ts[ts.organelle==o].copy(); d['mv']=d.movie
    d['outside_only']=np.where(d.cotracks==1,d.speed_outside,d.mean_speed)   # cotracking tracks: baseline outside episodes only
    rows=[]
    ests={
     'A raw (as before)':               (lambda s,y: raw(s,y),'mean_speed'),
     'B matched on length (bins)':      (lambda s,y: matched(s,y,['lbin']),'mean_speed'),
     'C matched on movie + length':     (lambda s,y: matched(s,y,['mv','lbin']),'mean_speed'),
     'D regression: movie FE + length': (lambda s,y: ols(s,y),'mean_speed'),
     'E baseline-only, movie+length':   (lambda s,y: matched(s.dropna(subset=[y]),y,['mv','lbin']),'outside_only'),
    }
    for k,(fn,y) in ests.items():
        dd=d if y=='mean_speed' else d.dropna(subset=['outside_only'])
        est=fn(dd,y); lo,hi=boot(dd,y,fn)
        rows.append((o,k,est,lo,hi,int(dd.cotracks.sum())))
    return pd.DataFrame(rows,columns=['organelle','estimator','diff_px_per_frame','ci_lo','ci_hi','n_cotracking_tracks'])
d=ts[ts.organelle=='lysosome']
print(d.groupby('lbin').agg(n=('cotracks','size'),cot=('cotracks','sum'),speed_never=('mean_speed',lambda x:x[d.loc[x.index,'cotracks']==0].mean())).assign(frac=lambda t:t.cot/t.n).round(3))

res=pd.concat([run('lysosome'),run('mitochondria')],ignore_index=True)
print(res.round(2).to_string(index=False))
res.to_csv(root/'speed_length_matched.csv',index=False)
