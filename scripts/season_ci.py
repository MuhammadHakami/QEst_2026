import pandas as pd, numpy as np, glob
R='results/'
def load(pat):
    fs=sorted(glob.glob(R+pat)); d=pd.concat([pd.read_parquet(f) for f in fs]); return d
d=load('days_*_qdecide_chronos.parquet'); n=load('days_*_qdecide_chronos_nqs.parquet')
print(d.columns.tolist()); print(d.strategy.unique()[:40]); print(n.strategy.unique())
A=pd.concat([d,n]).drop_duplicates(['strategy','day','unit'])
col=[c for c in A.columns if c in ('total','eur','revenue','score')][0]; print('col',col)
T=A.pivot_table(index=['day','unit'],columns='strategy',values=col,aggfunc='first')
base=[c for c in T.columns if c.startswith('milp')][0]; print('base',base, 'rows',len(T), 'days',T.index.get_level_values(0).nunique())
D=T.groupby(level=0).sum()
for s in T.columns:
    if s==base: continue
    diff=(D[s]-D[base]).dropna(); tot=D[s].sum(); 
    print(f'{s:28s} season {tot:14,.0f}  vs {D[base].sum():14,.0f}  {100*(tot/D[base].sum()-1):+.2f}%  diff {diff.sum():+12,.0f} ±{1.96*diff.std()*np.sqrt(len(diff)):10,.0f} (n={len(diff)})')
# per quarter
q=pd.to_datetime(D.index).quarter
for s in T.columns: print(s, {int(k):round(v) for k,v in D[s].groupby(q).sum().items()})
