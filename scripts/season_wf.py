import pandas as pd, numpy as np, sys
R='/home/mohhakami/Documents/hackathon/QEst_2026/results/'
ACT = sys.argv[1] if len(sys.argv) > 1 else 'milp_act'  # or milp_act_gated / milp_act_rank (their runs carry no baseline: taken from the _wf files)
GT = sys.argv[2] if len(sys.argv) > 2 else '_wfg'  # tag of the gated runs
for tag in ('_wf','_wf_cap25'):
    fs=[f'{R}days_{sp}_actfc{tag}.parquet' for sp in ('train','val','test')]
    try: D=pd.concat([pd.read_parquet(f) for f in fs])
    except Exception as e: print(tag,'missing',e); continue
    BASE = sys.argv[3] if len(sys.argv) > 3 else 'milp_chronos'  # the reference strategy, from the _wf* files (or _wfr* for milp_act_rank)
    if ACT != 'milp_act':
        g=tag.replace('_wf',GT); import os
        fs2=[f'{R}days_{sp}_actfc{g}.parquet' for sp in ('train','val','test') if os.path.exists(f'{R}days_{sp}_actfc{g}.parquet')]
        if not fs2: print(tag,'gated missing'); continue
        if BASE == 'milp_act_rank': D=pd.concat([pd.read_parquet(f'{R}days_{sp}_actfc{tag.replace("_wf","_wfr")}.parquet') for sp in ('train','val','test')])
        G=pd.concat([pd.read_parquet(f) for f in fs2]); D=pd.concat([D[D.strategy==BASE]]+[G[G.strategy==ACT]]); D=D[D.day.isin(G.day.unique())]
    T=D.groupby(['strategy','day']).total.sum().unstack(0).rename(columns={ACT:'milp_act', BASE:'milp_chronos'}); T.index=pd.to_datetime(T.index); d=T['milp_act']-T['milp_chronos']
    print(f'=== {tag}: {len(T)} days {T.index.min().date()}..{T.index.max().date()}')
    M=T.groupby(T.index.to_period('M')).sum(); M['gain_%']=(100*(M.milp_act/M.milp_chronos-1)).round(1); M['n']=T.groupby(T.index.to_period('M')).size()
    ci=d.groupby(d.index.to_period('M')).agg(lambda x: 1.96*x.std()*np.sqrt(len(x))); M['diff']=d.groupby(d.index.to_period('M')).sum().round(); M['±95']=ci.round()
    print(M.round(0).to_string())
    for lab,m in (('season',T.index>=T.index.min()),('Feb-Oct (clean walk-forward)',T.index.month>=2),('Q4 only',T.index.month>=10)):
        x=d[m]; print(f'{lab:30s} base {T.milp_chronos[m].sum():13,.0f}  act {T.milp_act[m].sum():13,.0f}  {100*(T.milp_act[m].sum()/T.milp_chronos[m].sum()-1):+6.2f}%  diff {x.sum():+12,.0f} ± {1.96*x.std()*np.sqrt(len(x)):10,.0f}  win {int((x>0).sum())}/{len(x)}')
