import pandas as pd
R='results/'
for f in ['twin_reach_days.parquet','twin_oracle_days.parquet']:
    d=pd.read_parquet(R+f); print(f, d.columns.tolist()[:12], len(d))
    c='total' if 'total' in d else [x for x in d.columns if d[x].dtype.kind=='f'][0]
    if 'strategy' in d: print(d.groupby('strategy')[c].sum().round())
    else: print(c, round(d[c].sum()))
for f in ['days_test_oracle_parts_capaware25.parquet','days_test_qdecide_chronos_cap25.parquet','days_test_oracle_parts_cap100.parquet','days_test_oracle_parts.parquet']:
    try: d=pd.read_parquet(R+f)
    except Exception as e: print(f, e); continue
    print(f); print(d.groupby('strategy').total.sum().round().to_string())
