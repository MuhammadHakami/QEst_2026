import sys, os, pandas as pd
sys.path.insert(0, os.getcwd())
from qest import market, evaluate
from qest.paths import RESULTS
split, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '')
strats = sys.argv[3].split(',') if len(sys.argv) > 3 else ['milp_chronos', 'milp_act', 'milp_mk']
cfg = market.Config(settlement='official'); units = market.portfolio(['battery', 'hybrid'], official=True)
R, tar = evaluate.run(split, strats, units, cfg)
R.to_parquet(RESULTS / f'days_{split}_actfc{tag}.parquet')
pd.set_option('display.width', 250)
print(R.groupby('strategy')[['cap', 'da', 'mfrr', 'afrr', 'imbalance', 'total']].sum().round(0).to_string())
print(R.groupby(['strategy', 'unit']).total.sum().unstack().round(0).to_string())
