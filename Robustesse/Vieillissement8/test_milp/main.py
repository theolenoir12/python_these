# Depuis 50-50/ ou un nouveau dossier MILP/
import sys
sys.path.append('..')
from Common.milp_weekly import run_pareto_milp
import numpy as np

pareto = run_pareto_milp(
    epsilons=np.linspace(0, 1, 11),   # 11 points
    output_root='Results_MILP',
    N_week=168,
    verbose=True,
)