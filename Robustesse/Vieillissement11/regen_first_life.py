"""Regenere les metriques RB2 depuis une simulation du code courant."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from Common.lifetime_metrics import export_first_life_metrics
from Common.main_init_and_loop import init_and_run_loop
from RB2.rb2_policy import make_rb2_policy

OUT = os.path.join(HERE, "RB2", "Figures", "218999h", "first_life_metrics.txt")
policy = make_rb2_policy(0.59, 0.49)
data = init_and_run_loop(policy, n_years=25, replacement_accounting="corrected")
export_first_life_metrics(data["first_life_metrics"], OUT)
print("regenere :", OUT)
