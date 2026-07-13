"""
render_all_aging.py
===================
Harnais de rendu de all_aging_2 (RB2) sans re-simuler a chaque iteration.
La 1re execution simule RB2 (25 ans, ~48 s) et met data en cache pickle ;
les suivantes rechargent le cache et re-rendent instantanement.

Lancer depuis Robustesse/Vieillissement10/ :
    python render_all_aging.py            # rend all_aging_2 (RB2/Figures/<n>h/)
    python render_all_aging.py --resim    # force une nouvelle simulation
"""
import os, sys, pickle, argparse

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "RB2"))

from Common.main_init_and_loop import init_and_run_loop
from Common.main_plot import run_main_plot

CACHE = os.path.join(HERE, "RB2", "_rb2_data_cache.pkl")


def get_data(resim=False):
    if not resim and os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    from rb2_policy import make_rb2_policy
    policy = make_rb2_policy(0.31, 0.22, 0.90, 0.225)
    data = init_and_run_loop(policy, n_years=25)
    with open(CACHE, "wb") as fh:
        pickle.dump(data, fh)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resim", action="store_true", help="force une nouvelle simulation")
    args = ap.parse_args()
    data = get_data(resim=args.resim)
    run_main_plot(data, strategy_name="RB2")
    print("all_aging_2 rendu.")
