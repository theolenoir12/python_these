"""Regenere RB2/Figures/218999h/first_life_metrics.txt depuis le cache RB2
(valeurs du code actuel). One-off : python regen_first_life.py"""
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from Common.lifetime_metrics import export_first_life_metrics

CACHE = os.path.join(HERE, "RB2", "_rb2_data_cache.pkl")
OUT = os.path.join(HERE, "RB2", "Figures", "218999h", "first_life_metrics.txt")
with open(CACHE, "rb") as fh:
    data = pickle.load(fh)
export_first_life_metrics(data["first_life_metrics"], OUT)
print("regenere :", OUT)
