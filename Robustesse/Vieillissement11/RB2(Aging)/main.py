import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Common.main_init_and_loop import init_and_run_loop
from Common.main_plot import run_main_plot
from get_optimal_action_RB import get_optimal_action_RB


if __name__ == "__main__":
    run_main_plot(init_and_run_loop(get_optimal_action_RB))
