import os
import sys
from time import time as timer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Common.main_init_and_loop import init_and_run_loop
from Common.main_plot import run_main_plot
from get_optimal_action_RB import get_optimal_action_RB


def main():
    start = timer()
    data = init_and_run_loop(get_optimal_action_RB)
    run_main_plot(data)
    print("--- Simulation terminee en %.2f s ---" % (timer() - start))


if __name__ == "__main__":
    main()
