"""Point d'entree unique des controles rapides du dossier Robustesse."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-simulation-smoke", action="store_true",
        help="ajoute le diagnostic V9_4 sur 0.1 an (quelques secondes, SciPy requis)",
    )
    args = parser.parse_args()
    command = [sys.executable, "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"]
    result = subprocess.run(command, cwd=HERE.parent)
    if result.returncode:
        return result.returncode
    if args.with_simulation_smoke:
        script = HERE / "Vieillissement9_4" / "check_cost_reset_invariance.py"
        output = Path(os.environ.get("TMPDIR", "/tmp")) / "genial_invariance_smoke.txt"
        result = subprocess.run(
            [sys.executable, str(script), "--years", "0.1", "--no-legacy",
             "--no-corrective", "--output", str(output)],
            cwd=script.parent,
        )
        if result.returncode:
            return result.returncode
        print("Smoke simulation : %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
