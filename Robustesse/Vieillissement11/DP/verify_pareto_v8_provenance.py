"""Vérifie localement la provenance exacte de la chaîne Pareto_V8.

Ce contrôle est destiné au workspace complet. Il n'est pas requis sur le
mésocentre, où les archives V8 et Pareto_V8 ne sont pas nécessairement copiées.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROBUSTESSE = HERE.parents[1]

EXPECTED_SOURCES = {
    ROBUSTESSE / "Vieillissement8" / "DP" / "dp_core.py":
        "5a96b17d953598e5ef0dd8295373a93d75229118ed4f3c01fede75761426d0cd",
    ROBUSTESSE / "Vieillissement8" / "DP" / "dp_aging.py":
        "b1e4d0c2726ab8df9af7ac91b1d5cbfdeab4dd74fb15df203b10fc66c4189a21",
    ROBUSTESSE / "Vieillissement8" / "DP" / "dp_pareto.py":
        "c644d4e2c3bf33dc818f210b6268bdf1a6f8c3e3d53649607806daf39fc60cfc",
    ROBUSTESSE / "Vieillissement8" / "DP" / "run_dp_pareto.slurm":
        "7986dd7c70f674bcdff61dba739fb0433670213844da788fd0daf09b7bdf817f",
}

EXPECTED_FRONT = (
    "7447b233f23159eb8944cea6050a06bad2b64853ca1b2a739159226963dae0b7"
)
FRONT_COPIES = (
    ROBUSTESSE / "Pareto_V8" / "data" / "dp_pareto_25y_51x51_v2.npz",
    ROBUSTESSE / "Vieillissement8" / "DP" / "results_meso"
    / "dp_pareto_25y_51x51_v2.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(
            f"empreinte inattendue pour {path}: {actual} != {expected}"
        )
    print(f"OK  {path.relative_to(ROBUSTESSE)}  {actual}")


def main() -> None:
    for path, expected in EXPECTED_SOURCES.items():
        verify(path, expected)
    for path in FRONT_COPIES:
        verify(path, EXPECTED_FRONT)
    print("OK -- provenance Pareto_V8 vérifiée bit à bit")


if __name__ == "__main__":
    main()
