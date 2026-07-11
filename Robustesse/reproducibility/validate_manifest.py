"""Valide le manifeste, ses variantes RB1 et les artefacts figes."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from provenance import sha256_file


def _literal_assignments(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def validate(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    robustesse = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    if data.get("schema_version") != 1:
        errors.append("schema_version doit valoir 1")
    variants = data.get("strategy_variants", [])
    ids = [v.get("id") for v in variants]
    if len(ids) != len(set(ids)):
        errors.append("identifiants de variantes dupliques")
    for variant in variants:
        folder = robustesse / variant["folder"]
        module = folder / "get_optimal_action_RB.py"
        if not module.is_file():
            errors.append("module de variante absent : %s" % module)
            continue
        values = _literal_assignments(module)
        expected = {
            "VARIANT_ID": variant["id"],
            "SOC_LOW": variant["soc_low"],
            "SOC_HIGH": variant["soc_high"],
        }
        for key, value in expected.items():
            if values.get(key) != value:
                errors.append(
                    "%s: %s=%r, attendu %r" % (module, key, values.get(key), value)
                )
        if not variant.get("immutable"):
            errors.append("variante publiee non immuable : %s" % variant["id"])

    experiments = data.get("experiments", [])
    exp_ids = [e.get("id") for e in experiments]
    if len(exp_ids) != len(set(exp_ids)):
        errors.append("identifiants d'experiences dupliques")
    known_variants = set(ids)
    for experiment in experiments:
        for strategy in experiment.get("strategies", []):
            if strategy.startswith("rb1_") and strategy not in known_variants:
                errors.append(
                    "%s reference une variante inconnue %s"
                    % (experiment.get("id"), strategy)
                )
        for artifact in experiment.get("artifacts", []):
            path = robustesse / artifact["path"]
            if not path.is_file():
                errors.append("artefact absent : %s" % path)
                continue
            actual = sha256_file(path)
            if actual != artifact["sha256"]:
                errors.append(
                    "artefact modifie : %s (%s != %s)"
                    % (path, actual, artifact["sha256"])
                )
    return errors


def main():
    default = Path(__file__).resolve().parents[1] / "EXPERIMENTS_MANIFEST.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    errors = validate(path)
    if errors:
        print("MANIFESTE INVALIDE")
        for error in errors:
            print(" - " + error)
        return 1
    print("MANIFESTE OK : %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
