"""Valide le manifeste, ses variantes RB1 et les artefacts figes."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

try:
    from .provenance import sha256_file
except ImportError:
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
    revisions = data.get("engine_revisions", [])
    revision_ids = [revision.get("id") for revision in revisions]
    if len(revision_ids) != len(set(revision_ids)):
        errors.append("identifiants de revisions moteur dupliques")
    for revision in revisions:
        for source in revision.get("sources", []):
            path = robustesse / source["path"]
            if not path.is_file():
                errors.append("source moteur absente : %s" % path)
            elif sha256_file(path) != source["sha256"]:
                errors.append("source moteur modifiee : %s" % path)
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
        actual_wrapper = sha256_file(module)
        if actual_wrapper != variant.get("wrapper_sha256"):
            errors.append(
                "wrapper modifie : %s (%s != %s)"
                % (module, actual_wrapper, variant.get("wrapper_sha256"))
            )
        logic = robustesse / variant.get("logic_path", "")
        if not logic.is_file():
            errors.append("noyau de strategie absent : %s" % logic)
        elif sha256_file(logic) != variant.get("logic_sha256"):
            errors.append("noyau de strategie modifie : %s" % logic)

    experiments = data.get("experiments", [])
    exp_ids = [e.get("id") for e in experiments]
    if len(exp_ids) != len(set(exp_ids)):
        errors.append("identifiants d'experiences dupliques")
    known_variants = set(ids)
    allowed_statuses = {
        "published_legacy_unfingerprinted",
        "legacy_pre_correction_unfingerprinted",
        "diagnostic_legacy_fingerprinted",
        "validated_fingerprinted",
    }
    artifact_paths = []
    retired_paths = []
    for experiment in experiments:
        if experiment.get("status") not in allowed_statuses:
            errors.append(
                "%s a un statut inconnu : %s"
                % (experiment.get("id"), experiment.get("status"))
            )
        for strategy in experiment.get("strategies", []):
            if strategy.startswith("rb1_") and strategy not in known_variants:
                errors.append(
                    "%s reference une variante inconnue %s"
                    % (experiment.get("id"), strategy)
                )
        for artifact in experiment.get("artifacts", []):
            artifact_paths.append(artifact["path"])
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
        for artifact in experiment.get("retired_artifacts", []):
            retired_paths.append(artifact.get("path"))
            digest = artifact.get("sha256", "")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                errors.append(
                    "SHA d'artefact retire invalide : %s"
                    % artifact.get("path")
                )
    if len(artifact_paths) != len(set(artifact_paths)):
        errors.append("un artefact est reference par plusieurs experiences")
    if len(retired_paths) != len(set(retired_paths)):
        errors.append("un artefact retire est reference plusieurs fois")
    overlap = set(artifact_paths).intersection(retired_paths)
    if overlap:
        errors.append("artefact a la fois actif et retire : %s" % sorted(overlap))
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
