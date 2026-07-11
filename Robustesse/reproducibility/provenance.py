"""Provenance de calcul fondee sur le contenu reel des sources et donnees.

Le hash Git seul ne suffit pas : un calcul peut etre lance depuis un worktree
sale ou depuis une copie sans ``.git`` sur le mesocentre. L'empreinte definie
ici couvre donc chaque fichier d'entree et les parametres scientifiques, puis
ajoute les informations Git uniquement comme metadonnees explicatives.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def sha256_file(path):
    """Retourne le SHA-256 hexadecimal d'un fichier."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _run_git(repo_root, args):
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root)] + list(args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_git_root(start):
    """Trouve la racine Git, ou ``None`` si le code est une copie exportee."""
    root = _run_git(Path(start).resolve(), ["rev-parse", "--show-toplevel"])
    return Path(root) if root else None


def _display_path(path, repo_root):
    resolved = Path(path).resolve()
    if repo_root is not None:
        try:
            return resolved.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


def build_provenance(experiment_id, files, parameters, repo_root=None):
    """Construit une fiche de provenance et son empreinte deterministe.

    ``files`` doit contenir toutes les sources, tables et donnees qui peuvent
    changer le resultat. L'horodatage, le commit et l'environnement ne sont pas
    inclus dans l'empreinte : deux executions bit-a-bit du meme protocole
    partagent ainsi le meme identifiant de run.
    """
    paths = [Path(p).resolve() for p in files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Entrees de provenance absentes : %s" % missing)

    if repo_root is None and paths:
        repo_root = find_git_root(paths[0].parent)
    elif repo_root is not None:
        repo_root = Path(repo_root).resolve()

    entries = []
    for path in sorted(paths, key=lambda p: _display_path(p, repo_root)):
        entries.append(
            {
                "path": _display_path(path, repo_root),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "parameters": parameters,
        "inputs": entries,
    }
    fingerprint = hashlib.sha256(_json_bytes(identity)).hexdigest()

    git_commit = _run_git(repo_root, ["rev-parse", "HEAD"]) if repo_root else None
    git_status = _run_git(repo_root, ["status", "--porcelain"]) if repo_root else None
    relevant_status = None
    if repo_root is not None:
        relative = []
        for path in paths:
            try:
                relative.append(str(path.relative_to(repo_root)))
            except ValueError:
                continue
        if relative:
            relevant_status = _run_git(
                repo_root, ["status", "--porcelain", "--"] + relative
            )

    try:
        import numpy as np

        numpy_version = np.__version__
    except ImportError:
        numpy_version = None

    return {
        **identity,
        "run_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "repo_dirty": bool(git_status),
            "relevant_inputs_dirty": bool(relevant_status),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": numpy_version,
            "platform": platform.platform(),
        },
    }


def provenance_header_lines(record):
    """Lignes de commentaire compactes a inserer dans une sortie texte."""
    git = record.get("git", {})
    return [
        "# provenance_schema=%s" % record["schema_version"],
        "# experiment_id=%s" % record["experiment_id"],
        "# run_fingerprint=%s" % record["run_fingerprint"],
        "# git_commit=%s" % (git.get("commit") or "unavailable"),
        "# git_repo_dirty=%s | relevant_inputs_dirty=%s"
        % (int(bool(git.get("repo_dirty"))), int(bool(git.get("relevant_inputs_dirty")))),
    ]


def read_provenance_header(path, max_lines=40):
    """Lit les champs ``cle=valeur`` du preambule commente d'un TXT."""
    fields = {}
    try:
        with open(path, encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= max_lines:
                    break
                if not line.startswith("#"):
                    if line.strip():
                        break
                    continue
                body = line[1:].strip()
                if "=" in body and " | " not in body:
                    key, value = body.split("=", 1)
                    fields[key.strip()] = value.strip()
    except FileNotFoundError:
        return {}
    return fields


def validate_cache(path, expected_record, allow_legacy=False):
    """Valide l'empreinte d'une sortie reutilisee comme cache."""
    if not os.path.isfile(path):
        return False, "absent"
    fields = read_provenance_header(path)
    actual = fields.get("run_fingerprint")
    if actual is None:
        if allow_legacy:
            return True, "legacy explicitement autorise, sans preuve de sources"
        return False, "legacy sans empreinte"
    expected = expected_record["run_fingerprint"]
    if actual != expected:
        return False, "empreinte differente (%s != %s)" % (actual[:12], expected[:12])
    return True, "empreinte identique"


def write_provenance_sidecar(path, record, artifacts=()):
    """Ecrit une fiche JSON, avec hash des artefacts deja produits."""
    enriched = dict(record)
    enriched["artifacts"] = [
        {
            "path": str(Path(p)),
            "size": Path(p).stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in artifacts
        if Path(p).is_file()
    ]
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(enriched, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def fingerprinted_run_dir(base_dir, slug, record):
    """Dossier immuable par contenu pour une nouvelle execution."""
    return Path(base_dir) / "runs" / (slug + "_" + record["run_fingerprint"][:12])

