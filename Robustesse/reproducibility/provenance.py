"""Provenance de calcul fondee sur le contenu reel des sources et donnees.

Le hash Git seul ne suffit pas : un calcul peut etre lance depuis un worktree
sale ou depuis une copie sans ``.git`` sur le mesocentre. L'empreinte definie
ici couvre donc chaque fichier d'entree et les parametres scientifiques, puis
ajoute les informations Git uniquement comme metadonnees explicatives.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3


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


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def _normalise_files(files, repo_root):
    """Retourne ``(nom_logique, chemin)`` sans faire entrer le chemin physique
    des donnees externes dans l'identite du calcul.

    Un element peut etre un simple chemin, ou un couple
    ``("nom/logique/stable", chemin_physique)``.
    """
    normalised = []
    for item in files:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            logical, physical = item
            logical = str(logical).replace("\\", "/")
            path = Path(physical).resolve()
        else:
            path = Path(item).resolve()
            logical = _display_path(path, repo_root)
        normalised.append((logical, path))
    labels = [logical for logical, _ in normalised]
    if len(labels) != len(set(labels)):
        raise ValueError("noms logiques de provenance dupliques")
    return normalised


def build_provenance(experiment_id, files, parameters, repo_root=None):
    """Construit une fiche de provenance et son empreinte deterministe.

    ``files`` doit contenir toutes les sources, tables et donnees qui peuvent
    changer le resultat. Les versions Python/NumPy/SciPy/SymPy font partie de
    l'identite numerique. L'horodatage, le commit, l'hote et les variables Slurm
    restent des metadonnees : deux executions du meme protocole dans le meme
    runtime numerique partagent ainsi le meme identifiant de run.
    """
    raw_paths = [item[1] if isinstance(item, (tuple, list)) and len(item) == 2 else item
                 for item in files]
    paths = [Path(p).resolve() for p in raw_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Entrees de provenance absentes : %s" % missing)

    if repo_root is None and paths:
        repo_root = find_git_root(paths[0].parent)
    elif repo_root is not None:
        repo_root = Path(repo_root).resolve()

    logical_files = _normalise_files(files, repo_root)
    entries = []
    locations = []
    for logical, path in sorted(logical_files, key=lambda item: item[0]):
        entries.append(
            {
                "path": logical,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        locations.append({"path": logical, "resolved_path": path.as_posix()})

    numerical_runtime = {
        "python": platform.python_version(),
        "numpy": _distribution_version("numpy"),
        "scipy": _distribution_version("scipy"),
        "sympy": _distribution_version("sympy"),
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "parameters": parameters,
        "inputs": entries,
        "numerical_runtime": numerical_runtime,
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

    slurm = {
        key: os.environ.get(key)
        for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK", "SLURM_SUBMIT_DIR")
        if os.environ.get(key) is not None
    }

    return {
        **identity,
        "input_locations": locations,
        "run_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "repo_dirty": None if git_status is None else bool(git_status),
            "relevant_inputs_dirty": (
                None if relevant_status is None else bool(relevant_status)
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": numerical_runtime["numpy"],
            "scipy": numerical_runtime["scipy"],
            "sympy": numerical_runtime["sympy"],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "argv": list(sys.argv),
            "slurm": slurm,
        },
    }


def provenance_header_lines(record):
    """Lignes de commentaire compactes a inserer dans une sortie texte."""
    git = record.get("git", {})
    def flag(value):
        return "unavailable" if value is None else str(int(bool(value)))
    return [
        "# provenance_schema=%s" % record["schema_version"],
        "# experiment_id=%s" % record["experiment_id"],
        "# run_fingerprint=%s" % record["run_fingerprint"],
        "# git_commit=%s" % (git.get("commit") or "unavailable"),
        "# git_repo_dirty=%s | relevant_inputs_dirty=%s"
        % (flag(git.get("repo_dirty")), flag(git.get("relevant_inputs_dirty"))),
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
    path = Path(path)
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
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(enriched, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_sidecar_artifact(sidecar_path, artifact_path, expected_record):
    """Valide l'empreinte du run et le SHA d'un artefact de cache binaire."""
    sidecar_path = Path(sidecar_path)
    artifact_path = Path(artifact_path)
    if not sidecar_path.is_file() or not artifact_path.is_file():
        return False, "artefact ou fiche absent"
    try:
        with open(sidecar_path, encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, ValueError) as exc:
        return False, "fiche illisible : %s" % exc
    if record.get("run_fingerprint") != expected_record["run_fingerprint"]:
        return False, "empreinte de fiche differente"
    actual = sha256_file(artifact_path)
    for artifact in record.get("artifacts", []):
        if Path(artifact.get("path", "")).name == artifact_path.name:
            if artifact.get("sha256") == actual:
                return True, "empreinte et SHA identiques"
            return False, "SHA artefact different"
    return False, "artefact absent de la fiche"


def validate_append_only_artifact(sidecar_path, artifact_path, expected_record):
    """Valide un journal, y compris une extension apres un crash pre-sidecar.

    Si le SHA courant differe mais que le prefixe de taille enregistree possede
    toujours le SHA consigne, le fichier n'a subi qu'un append. L'appelant doit
    encore parser integralement les nouvelles lignes avant de ratifier le SHA.
    """
    sidecar_path = Path(sidecar_path)
    artifact_path = Path(artifact_path)
    if not sidecar_path.is_file() or not artifact_path.is_file():
        return False, "artefact ou fiche absent"
    try:
        with open(sidecar_path, encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, ValueError) as exc:
        return False, "fiche illisible : %s" % exc
    if record.get("run_fingerprint") != expected_record["run_fingerprint"]:
        return False, "empreinte de fiche differente"
    entry = next(
        (item for item in record.get("artifacts", [])
         if Path(item.get("path", "")).name == artifact_path.name),
        None,
    )
    if entry is None:
        return False, "artefact absent de la fiche"
    actual_size = artifact_path.stat().st_size
    recorded_size = int(entry.get("size", -1))
    if actual_size == recorded_size:
        same = sha256_file(artifact_path) == entry.get("sha256")
        return same, "SHA identique" if same else "SHA different"
    if actual_size < recorded_size or recorded_size < 0:
        return False, "journal tronque"
    digest = hashlib.sha256()
    remaining = recorded_size
    with open(artifact_path, "rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                return False, "prefixe tronque"
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != entry.get("sha256"):
        return False, "prefixe historique modifie"
    return True, "extension append-only a parser"


def fingerprinted_run_dir(base_dir, slug, record):
    """Dossier immuable par contenu pour une nouvelle execution."""
    return Path(base_dir) / "runs" / (slug + "_" + record["run_fingerprint"][:12])


def acquire_run_lock(run_dir):
    """Verrouille un dossier de run sur POSIX pour eviter un double append."""
    path = Path(run_dir) / ".run.lock"
    stream = open(path, "a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        # Windows : pas de verrou inter-processus sans dependance additionnelle.
        # Les jobs Slurm, cible critique, sont POSIX et donc proteges.
        return stream
    except BlockingIOError:
        stream.close()
        raise RuntimeError("un autre processus utilise deja %s" % run_dir)
    stream.seek(0)
    stream.truncate()
    stream.write("pid=%d host=%s\n" % (os.getpid(), socket.gethostname()))
    stream.flush()
    return stream
