import tempfile
import unittest
import os
from pathlib import Path

import sys

ROBUSTESSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBUSTESSE))

from reproducibility.provenance import (
    acquire_run_lock,
    build_provenance,
    provenance_header_lines,
    validate_cache,
    validate_append_only_artifact,
    validate_sidecar_artifact,
    write_provenance_sidecar,
)


class TestProvenance(unittest.TestCase):
    def test_content_and_parameters_determine_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text("alpha\n", encoding="utf-8")
            a = build_provenance("exp", [path], {"seed": 1})
            b = build_provenance("exp", [path], {"seed": 1})
            self.assertEqual(a["run_fingerprint"], b["run_fingerprint"])
            path.write_text("beta\n", encoding="utf-8")
            c = build_provenance("exp", [path], {"seed": 1})
            self.assertNotEqual(a["run_fingerprint"], c["run_fingerprint"])
            d = build_provenance("exp", [path], {"seed": 2})
            self.assertNotEqual(c["run_fingerprint"], d["run_fingerprint"])

    def test_cache_requires_matching_header(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_text("x", encoding="utf-8")
            record = build_provenance("exp", [source], {"seed": 1})
            cache = Path(directory) / "cache.txt"
            cache.write_text("\n".join(provenance_header_lines(record)) + "\nDATA\n", encoding="utf-8")
            self.assertTrue(validate_cache(cache, record)[0])
            other = build_provenance("exp", [source], {"seed": 2})
            self.assertFalse(validate_cache(cache, other)[0])
            legacy = Path(directory) / "legacy.txt"
            legacy.write_text("# ancien\n", encoding="utf-8")
            self.assertFalse(validate_cache(legacy, record)[0])
            self.assertTrue(validate_cache(legacy, record, allow_legacy=True)[0])

    def test_logical_path_makes_external_file_portable(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = Path(first) / "sidelec.csv"
            b = Path(second) / "sidelec.csv"
            a.write_text("1;2\n", encoding="utf-8")
            b.write_text("1;2\n", encoding="utf-8")
            rec_a = build_provenance("exp", [("data/sidelec.csv", a)], {"seed": 1})
            rec_b = build_provenance("exp", [("data/sidelec.csv", b)], {"seed": 1})
            self.assertEqual(rec_a["run_fingerprint"], rec_b["run_fingerprint"])

    def test_sidecar_detects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            artifact = Path(directory) / "cache.npz"
            sidecar = Path(directory) / "cache.provenance.json"
            source.write_text("source", encoding="utf-8")
            artifact.write_bytes(b"first")
            record = build_provenance("exp", [source], {"seed": 1})
            write_provenance_sidecar(sidecar, record, [artifact])
            self.assertTrue(validate_sidecar_artifact(sidecar, artifact, record)[0])
            artifact.write_bytes(b"changed")
            self.assertFalse(validate_sidecar_artifact(sidecar, artifact, record)[0])

    def test_append_only_recovery_preserves_valid_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            journal = Path(directory) / "raw.tsv"
            sidecar = Path(directory) / "raw.tsv.provenance.json"
            source.write_text("source", encoding="utf-8")
            journal.write_bytes(b"header\nrow1\n")
            record = build_provenance("exp", [source], {"seed": 1})
            write_provenance_sidecar(sidecar, record, [journal])
            with journal.open("ab") as stream:
                stream.write(b"row2\n")
            ok, reason = validate_append_only_artifact(sidecar, journal, record)
            self.assertTrue(ok, reason)
            data = journal.read_bytes()
            journal.write_bytes(b"X" + data[1:])
            self.assertFalse(validate_append_only_artifact(sidecar, journal, record)[0])

    @unittest.skipUnless(os.name == "posix", "verrou flock POSIX")
    def test_run_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            first = acquire_run_lock(directory)
            try:
                with self.assertRaises(RuntimeError):
                    acquire_run_lock(directory)
            finally:
                first.close()
            second = acquire_run_lock(directory)
            second.close()


if __name__ == "__main__":
    unittest.main()
