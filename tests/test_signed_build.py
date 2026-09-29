"""Focused checks for the binary-preserving signed wheel packager."""

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_signed_wheel.py"
SPEC = importlib.util.spec_from_file_location("build_signed_wheel", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class SignedBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def base_wheel(self, metadata=b"Metadata-Version: 2.4\nName: pyqlib\nVersion: 0.9.7\n"):
        wheel = self.root / "pyqlib-0.9.7-cp312-cp312-win_amd64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("qlib/__init__.py", b'__version__ = "0.9.7"\n')
            archive.writestr("qlib/data/_libs/rolling.cp312-win_amd64.pyd", b"native-data")
            archive.writestr("pyqlib-0.9.7.dist-info/METADATA", metadata)
            archive.writestr("pyqlib-0.9.7.dist-info/WHEEL", b"Wheel-Version: 1.0\nTag: cp312-cp312-win_amd64\n")
            archive.writestr("pyqlib-0.9.7.dist-info/RECORD", b"test,,\n")
        return wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()

    def test_hash_and_package_metadata_rejected(self):
        wheel, digest = self.base_wheel()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            builder.checked_base(wheel, "0" * 64)
        bad, bad_digest = self.base_wheel(b"Name: other\nVersion: 0.9.7\n")
        with self.assertRaisesRegex(ValueError, "package/version"):
            builder.checked_base(bad, bad_digest)

    def test_build_preserves_native_rebuilds_record_and_refuses_collision(self):
        wheel, digest = self.base_wheel()
        overlay = {"qlib/__init__.py": b'__version__ = "0.9.7+tubby.1"\n'}
        output = self.root / "out"
        with mock.patch.object(builder, "source_overlay", return_value=(overlay, "a" * 40, False)), mock.patch.object(
            builder, "run_git", return_value=b"MIT license\n"
        ):
            target = builder.build(self.root, wheel, digest, output)
            with self.assertRaises(FileExistsError):
                builder.build(self.root, wheel, digest, output)
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            self.assertEqual(names, sorted(names))
            self.assertEqual(archive.read("qlib/data/_libs/rolling.cp312-win_amd64.pyd"), b"native-data")
            prefix = "pyqlib-0.9.7+tubby.1.dist-info/"
            self.assertIn(prefix + "licenses/LICENSE", names)
            self.assertIn(b"Version: 0.9.7+tubby.1", archive.read(prefix + "METADATA"))
            manifest = json.loads(archive.read("qlib/_tubby_build.json"))
            self.assertEqual(manifest["native_sha256"]["qlib/data/_libs/rolling.cp312-win_amd64.pyd"], hashlib.sha256(b"native-data").hexdigest())
            rows = list(csv.reader(io.StringIO(archive.read(prefix + "RECORD").decode())))
            self.assertEqual({row[0] for row in rows}, set(names))
            self.assertEqual(rows[-1], [prefix + "RECORD", "", ""])
            self.assertTrue(all(info.date_time == builder.FIXED_TIME for info in archive.infolist()))

    def test_dirty_and_native_source_change_rejected(self):
        repo = self.root / "source"
        (repo / "qlib" / "data" / "_libs").mkdir(parents=True)
        (repo / "qlib" / "__init__.py").write_text('__version__ = "0.9.7"\n')
        native = repo / "qlib" / "data" / "_libs" / "rolling.pyx"
        native.write_text("cdef int n = 1\n")
        (repo / "LICENSE").write_text("MIT\n")
        def git(*args):
            subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
        git("init", "-q")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "Test")
        git("add", ".")
        git("commit", "-qm", "baseline")
        baseline = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        native.write_text("cdef int n = 2\n")
        with mock.patch.object(builder, "BASE_COMMIT", baseline):
            with self.assertRaisesRegex(ValueError, "dirty"):
                builder.source_overlay(repo, False)
            with self.assertRaisesRegex(ValueError, "native/build"):
                builder.source_overlay(repo, True)
            native.write_text("cdef int n = 1\n")
            (repo / "qlib" / "new_module.py").write_text("pass\n")
            with self.assertRaisesRegex(ValueError, "untracked qlib"):
                builder.source_overlay(repo, True)


if __name__ == "__main__":
    unittest.main()
