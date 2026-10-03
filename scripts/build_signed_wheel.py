"""Repack the verified Qlib Windows wheel with tracked, Python-only source changes.

No build backend or installed Qlib is required. A dirty source checkout is useful
only for development; release builds should omit --allow-dirty.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile


BASE_COMMIT = "da920b7f954f48ab1bb64117c976710de198373e"
BASE_VERSION = "0.9.7"
VERSION = "0.9.7+tubby.3"
# ZipInfo records the creating OS (0 on Windows, 3 elsewhere); fixing it makes a wheel built on any
# platform byte-identical, so one pinned digest verifies on Windows and macOS alike.
CREATE_SYSTEM = 3
DIST = "pyqlib"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def git_paths(root: Path, *args: str) -> set[str]:
    return {os.fsdecode(p).replace("\\", "/") for p in run_git(root, *args).split(b"\0") if p}


def source_overlay(root: Path, allow_dirty: bool) -> tuple[dict[str, bytes], str, bool]:
    actual = run_git(root, "rev-parse", "--show-toplevel").decode().strip().replace("\\", "/")
    if actual.lower() != root.resolve().as_posix().lower():
        raise ValueError("script must run against its own Git checkout root")
    run_git(root, "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}")
    commit = run_git(root, "rev-parse", "HEAD").decode().strip()
    if run_git(root, "merge-base", BASE_COMMIT, "HEAD").decode().strip() != BASE_COMMIT:
        raise ValueError("checkout is not descended from the pinned upstream commit")
    status = run_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise ValueError("source checkout is dirty; commit it or pass --allow-dirty for development")
    tracked = git_paths(root, "ls-files", "-z", "--cached")
    baseline = git_paths(root, "ls-tree", "-rz", "--name-only", BASE_COMMIT, "--", "qlib")
    # The overlay can never silently omit a newly created Qlib source module.
    untracked_qlib = git_paths(root, "ls-files", "-z", "--others", "--exclude-standard", "--", "qlib")
    if untracked_qlib:
        raise ValueError("untracked qlib files must be staged or committed: " + ", ".join(sorted(untracked_qlib)))
    forbidden = {"setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"}
    changed = git_paths(root, "diff", "-z", "--name-only", BASE_COMMIT, "--")
    changed |= git_paths(root, "diff", "-z", "--name-only", "--cached", BASE_COMMIT, "--")
    # Diff does not list a staged intent-to-add reliably; compare tracked paths too.
    changed |= tracked - git_paths(root, "ls-tree", "-rz", "--name-only", BASE_COMMIT)
    working_changes = git_paths(root, "diff", "-z", "--name-only", "HEAD", "--")
    staged_changes = git_paths(root, "diff", "-z", "--name-only", "--cached", "HEAD", "--")
    for path in sorted(changed):
        if path in forbidden or path.endswith(('.pyx', '.pxd', '.c', '.cpp')) and path.startswith('qlib/'):
            raise ValueError(f"native/build source changed: {path}")
        if path.startswith("qlib/") and not path.endswith(".py"):
            raise ValueError(f"non-Python qlib change rejected: {path}")
    overlay = {}
    for path in sorted(changed):
        if not path.startswith("qlib/"):
            continue
        if path not in tracked or not (root / path).is_file():
            raise ValueError(f"deleted or unavailable qlib Python file: {path}")
        # Git blobs are stable across checkout newline settings. In development,
        # prefer the worktree only for unstaged edits, then the index for staged edits.
        if path in working_changes:
            overlay[path] = (root / path).read_bytes()
        elif path in staged_changes:
            overlay[path] = run_git(root, "show", f":{path}")
        else:
            overlay[path] = run_git(root, "show", f"HEAD:{path}")
    init = overlay.get("qlib/__init__.py")
    if init is None:
        raise ValueError("qlib/__init__.py must set the derivative version")
    if not re.search(rb'^__version__\s*=\s*[\'\"]' + re.escape(VERSION.encode()) + rb'[\'\"]', init, re.M):
        raise ValueError("qlib/__init__.py version does not match derivative wheel")
    return overlay, commit, dirty


def checked_base(path: Path, expected_sha: str) -> tuple[dict[str, bytes], str, dict[str, int]]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha):
        raise ValueError("--base-sha256 must be a 64-character hex digest")
    if sha256(path.read_bytes()) != expected_sha.lower():
        raise ValueError("base wheel SHA256 mismatch")
    match = re.fullmatch(r"pyqlib-0\.9\.7-([^-]+)-([^-]+)-([^-]+)\.whl", path.name)
    if not match:
        raise ValueError("base wheel filename must identify pyqlib 0.9.7")
    tag = "-".join(match.groups())
    prefix = f"{DIST}-{BASE_VERSION}.dist-info/"
    entries: dict[str, bytes] = {}
    attrs: dict[str, int] = {}
    with zipfile.ZipFile(path) as wheel:
        for info in wheel.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if name in entries or not parts or name.startswith("/") or ".." in parts or "\\" in name:
                raise ValueError(f"unsafe or duplicate wheel member: {name}")
            if name.endswith("/"):
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symlink wheel member: {name}")
            entries[name] = wheel.read(info)
            attrs[name] = info.external_attr
    if prefix + "METADATA" not in entries or prefix + "WHEEL" not in entries or prefix + "RECORD" not in entries:
        raise ValueError("base wheel metadata is incomplete")
    metadata = entries[prefix + "METADATA"]
    if not re.search(rb"(?m)^Name: pyqlib\r?$", metadata) or not re.search(rb"(?m)^Version: 0\.9\.7\r?$", metadata):
        raise ValueError("base wheel package/version metadata mismatch")
    if not re.search(rf"(?m)^Tag: {re.escape(tag)}\r?$", entries[prefix + "WHEEL"].decode()):
        raise ValueError("base wheel platform tag mismatch")
    if not any(n.endswith(".pyd") or n.endswith(".so") for n in entries):
        raise ValueError("base wheel contains no native library")
    return entries, tag, attrs


def build(root: Path, base_wheel: Path, base_sha256: str, output_dir: Path, allow_dirty: bool = False) -> Path:
    entries, tag, attrs = checked_base(base_wheel, base_sha256)
    overlay, commit, dirty = source_overlay(root, allow_dirty)
    old_prefix = f"{DIST}-{BASE_VERSION}.dist-info/"
    new_prefix = f"{DIST}-{VERSION}.dist-info/"
    record = old_prefix + "RECORD"
    entries.pop(record)
    attrs.pop(record)
    renamed = {}
    renamed_attrs = {}
    for name, data in entries.items():
        target = new_prefix + name[len(old_prefix):] if name.startswith(old_prefix) else name
        if target in renamed:
            raise ValueError(f"wheel member collision: {target}")
        renamed[target] = data
        renamed_attrs[target] = attrs[name]
    metadata_path = new_prefix + "METADATA"
    renamed[metadata_path], count = re.subn(
        rb"(?m)^Version: 0\.9\.7(\r?)$", lambda m: b"Version: " + VERSION.encode() + m[1], renamed[metadata_path]
    )
    if count != 1:
        raise ValueError("unexpected METADATA version fields")
    for name, data in overlay.items():
        if name not in renamed and name in attrs:
            raise ValueError(f"wheel member collision: {name}")
        renamed[name] = data
    # The upstream license is pinned in Git; include it even if the wheel lacks it.
    license_bytes = run_git(root, "show", f"{BASE_COMMIT}:LICENSE")
    license_paths = [n for n in renamed if n == "LICENSE" or n.endswith("/LICENSE")]
    if not license_paths:
        renamed[new_prefix + "licenses/LICENSE"] = license_bytes
    native = {n: sha256(data) for n, data in renamed.items() if n.endswith((".pyd", ".so"))}
    manifest = {
        "base_wheel_sha256": base_sha256.lower(),
        "changed_python_sha256": {n: sha256(d) for n, d in sorted(overlay.items())},
        "commit": commit,
        "dirty": dirty,
        "native_sha256": dict(sorted(native.items())),
        "upstream_commit": BASE_COMMIT,
    }
    renamed["qlib/_tubby_build.json"] = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    record_path = new_prefix + "RECORD"
    rows = []
    for name, data in sorted(renamed.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append((name, "sha256=" + digest, str(len(data))))
    rows.append((record_path, "", ""))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    renamed[record_path] = buffer.getvalue().encode()
    filename = f"{DIST}-{VERSION}-{tag}.whl"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    if destination.exists():
        raise FileExistsError(destination)
    fd, temp_name = tempfile.mkstemp(prefix=".signed-wheel-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as file, zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
            for name, data in sorted(renamed.items()):
                info = zipfile.ZipInfo(name, FIXED_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = CREATE_SYSTEM
                info.external_attr = renamed_attrs.get(name, 0o100644 << 16)
                wheel.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.link(temp_name, destination)  # Atomic creation; never replaces an existing wheel.
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wheel", required=True, type=Path)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    try:
        path = build(Path(__file__).resolve().parent.parent, args.base_wheel, args.base_sha256, args.output_dir, args.allow_dirty)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(path)


if __name__ == "__main__":
    main()
