"""Tests for the packaged-app "bundle isolation" check in packaging/tren_selftest.py,
without PyInstaller: modules are built the way PyInstaller's bootloader and torch
build theirs, and an archive is written in the CArchive layout PyInstaller's own
reader (PyInstaller.archive.readers.CArchiveReader) documents.
"""
import importlib.util
import marshal
import os
import struct
import sys
import types
import zlib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packaging"))

import tren_selftest as ts  # noqa: E402

BOOT_SOURCE = "import os\n\ndef helper():\n    return os.sep\n"


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    prefix = os.path.normcase(str(root)) + os.sep
    return root, (lambda p: os.path.normcase(os.path.abspath(p)).startswith(prefix))


def _bootloader_module(name, code, launch_dir, monkeypatch):
    """Execute `code` as PyInstaller's bootloader does for its archive modules:
    relative __file__, and a spec whose origin is the launch directory + name.py."""
    monkeypatch.chdir(launch_dir)
    module = types.ModuleType(name)
    module.__file__ = code.co_filename
    module.__spec__ = importlib.util.spec_from_file_location(name, code.co_filename)
    exec(code, module.__dict__)
    return module


def test_bootloader_module_from_archive_is_inside(bundle, tmp_path, monkeypatch):
    _, inside = bundle
    code = compile(BOOT_SOURCE, "bootmod.py", "exec")
    module = _bootloader_module("bootmod", code, tmp_path, monkeypatch)
    assert ts._outside_bundle_reason("bootmod", module, inside, {"bootmod": code}) is None


def test_relative_module_not_in_archive_is_outside(bundle, tmp_path, monkeypatch):
    _, inside = bundle
    code = compile(BOOT_SOURCE, "bootmod.py", "exec")
    module = _bootloader_module("bootmod", code, tmp_path, monkeypatch)
    assert ts._outside_bundle_reason("bootmod", module, inside, {})


def test_same_name_with_other_code_is_outside(bundle, tmp_path, monkeypatch):
    _, inside = bundle
    archived = compile(BOOT_SOURCE, "bootmod.py", "exec")
    impostor = compile("def helper():\n    return 'x'\n", "bootmod.py", "exec")
    module = _bootloader_module("bootmod", impostor, tmp_path, monkeypatch)
    assert ts._outside_bundle_reason("bootmod", module, inside, {"bootmod": archived})


def test_same_name_with_a_file_at_its_origin_is_outside(bundle, tmp_path, monkeypatch):
    _, inside = bundle
    code = compile(BOOT_SOURCE, "bootmod.py", "exec")
    (tmp_path / "bootmod.py").write_text(BOOT_SOURCE, encoding="utf-8")
    module = _bootloader_module("bootmod", code, tmp_path, monkeypatch)
    assert ts._outside_bundle_reason("bootmod", module, inside, {"bootmod": code})


def test_absolute_paths(bundle, tmp_path):
    root, inside = bundle
    in_bundle = types.ModuleType("a")
    in_bundle.__file__ = str(root / "a.pyc")
    outside = types.ModuleType("b")
    outside.__file__ = str(tmp_path / "b.py")
    assert ts._outside_bundle_reason("a", in_bundle, inside, {}) is None
    assert ts._outside_bundle_reason("b", outside, inside, {})


def test_module_object_is_judged_by_the_module_defining_its_type(bundle, tmp_path, monkeypatch):
    root, inside = bundle

    def module_object(owner_file):
        owner = types.ModuleType("fake_torch_ops_owner")
        owner.__file__ = owner_file
        exec("import types\n\nclass _Ops(types.ModuleType):\n    __file__ = '_ops.py'\n", owner.__dict__)
        monkeypatch.setitem(sys.modules, "fake_torch_ops_owner", owner)
        return owner._Ops("fake_torch_ops_owner.ops")

    ops = module_object(str(root / "fake_torch_ops_owner.pyc"))
    assert "__file__" not in vars(ops) and ops.__spec__ is None
    assert ts._outside_bundle_reason("fake_torch_ops_owner.ops", ops, inside, {}) is None
    ops = module_object(str(tmp_path / "fake_torch_ops_owner.py"))
    assert ts._outside_bundle_reason("fake_torch_ops_owner.ops", ops, inside, {})


def test_negative_controls_reject_every_external_module(bundle, tmp_path):
    _, inside = bundle
    archive = {"bootmod": compile(BOOT_SOURCE, "bootmod.py", "exec")}
    cwd = os.getcwd()
    reasons = ts._isolation_negative_controls(inside, archive, str(tmp_path / "negative"))
    assert os.getcwd() == cwd and "tren_external_probe" not in sys.modules
    assert len(reasons) == 5 and all(reasons.values())


def test_carchive_parser_reads_module_entries(tmp_path):
    code = compile(BOOT_SOURCE, "bootmod.py", "exec")
    blob = zlib.compress(marshal.dumps(code))
    magic = b"MEI\014\013\012\013\016"

    def entry(name, offset, length, typecode):
        padded = name.encode() + b"\0" * (16 - (18 + len(name)) % 16)
        return struct.pack("!IIIIBc", 18 + len(padded), offset, length, 0, 1, typecode) + padded

    toc = entry("bootmod", 0, len(blob), b"m") + entry("somedata", 0, len(blob), b"x")
    archive_body = blob + toc
    cookie = struct.pack("!8sIIII64s", magic, len(archive_body) + 88, len(blob), len(toc), 311, b"python311.dll")
    exe = tmp_path / "app.exe"
    exe.write_bytes(b"bootloader " + magic + b" bytes" + archive_body + cookie)
    assert ts._carchive_module_code(str(exe)) == {"bootmod": code}
