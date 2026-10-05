"""Build a fasttext 0.9.3 wheel that compiles with MSVC on Python 3.10+.

fasttext 0.9.3 has no Windows wheel on PyPI, and its published source does not
compile with MSVC on Python 3.10 or later: lines 172 and 186 of
python/fasttext_module/fasttext/pybind/fasttext_pybind.cc use an unqualified
`ssize_t`. POSIX declares it in <sys/types.h>; on Windows only CPython's
pyconfig.h declared it globally, and only up to 3.9 (CPython commit c994ffe695,
bpo-11717, renamed it to Py_ssize_t). pybind11, from 2.2 through 3.1, defines
`ssize_t` only inside `namespace pybind11`, and this file does not import that
namespace, so no pybind11 version supplies the bare name.

This script downloads the sdist from PyPI, checks it against PyPI's SHA-256,
checks that the file to patch is byte-identical to the expected original (the
same file as upstream fastText's main branch), replaces those two `ssize_t`
with `py::ssize_t` (pybind11's alias for Py_ssize_t, a 64-bit signed integer on
Windows x64, the type the wrapped constructors Vector(int64_t) and
DenseMatrix(int64_t, int64_t) take), checks the result, and builds a wheel with
pip. The C++ library sources and TREN's model are not modified. MSVC also needs
/std:c++17 (the sources use std::string_view, and setup.py passes a language
standard only to non-MSVC compilers); it is added through the CL environment
variable for the build. Used by packaging/install_windows_deps.ps1.

    python packaging/build_fasttext_windows.py --wheel-dir DIR [--sdist FILE] [--patch-only]
"""
import argparse
import difflib
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

SDIST_URL = ("https://files.pythonhosted.org/packages/9f/3b/"
             "9a10b95eaf565358339162848863197c3f0a29b540ca22b2951df2d66a48/fasttext-0.9.3.tar.gz")
SDIST_SHA256 = "eb03f2ef6340c6ac9e4398a30026f05471da99381b307aafe2f56e4cd26baaef"
SRC_DIR_NAME = "fasttext-0.9.3"
PATCHED_FILE = "python/fasttext_module/fasttext/pybind/fasttext_pybind.cc"
ORIGINAL_SHA256 = "5f3f7b3a86aa4f035a76682eb9a619e4460638d5cb1f1111b48164c001fca12d"
PATCHED_SHA256 = "7a1500fc5ca7ceb1e90dce72bd7dc058d4f8171819e197300788ebebb7e2ea30"
REPLACEMENTS = [
    (b"      .def(py::init<ssize_t>())\n",
     b"      .def(py::init<py::ssize_t>())\n"),
    (b"      .def(py::init<ssize_t, ssize_t>())\n",
     b"      .def(py::init<py::ssize_t, py::ssize_t>())\n"),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha256(path, expected, what):
    actual = sha256_of(path)
    if actual != expected:
        raise SystemExit(f"{what}: SHA-256 {actual} does not match the expected {expected}")
    print(f"{what}: SHA-256 OK ({expected})")


def extract(sdist, dest):
    with tarfile.open(sdist, "r:gz") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
            return
        for member in tar.getmembers():
            name = os.path.normpath(member.name)
            if os.path.isabs(name) or name.startswith(".."):
                raise SystemExit(f"refusing to extract unsafe path {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"refusing to extract non-regular member {member.name!r}")
        tar.extractall(dest)


def apply_patch(src_dir):
    path = os.path.join(src_dir, *PATCHED_FILE.split("/"))
    require_sha256(path, ORIGINAL_SHA256, f"original {PATCHED_FILE}")
    with open(path, "rb") as f:
        original = f.read()
    patched = original
    for old, new in REPLACEMENTS:
        if patched.count(old) != 1:
            raise SystemExit(f"expected exactly one occurrence of {old!r} in {PATCHED_FILE}")
        patched = patched.replace(old, new)
    with open(path, "wb") as f:
        f.write(patched)
    require_sha256(path, PATCHED_SHA256, f"patched {PATCHED_FILE}")
    sys.stdout.writelines(difflib.unified_diff(
        original.decode("utf-8").splitlines(keepends=True),
        patched.decode("utf-8").splitlines(keepends=True),
        fromfile=f"a/{PATCHED_FILE}", tofile=f"b/{PATCHED_FILE}"))


def build_wheel(src_dir, wheel_dir):
    env = dict(os.environ)
    if os.name == "nt":
        env["CL"] = ("/std:c++17 " + env.get("CL", "")).strip()
        print(f"CL={env['CL']}")
    sys.stdout.flush()  # keep this script's output before pip's in the log
    # -v logs the build requirements pip resolves (setuptools, wheel, pybind11) and the compiler output.
    subprocess.run([sys.executable, "-m", "pip", "wheel", "-v", "--no-deps", "--wheel-dir", wheel_dir, src_dir],
                   env=env, check=True)
    wheels = glob.glob(os.path.join(wheel_dir, "fasttext-0.9.3-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one fasttext-0.9.3 wheel in {wheel_dir}, found {wheels}")
    print(f"built {wheels[0]} (SHA-256 {sha256_of(wheels[0])})")
    return wheels[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel-dir", required=True, help="where the built wheel is written")
    parser.add_argument("--sdist", help="use this fasttext-0.9.3.tar.gz instead of downloading it")
    parser.add_argument("--patch-only", action="store_true",
                        help="verify and patch the source into --wheel-dir/fasttext-0.9.3, but do not build")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    wheel_dir = os.path.abspath(args.wheel_dir)
    os.makedirs(wheel_dir, exist_ok=True)

    work = tempfile.mkdtemp(prefix="tren-fasttext-")
    try:
        sdist = args.sdist
        if not sdist:
            sdist = os.path.join(work, "fasttext-0.9.3.tar.gz")
            print(f"downloading {SDIST_URL}")
            with urllib.request.urlopen(SDIST_URL, timeout=120) as resp, open(sdist, "wb") as out:
                shutil.copyfileobj(resp, out)
        require_sha256(sdist, SDIST_SHA256, "fasttext-0.9.3.tar.gz")

        unpack_root = wheel_dir if args.patch_only else work
        extract(sdist, unpack_root)
        src_dir = os.path.join(unpack_root, SRC_DIR_NAME)
        apply_patch(src_dir)
        if not args.patch_only:
            build_wheel(src_dir, wheel_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
