"""Collect the license files of the third-party code a PyInstaller build bundles.

packaging/TREN.spec and packaging/TREN_windows.spec call this right after Analysis(...).
It does not start from requirements.txt: every file the analysis put into the bundle
(pure modules, binaries, data files) is mapped to the installed distribution whose RECORD
lists it, so packages that arrive indirectly, or only because they are installed in the
build environment, are covered as well. For each such distribution the license files
shipped in its metadata (or, failing that, in its package folder) are copied to
<out>/<name>-<version>/, and <out>/PACKAGES.txt lists every distribution with its declared
license and says plainly when no license file was found in the installed distribution
(which is not the same as having no license). Bundled
binaries that no distribution owns (the Python runtime's native libraries) are listed with
the interpreter, OpenSSL and Tcl/Tk versions of the build, and the Python and Tcl/Tk
license texts are copied when the interpreter ships them.

    python packaging/collect_licenses.py --out DIR --sources-json TOC.json

The command line only exists to try the collector without a build: TOC.json holds
{"pure": [[dest, src, type], ...], "binaries": [...], "datas": [...]}. Standard library only.
"""
import argparse
import glob
import json
import os
import re
import shutil
import ssl
import sys
import sysconfig
from importlib import metadata

LICENSE_NAME = re.compile(r"^(licen[cs]e|copying|notice|copyright)", re.IGNORECASE)
PACKAGE_LICENSE_NAME = re.compile(r"licen[cs]e|copying|notice", re.IGNORECASE)
LICENSE_DIRS = ("licenses", "license_files")
# PyInstaller's bootloader is the executable itself, which no analysis list contains.
ALWAYS_INCLUDE = ("pyinstaller",)


def _norm(path):
    return os.path.normcase(os.path.realpath(path))


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _owner_index(distributions):
    owners = {}
    for dist in distributions:
        for f in dist.files or ():
            try:
                owners[_norm(dist.locate_file(f))] = dist
            except OSError:
                continue
    return owners


def license_files(dist):
    """(file, path below the distribution's folder) for the license files it ships.

    These are the license files in its metadata folder; only when there are none, the license
    and notice files directly in its package folder (onnxruntime ships its LICENSE and
    ThirdPartyNotices.txt there).
    """
    found, fallback = [], []
    for f in dist.files or ():
        parts = f.parts
        if len(parts) >= 2 and parts[0].endswith((".dist-info", ".egg-info")):
            rel = parts[1:]
            if LICENSE_NAME.match(rel[-1]) or rel[0].lower() in LICENSE_DIRS:
                found.append((f, "/".join(rel)))
        elif len(parts) == 2 and PACKAGE_LICENSE_NAME.search(parts[1]) and f.suffix.lower() in (".txt", ".md", ".rst", ""):
            fallback.append((f, "/".join(parts)))
    return found or fallback


def declared_license(dist):
    md = dist.metadata
    expression = (md.get("License-Expression") or "").strip()
    if expression:
        return expression
    text = (md.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 60 and text.upper() != "UNKNOWN":
        return text
    classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or () if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    return "see the license files" if text else "not declared in the metadata"


def _python_license():
    stdlib = sysconfig.get_paths().get("stdlib") or ""
    for base in (stdlib, os.path.dirname(stdlib), sys.base_prefix):
        for name in ("LICENSE.txt", "LICENSE"):
            if base and os.path.isfile(os.path.join(base, name)):
                return os.path.join(base, name)
    return None


def _tcl_tk_license():
    try:
        import tkinter
        library = tkinter.Tcl().eval("info library")
    except Exception:
        return None
    root = os.path.dirname(library)
    candidates = [os.path.join(library, "license.terms")]
    candidates += sorted(glob.glob(os.path.join(root, "tk*", "license.terms")))
    candidates += sorted(glob.glob(os.path.join(root, "tk*", "demos", "license.terms")))
    return next((p for p in candidates if os.path.isfile(p)), None)


def _tcl_tk_version():
    try:
        import tkinter
        return tkinter.Tcl().eval("info patchlevel")
    except Exception:
        return "unknown"


def collect(sources, out_dir, binaries=()):
    """Copy the license files of every distribution that owns one of `sources` into `out_dir`.

    `sources` are the files the build bundles; `binaries` are the bundled (destination, source)
    pairs of the binaries, reported when no distribution owns them. Returns a summary dict.
    """
    distributions = [d for d in metadata.distributions() if d.metadata["Name"]]
    owners = _owner_index(distributions)
    used = {}
    for src in sources:
        dist = owners.get(_norm(src))
        if dist is not None:
            used[_canonical(dist.metadata["Name"])] = dist
    for dist in distributions:
        if _canonical(dist.metadata["Name"]) in ALWAYS_INCLUDE:
            used[_canonical(dist.metadata["Name"])] = dist

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rows, missing = [], []
    for key in sorted(used):
        dist = used[key]
        folder = f"{key}-{dist.version}"
        copied = 0
        for f, rel in license_files(dist):
            dst = os.path.join(out_dir, folder, *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.copyfile(dist.locate_file(f), dst)
            except OSError:
                continue
            copied += 1
        rows.append((dist.metadata["Name"], dist.version, declared_license(dist), copied))
        if not copied:
            missing.append(f"{dist.metadata['Name']} {dist.version}")

    runtime = {}
    for label, finder, name in (("Python", _python_license, "python-LICENSE.txt"),
                                ("Tcl/Tk", _tcl_tk_license, "tcl-tk-license.terms")):
        path = finder()
        runtime[label] = bool(path)
        if path:
            os.makedirs(os.path.join(out_dir, "runtime"), exist_ok=True)
            shutil.copyfile(path, os.path.join(out_dir, "runtime", name))

    native = sorted({os.path.basename(dest) for dest, src in binaries
                     if _norm(src) not in owners and not dest.lower().endswith((".so", ".pyd"))})
    _write_index(out_dir, rows, missing, runtime, native)
    print(f"collect_licenses: {len(rows)} distributions, {len(missing)} without a license file in the installed package")
    for name in missing:
        print(f"collect_licenses: WARNING no license file for {name}; check its project page")
    for label, ok in runtime.items():
        if not ok:
            print(f"collect_licenses: WARNING {label} license text not found in this interpreter")
    return {"distributions": rows, "without_license_file": missing, "runtime": runtime, "native": native}


def _write_index(out_dir, rows, missing, runtime, native):
    lines = [
        "Third-party license index (generated by packaging/collect_licenses.py at build time)",
        "",
        "Every Python distribution below owns at least one file in this application. Its license",
        "files, as shipped with the installed package, are in the folder <name>-<version>/. TREN's",
        "own license and the notices TREN writes by hand are in LICENSE and THIRD_PARTY_NOTICES.md.",
        "",
        f"Build: Python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}, Tcl/Tk {_tcl_tk_version()}",
        "",
        "name | version | declared license | license files copied",
    ]
    lines += [f"{n} | {v} | {lic} | {count}" for n, v, lic, count in rows]
    lines += ["", f"Distributions without a license file in the installed package ({len(missing)}):"]
    lines += [f"  {m}" for m in missing] or ["  none"]
    lines += ["", "No license file in the installed package does not mean it is unlicensed: see its project page."]
    lines += ["", "Python runtime license texts copied from the interpreter (runtime/):"]
    lines += [f"  {label}: {'copied' if ok else 'NOT FOUND in this interpreter'}" for label, ok in runtime.items()]
    lines += ["", f"Bundled binaries owned by no Python distribution ({len(native)}), mostly the Python runtime:"]
    lines += [f"  {n}" for n in native] or ["  none"]
    with open(os.path.join(out_dir, "PACKAGES.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def collect_from_analysis(analysis, out_dir):
    tocs = (analysis.pure, analysis.binaries, analysis.datas)
    sources = [e[1] for toc in tocs for e in toc if len(e) >= 2 and e[1]]
    return collect(sources, out_dir, [(e[0], e[1]) for e in analysis.binaries])


def data_entries(out_dir, dest):
    """Analysis data entries that put the collected folder into the bundle under `dest`."""
    entries = []
    for root, _dirs, files in os.walk(out_dir):
        below = os.path.relpath(root, out_dir)
        target = dest if below == "." else os.path.join(dest, below)
        entries += [(os.path.join(target, name), os.path.join(root, name), "DATA") for name in sorted(files)]
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--sources-json", required=True)
    args = parser.parse_args(argv)
    with open(args.sources_json, encoding="utf-8") as f:
        toc = json.load(f)
    sources = [e[1] for kind in ("pure", "binaries", "datas") for e in toc.get(kind, ()) if len(e) >= 2 and e[1]]
    collect(sources, args.out, [(e[0], e[1]) for e in toc.get("binaries", ())])
    return 0


if __name__ == "__main__":
    sys.exit(main())
