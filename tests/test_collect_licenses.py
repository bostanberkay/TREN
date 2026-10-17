"""packaging/collect_licenses.py: maps the files a PyInstaller build bundles to the installed
distributions that own them and copies the license files their metadata ships. Uses fake
distributions in a temporary folder on sys.path; no network, display, or build. Both specs
must call it after their analysis.
"""
import email
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "packaging"))

import collect_licenses as cl  # noqa: E402


def make_dist(site, name, version, extra_metadata="", license_files=None):
    """A package folder and its <name>-<version>.dist-info (METADATA, RECORD, license files) in `site`."""
    package = name.replace("-", "_")
    info = f"{package}-{version}.dist-info"
    files = {f"{package}/__init__.py": "", f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{extra_metadata}"}
    for rel, text in (license_files or {}).items():
        files[f"{info}/{rel}"] = text
    files[f"{info}/entry_points.txt"] = "[console_scripts]\n"
    for rel, text in files.items():
        path = os.path.join(site, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(site, info, "RECORD"), "w", encoding="utf-8") as f:
        f.write("".join(f"{rel},,\n" for rel in files) + f"{info}/RECORD,,\n")
    return os.path.join(site, package, "__init__.py")


def test_copies_license_files_of_the_owner_and_flags_a_distribution_without_one(tmp_path, monkeypatch):
    site = str(tmp_path / "site")
    licensed = make_dist(site, "licensed-pkg", "1.0", "License-Expression: MIT\n",
                         {"licenses/LICENSE.txt": "MIT text", "licenses/sub/NOTICE": "notice text", "LICENSE": "root text"})
    bare = make_dist(site, "bare-pkg", "2.0", "Classifier: License :: OSI Approved :: BSD License\n")
    stray = tmp_path / "unowned.bin"
    stray.write_bytes(b"x")
    monkeypatch.syspath_prepend(site)

    out = str(tmp_path / "out")
    summary = cl.collect([licensed, bare, str(stray)], out, binaries=[
        ("libfoo.dylib", str(stray)), ("x.cpython-311-darwin.so", str(stray)), ("y.pyd", str(stray)),
        ("owned.dylib", licensed)])

    folder = os.path.join(out, "licensed-pkg-1.0")
    assert open(os.path.join(folder, "licenses", "LICENSE.txt"), encoding="utf-8").read() == "MIT text"
    assert open(os.path.join(folder, "licenses", "sub", "NOTICE"), encoding="utf-8").read() == "notice text"
    assert open(os.path.join(folder, "LICENSE"), encoding="utf-8").read() == "root text"
    assert not os.path.exists(os.path.join(folder, "entry_points.txt"))
    assert not os.path.exists(os.path.join(out, "bare-pkg-2.0"))

    rows = {name: (version, lic, count) for name, version, lic, count in summary["distributions"]}
    assert rows["licensed-pkg"] == ("1.0", "MIT", 3)
    assert rows["bare-pkg"] == ("2.0", "BSD License", 0)
    assert "bare-pkg 2.0" in summary["without_license_file"]
    assert "licensed-pkg 1.0" not in summary["without_license_file"]
    assert summary["native"] == ["libfoo.dylib"]

    index = open(os.path.join(out, "PACKAGES.txt"), encoding="utf-8").read()
    assert "licensed-pkg | 1.0 | MIT | 3" in index
    assert "bare-pkg | 2.0 | BSD License | 0" in index
    assert "bare-pkg 2.0" in index.split("without a license file in the installed package")[1]
    assert "libfoo.dylib" in index


def test_only_distributions_that_own_bundled_files_are_included(tmp_path, monkeypatch):
    site = str(tmp_path / "site")
    bundled = make_dist(site, "bundled-pkg", "1.0", "License-Expression: MIT\n", {"licenses/LICENSE": "a"})
    make_dist(site, "installed-only-pkg", "3.0", "License-Expression: MIT\n", {"licenses/LICENSE": "b"})
    monkeypatch.syspath_prepend(site)

    out = str(tmp_path / "out")
    summary = cl.collect([bundled], out)

    names = [row[0] for row in summary["distributions"]]
    assert "bundled-pkg" in names
    assert "installed-only-pkg" not in names
    assert not os.path.exists(os.path.join(out, "installed-only-pkg-3.0"))


def test_collect_replaces_the_previous_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old", encoding="utf-8")
    cl.collect([], str(out))
    assert not (out / "stale.txt").exists()
    assert (out / "PACKAGES.txt").is_file()


def test_declared_license_prefers_the_expression_then_short_text_then_classifiers():
    def dist(headers):
        return types.SimpleNamespace(metadata=email.message_from_string(headers + "\n"))

    assert cl.declared_license(dist("Name: a\nLicense-Expression: Apache-2.0 OR MIT\nLicense: ignored")) == "Apache-2.0 OR MIT"
    assert cl.declared_license(dist("Name: a\nLicense: BSD-3-Clause")) == "BSD-3-Clause"
    assert cl.declared_license(dist("Name: a\nLicense: UNKNOWN\nClassifier: License :: OSI Approved :: MIT License")) == "MIT License"
    assert cl.declared_license(dist("Name: a")) == "not declared in the metadata"


def test_collect_from_analysis_reads_pure_binaries_and_datas(tmp_path, monkeypatch):
    site = str(tmp_path / "site")
    pure = make_dist(site, "pure-pkg", "1.0", "License-Expression: MIT\n", {"licenses/LICENSE": "p"})
    data = make_dist(site, "data-pkg", "1.0", "License-Expression: MIT\n", {"licenses/LICENSE": "d"})
    binary = make_dist(site, "binary-pkg", "1.0", "License-Expression: MIT\n", {"licenses/LICENSE": "b"})
    monkeypatch.syspath_prepend(site)
    analysis = types.SimpleNamespace(
        pure=[("pure_pkg", pure, "PYMODULE")],
        binaries=[("binary_pkg/lib.dylib", binary, "BINARY"), ("libz.dylib", str(tmp_path / "libz.dylib"), "BINARY")],
        datas=[("data_pkg/x", data, "DATA")],
    )

    out = str(tmp_path / "out")
    summary = cl.collect_from_analysis(analysis, out)

    names = {row[0] for row in summary["distributions"]}
    assert {"pure-pkg", "data-pkg", "binary-pkg"} <= names
    assert summary["native"] == ["libz.dylib"]


def test_data_entries_put_the_collected_folder_under_the_destination(tmp_path):
    out = tmp_path / "out"
    (out / "pkg-1.0" / "licenses").mkdir(parents=True)
    (out / "PACKAGES.txt").write_text("index", encoding="utf-8")
    (out / "pkg-1.0" / "licenses" / "LICENSE").write_text("text", encoding="utf-8")

    dest = os.path.join("licenses", "third_party")
    entries = cl.data_entries(str(out), dest)

    assert sorted(entries) == sorted([
        (os.path.join(dest, "PACKAGES.txt"), str(out / "PACKAGES.txt"), "DATA"),
        (os.path.join(dest, "pkg-1.0", "licenses", "LICENSE"), str(out / "pkg-1.0" / "licenses" / "LICENSE"), "DATA"),
    ])


def test_both_specs_collect_licenses_after_their_analysis():
    for spec in ("packaging/TREN.spec", "packaging/TREN_windows.spec"):
        with open(os.path.join(ROOT, spec), encoding="utf-8") as f:
            text = f.read()
        call = text.index("collect_licenses.collect_from_analysis(a, LICENSES_DIR)")
        assert text.index("a = Analysis(") < call < text.index("pyz = PYZ(a.pure)"), spec
        assert "a.datas += collect_licenses.data_entries(LICENSES_DIR" in text, spec


def test_package_folder_licenses_are_used_only_when_the_metadata_has_none(tmp_path, monkeypatch):
    site = str(tmp_path / "site")
    only_in_package = make_dist(site, "pkgonly", "1.0", "License-Expression: MIT\n")
    with_metadata = make_dist(site, "withmeta", "1.0", "License-Expression: MIT\n", {"licenses/LICENSE": "meta"})
    for package in ("pkgonly", "withmeta"):
        for name, text in (("LICENSE", "pkg license"), ("ThirdPartyNotices.txt", "notices"), ("data.bin", "x")):
            with open(os.path.join(site, package, name), "w", encoding="utf-8") as f:
                f.write(text)
    for dist in ("pkgonly-1.0", "withmeta-1.0"):
        with open(os.path.join(site, f"{dist}.dist-info", "RECORD"), "a", encoding="utf-8") as f:
            package = dist.split("-")[0]
            f.write("".join(f"{package}/{name},,\n" for name in ("LICENSE", "ThirdPartyNotices.txt", "data.bin")))
    monkeypatch.syspath_prepend(site)

    out = str(tmp_path / "out")
    cl.collect([only_in_package, with_metadata], out)

    fallback = os.path.join(out, "pkgonly-1.0", "pkgonly")
    assert sorted(os.listdir(fallback)) == ["LICENSE", "ThirdPartyNotices.txt"]
    assert os.listdir(os.path.join(out, "withmeta-1.0")) == ["licenses"]
