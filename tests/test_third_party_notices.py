"""The third-party notices that ship with TREN: the fastText library (MIT), the bundled
language-identification model (CC BY-SA 3.0, separate from TREN's GPL-3.0), pybind11
(BSD 3-Clause), the word lists (CC BY-SA 4.0), and the license texts that must accompany
the bundled runtime; the checksums recorded for the bundled model; and that the packaging
recipes include the notices. Reads files only: no display and no network.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICES = "THIRD_PARTY_NOTICES.md"


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_notices_cover_the_fasttext_components_and_separate_the_model_license():
    text = _read(NOTICES)
    assert "MIT License" in text and "Copyright (c) 2016-present, Facebook, Inc." in text
    assert "Creative Commons Attribution-ShareAlike 3.0" in text
    assert "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" in text
    assert "**not** covered by TREN's GPL-3.0 license" in text
    assert "arXiv:1607.01759" in text and "arXiv:1612.03651" in text
    assert "Copyright (c) 2016 Wenzel Jakob" in text


def test_notices_record_the_size_and_checksums_of_the_bundled_model():
    with open(os.path.join(ROOT, "src", "tren", "resources", "lid.176.ftz"), "rb") as f:
        data = f.read()
    text = _read(NOTICES)
    assert f"{len(data):,} bytes" in text
    assert hashlib.sha256(data).hexdigest() in text
    assert hashlib.md5(data, usedforsecurity=False).hexdigest() in text


def test_readme_points_to_the_notices_and_names_the_model_license():
    readme = _read("README.md")
    assert NOTICES in readme
    assert "CC BY-SA 3.0" in readme


def test_packaging_recipes_include_the_notices():
    for recipe in ("packaging/build_windows.ps1", "packaging/verify_windows_package.ps1",
                   "packaging/build_macos.sh", "packaging/TREN.spec"):
        assert NOTICES in _read(recipe), recipe
    assert 'cp "$ROOT/LICENSE"' in _read("packaging/build_macos.sh")
    assert '"LICENSE"' in _read("packaging/TREN.spec")


def test_notices_attribute_the_word_lists_and_keep_them_out_of_the_gpl():
    text = _read(NOTICES)
    assert "FrequencyWords by Hermit Dave" in text
    assert "Creative Commons Attribution-ShareAlike 4.0" in text
    assert "`a 220467`, `i 182323`, `the 33111`" in text
    assert "Lison and J. Tiedemann" in text
    assert "https://opus.nlpl.eu/legacy/OpenSubtitles2018.php" in text


def test_word_lists_still_match_what_the_notices_say_about_them():
    english = _read("src/tren/resources/frequent_en_words.txt").splitlines()
    turkish = _read("src/tren/resources/frequent_tr_words.txt").splitlines()
    assert len(english) == 50000 and len(turkish) == 49997
    assert not {"a", "i", "the"} & {line.split()[0] for line in turkish}


def test_notices_reproduce_the_license_texts_that_must_accompany_the_bundled_runtime():
    text = _read(NOTICES)
    assert "OpenSSL License" in text and "Original SSLeay License" in text
    assert "Permission is hereby granted, free of charge" in text.split("### ncurses license")[1]
    assert "Astloch" not in text and "Liberation" not in text


def test_notices_say_where_the_generated_package_licenses_are():
    text = _read(NOTICES)
    assert "licenses/third_party/" in text and "PACKAGES.txt" in text
    assert "not from `requirements.txt`" in text


def test_windows_verification_checks_the_generated_license_index():
    assert "licenses\\third_party\\PACKAGES.txt" in _read("packaging/verify_windows_package.ps1")
