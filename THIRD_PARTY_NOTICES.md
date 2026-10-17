# Third-party notices

TREN's own code is licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). Its source code is at <https://github.com/bostanberkay/TREN>; the source of a packaged version is the release tag of that version. The components below are **not** covered by that license; each keeps the license given here.

**Where the texts are.** This file holds the models, data, and components that are not Python packages, and the texts that have to be reproduced by hand. For the Python packages that an app bundles, the build is set up to generate a `licenses/third_party/` folder inside the app (Windows: `TREN\_internal\licenses\third_party\`) from what the build actually bundles, not from `requirements.txt` (`packaging/collect_licenses.py`, called from both PyInstaller specs). It holds the license files that each bundled package ships; a `PACKAGES.txt` index with every package's version and declared license, the packages whose installed metadata ships no license file, and the Python, OpenSSL, and Tcl/Tk versions of the build; and the Python and Tcl/Tk license texts (`runtime/`).

**Status.** No package has been built with this setup yet. It has been tried only on saved analysis data of a macOS build and in an analysis-only PyInstaller run on macOS, not on Windows, and where the folder ends up inside `TREN.app` has not been checked. The published v1.4.1 Windows package contains neither this file nor that folder (checked); the v1.4.1 macOS disk image was not opened.

## fastText language-identification model `lid.176.ftz`: CC BY-SA 3.0

- **File:** `src/tren/resources/lid.176.ftz`, the compressed language-identification model released by fastText (Facebook AI Research). 938,013 bytes; SHA-256 `8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83`; MD5 `340156704bb8c8e50c4abf35a7ec2569`.
- **Source:** <https://fasttext.cc/docs/en/language-identification.html>; file URL <https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz>. TREN ships the file unmodified: it is loaded as data by the fastText library and is not retrained, changed, or relicensed by TREN.
- **License:** Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0), <https://creativecommons.org/licenses/by-sa/3.0/> (legal code: <https://creativecommons.org/licenses/by-sa/3.0/legalcode>). The fastText authors state that the models are distributed under this license and were trained on data from Wikipedia, Tatoeba, and SETimes, used under CC-BY-SA.
- **Separate from TREN's license:** this file is **not** covered by TREN's GPL-3.0 license and remains under CC BY-SA 3.0, and TREN's own code is not licensed under CC BY-SA. If you redistribute the model, CC BY-SA 3.0 applies to it (attribution, a copy of or link to the license, and ShareAlike for adaptations).
- **Attribution requested by the authors** (cite these papers when using the models):
  1. A. Joulin, E. Grave, P. Bojanowski, T. Mikolov. *Bag of Tricks for Efficient Text Classification.* arXiv:1607.01759, 2016. (Published version: *Proceedings of EACL 2017, Short Papers*, pp. 427–431, <https://doi.org/10.18653/v1/E17-2068>.)
  2. A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov. *FastText.zip: Compressing text classification models.* arXiv:1612.03651, 2016.

## Word lists `frequent_en_words.txt` and `frequent_tr_words.txt`: CC BY-SA 4.0

- **Files:** `src/tren/resources/frequent_en_words.txt` and `src/tren/resources/frequent_tr_words.txt`.
- **Source:** this repository does not record where these files came from. Comparing their contents (2026-10-14) shows that they are the 50,000-word lists `content/2018/en/en_50k.txt` and `content/2018/tr/tr_50k.txt` of FrequencyWords by Hermit Dave, <https://github.com/hermitdave/FrequencyWords>: the English file is identical (same lines, same order), and the Turkish file is identical except for the three lines named under Changes. This attribution rests on that comparison, not on a record kept by TREN. FrequencyWords states that its frequencies were generated from OpenSubtitles2018 (OPUS, <https://opus.nlpl.eu/legacy/OpenSubtitles2018.php>); the OPUS project asks users of the corpus to cite P. Lison and J. Tiedemann, *OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles*, Proceedings of LREC 2016.
- **License:** FrequencyWords states "CC-by-sa-4.0 for content" (its code is MIT-licensed): Creative Commons Attribution-ShareAlike 4.0 International, <https://creativecommons.org/licenses/by-sa/4.0/>. This relies on that statement; the terms of the subtitle data that FrequencyWords was generated from were not examined.
- **Changes:** the English list is unchanged. Three English function words (`a 220467`, `i 182323`, `the 33111`) were removed from the Turkish list in TREN 1.3.0 (see CHANGELOG), which is the only difference from its source file. As CC BY-SA 4.0 requires of adaptations, the Turkish list is distributed under the same license.
- **Separate from TREN's license:** these files are **not** covered by TREN's GPL-3.0 license and remain under CC BY-SA 4.0.

## Reranker model files

`src/tren/resources/models/model.joblib`, `vectorizer.joblib`, and `metadata.json` are the trained MIXED-token reranker of TREN. `metadata.json` (created 2026-07-29) records a scikit-learn 1.9.0 logistic regression on character 2- to 5-grams and structured features, fitted on 141 of 201 annotated blocks (30 more were used to choose the threshold and 30 held out for testing). The input files are identified there only by SHA-256 hashes, so the training data are not named in the repository.

No license is stated for these files separately. They are part of this repository, whose `LICENSE` file is the GPL-3.0; the license terms of the data they were trained on are not recorded here.

## Stanza NER models (downloaded, not bundled)

Stanza's Turkish NER models (about 200 MB) are not part of the packages. On the first run with NER on, Stanza downloads them (TREN asks first) from <https://huggingface.co/stanfordnlp/stanza-tr>, whose model card states the license `apache-2.0`. Stanza describes the corpora its NER models were trained on at <https://stanfordnlp.github.io/stanza/ner_models.html>.

## fastText library: MIT License

- **Component:** the fastText Python module (`fasttext` 0.9.3, <https://github.com/facebookresearch/fastText>), which contains the fastText C++ library. On macOS and Linux TREN uses the PyPI release unchanged. The Windows package builds the same PyPI source with two `ssize_t` to `py::ssize_t` edits in the binding file `fasttext_pybind.cc` so that it compiles with MSVC (`packaging/build_fasttext_windows.py`); the fastText C++ library sources are unchanged.
- **License:** MIT License, as published in the fastText repository (`LICENSE`) and on PyPI:

```
MIT License

Copyright (c) 2016-present, Facebook, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## pybind11: BSD 3-Clause License

pybind11 (<https://github.com/pybind/pybind11>) is a header-only library that is compiled into the fastText Python module. License text from its repository (`LICENSE`):

```
Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Python runtime and native libraries

The packages are built with PyInstaller around a Python interpreter, which brings libraries of its own; the versions of a given build are in `licenses/third_party/PACKAGES.txt`.

- **Python** (Python Software Foundation License): the license text of the build's interpreter is copied to `licenses/third_party/runtime/python-LICENSE.txt`. The software incorporated in Python is described at <https://docs.python.org/3/license.html>.
- **Tcl/Tk** (Tcl/Tk License, BSD-style): `license.terms` of the build's Tcl/Tk is copied to `licenses/third_party/runtime/tcl-tk-license.terms` (the Windows package also contains it as `_tk_data\license.terms`). Official text: <https://www.tcl-lang.org/software/tcltk/license.html>.
- **OpenSSL:** Python's `ssl` and `hashlib` modules use OpenSSL (`libssl`, `libcrypto`). The macOS app built with the python.org installer of Python 3.11.0 contains OpenSSL 1.1.1q, which is under the dual OpenSSL and SSLeay license (text below). The Windows package contains OpenSSL 3.x, which is under the Apache License 2.0 (the same text as the Apache-2.0 license files of other bundled packages, for example Stanza's). Official statement: <https://www.openssl.org/source/license.html>.
- **ncurses** (macOS app): `libncursesw.5.dylib` from the python.org framework, which reports itself as ncurses 5.9.20120616, under an MIT-style license (text below). Official page: <https://invisible-island.net/ncurses/ncurses-license.html>.
- **Microsoft Visual C++ and Universal C runtime** (Windows package): `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`, `vcomp140.dll`, `ucrtbase.dll`, the `api-ms-win-*.dll` files, and related DLLs are Microsoft runtime files. Microsoft states that distributing the Visual C++ runtime files is limited to licensed Visual Studio users and subject to the Microsoft Software License Terms: <https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files>. Those terms are not reproduced here, and it has not been checked which terms cover the copies in this package or where the build obtained them.

## certifi and tqdm: Mozilla Public License 2.0

`certifi` (including its CA bundle `cacert.pem`) is licensed under the Mozilla Public License 2.0, and `tqdm` under MPL-2.0 AND MIT. TREN does not modify them. Their source code is at <https://github.com/certifi/python-certifi> and <https://github.com/tqdm/tqdm>; the build copies their license texts to `licenses/third_party/`. The MPL-2.0 text: <https://www.mozilla.org/MPL/2.0/>.

## PyInstaller

The executable starts from PyInstaller's bootloader, which is licensed under the GNU General Public License v2 or later with a Bootloader Exception that allows linking or embedding the compiled bootloader into other programs and distributing the result without restriction from those files; PyInstaller's run-time hooks and additional run-time modules are licensed under the Apache License 2.0. The build copies PyInstaller's license text (`COPYING.txt`) to `licenses/third_party/pyinstaller-<version>/`; the original is at <https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt>.

## Stanza demo files (Windows package)

The Windows package contains seven files from Stanza's demo folder (`stanza/pipeline/demo/`), which TREN does not use (nothing in this repository refers to them); the macOS app does not contain them. Four of them (`stanza-brat.css`, `stanza-brat.js`, `stanza-parseviewer.js`, `loading.gif`) are in the Stanza repository, which is licensed under Apache-2.0; the demo folder states no separate license for them, the script and stylesheet headers name no third-party code, and the origin of the image was not checked. Three are fonts with their own licenses:

- `Astloch-Bold.ttf`: Copyright (c) 2010, 2011 Daniel Rhatigan (sparky@ultrasparky.org) with Reserved Font Name Astloch, SIL Open Font License 1.1 (the project's `OFL.txt` is at <https://github.com/google/fonts/blob/main/ofl/astloch/OFL.txt>).
- `PT_Sans-Caption-Web-Regular.ttf`: Copyright (c) 2010, ParaType Ltd. (<http://www.paratype.com/public>), with Reserved Font Names "PT Sans", "PT Serif" and "ParaType", SIL Open Font License 1.1 (the license text is embedded in the font file).
- `Liberation_Sans-Regular.ttf` (version 1.05): digitized data (c) 2007 Ascender Corporation; Liberation is a trademark of Red Hat, Inc. The font file states only that its use is subject to the license agreement under which the font software was accepted; it does not state the license. The Liberation Fonts project states on its wiki (<https://github.com/liberationfonts/liberation-fonts/wiki>) that its Version 1 releases are licensed under GPLv2 with a font embedding exception and its current Version 2 releases under the SIL Open Font License 1.1. The license text that accompanied this particular file was not found, and the GPLv2 and the exception are not reproduced in this package.

## License texts reproduced here

### OpenSSL 1.1.1 license (macOS app)

From the OpenSSL 1.1.1q source, <https://github.com/openssl/openssl/blob/OpenSSL_1_1_1q/LICENSE>:

```

  LICENSE ISSUES
  ==============

  The OpenSSL toolkit stays under a double license, i.e. both the conditions of
  the OpenSSL License and the original SSLeay license apply to the toolkit.
  See below for the actual license texts.

  OpenSSL License
  ---------------

/* ====================================================================
 * Copyright (c) 1998-2019 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com).
 *
 */

 Original SSLeay License
 -----------------------

/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 *
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 *
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 *
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.]
 */
```

### SIL Open Font License 1.1

The license text published with the Astloch font; the copyright notices of the fonts are listed above.

```
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded, 
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```

### ncurses license (macOS app)

The ncurses 5.9 source release has no separate license file; every source file carries the notice below with its own copyright years (for example "1998-2010,2011" in `include/curses.h.in`, "1998-2004,2006" in `README`, checked in <https://ftp.gnu.org/gnu/ncurses/ncurses-5.9.tar.gz>). The maintainer publishes the same terms at <https://invisible-island.net/ncurses/ncurses-license.html>. The bundled library reports itself as ncurses 5.9.20120616, a patched 5.9; the years of the files that patch changed were not checked.

```
Copyright (c) 1998-2010,2011 Free Software Foundation, Inc.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, distribute with modifications, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE ABOVE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name(s) of the above copyright
holders shall not be used in advertising or otherwise to promote the
sale, use or other dealings in this Software without prior written
authorization.
```
