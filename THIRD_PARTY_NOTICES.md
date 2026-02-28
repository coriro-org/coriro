# Third-Party Notices

This project depends on several third-party open-source libraries.

This notice is provided as a courtesy and attribution summary for direct dependencies
and optional dependencies referenced by this package. It is not a complete transitive
dependency inventory.

Last reviewed: 2026-02-25

## Core dependencies

### NumPy (`numpy`)
- Use in this project: core runtime dependency
- Upstream: https://pypi.org/project/numpy/
- Source: https://github.com/numpy/numpy
- Author / maintainer (PyPI metadata): Travis E. Oliphant et al.; NumPy Developers
- License (PyPI metadata): `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`

### Pillow (`Pillow`)
- Use in this project: core runtime dependency (image loading/conversion)
- Upstream: https://pypi.org/project/pillow/
- Source: https://github.com/python-pillow/Pillow
- Author (PyPI metadata): Jeffrey A. Clark
- License (PyPI metadata): `MIT-CMU`

## Optional Python dependencies (extras)

### pytesseract (`pytesseract`) — extra: `text`
- Upstream: https://pypi.org/project/pytesseract/
- Source: https://github.com/madmaze/pytesseract
- Author / maintainer (PyPI metadata): Samuel Hoffstaetter; Matthias Lee
- License (PyPI metadata): Apache License 2.0

### SciPy (`scipy`) — extra: `accents`
- Upstream: https://pypi.org/project/scipy/
- Source: https://github.com/scipy/scipy
- Maintainer (PyPI metadata): SciPy Developers
- License (PyPI metadata): BSD License (SciPy Developers / historical Enthought notice)

### PyTorch (`torch`) — extra: `cnn`
- Upstream: https://pypi.org/project/torch/
- Source: https://github.com/pytorch/pytorch
- Author (PyPI metadata): PyTorch Team
- License (PyPI metadata): BSD-3-Clause
- Note: PyTorch distributions may bundle or depend on additional third-party components;
  see upstream project notices for full details.

### PyTorch Image Models (`timm`) — extra: `cnn`
- Upstream: https://pypi.org/project/timm/
- Source: https://github.com/rwightman/pytorch-image-models
- Author (PyPI metadata): Ross Wightman
- License (PyPI metadata): Apache-2.0

## Optional system dependency

### Tesseract OCR (system binary; used with `pytesseract`)
- Upstream project: https://github.com/tesseract-ocr/tesseract
- License: Apache-2.0
- Project notes (upstream README): community-maintained in the `tesseract-ocr` GitHub organization

## Notes

- Coriro is licensed under the MIT License. See `LICENSE`.
- Please refer to each upstream project for full license text, notices, and attribution requirements.
