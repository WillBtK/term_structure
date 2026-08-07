# License status of the vendored AFNS engine — ACTION REQUIRED

The `arbfree_dyn_ns/` package in this directory is **vendored** from:

- Repo: https://github.com/snejens/arbfree_dynamic_nelson_siegel
- Commit: `2a402a84247648eb756df9253c281f6c18d0ab52` (2024-05-19)
- Author: the repository owner (single-author thesis project)

## The problem

**The upstream repository contains no license file.** Under default copyright
law that means *all rights reserved*: we have no explicit grant to copy,
modify, redistribute, or use this code. We have vendored it here to build on it
per the project decision, but this is a **known legal risk** that must be
resolved before any external distribution or publication of this repository.

## Do not, until a license is granted

- Do **not** publish this repo publicly or ship it to third parties.
- Do **not** relicense the vendored code.

## Required follow-up

Open an issue on the upstream repo (or email the author) requesting that they
add an explicit open-source license (e.g. MIT / BSD-3-Clause / Apache-2.0).
Track the response here:

- [ ] License request sent (date, link): ____________________
- [ ] License granted / SPDX id: ____________________

## What we changed vs upstream

Everything under `arbfree_dyn_ns/` is byte-for-byte upstream **except**
`config.py`, which we adapted so the package is importable off the author's
machine (it hardcoded an absolute path and crashed at import). The original is
preserved verbatim as `config_upstream.py.orig`. See the header comment in
`config.py` for the exact diff rationale. No model math was altered.

## The thesis PDF is intentionally not vendored

Upstream includes `Arbitrage-free_Dynamic_Nelson_Siegel_final.pdf` (the
author's ~1 MB thesis). We deliberately did **not** copy it here — it is the
most clearly copyrightable artifact and unnecessary for running the code.
Retrieve it from the upstream repo if you need the methodology write-up.
