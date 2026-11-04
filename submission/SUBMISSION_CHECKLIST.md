# Journal Submission Checklist

Target: *Journal of Forecasting* free-format original research submission.

## Files in this repository

- `output/q2_submission/anonymous_manuscript_q2.docx`: blinded main manuscript; do not add names, affiliations, acknowledgements, repository URL, or identifying metadata.
- `output/q2_submission/submission_metadata_template.docx`: author-side checklist and title-page inputs.
- `main_q2.py` and `config_q2.yaml`: fully reproducible primary study pipeline.
- `CITATION.cff`: GitHub citation metadata, intentionally author-neutral until author details are supplied.

## Required before portal upload

1. Complete author names, affiliations, corresponding-author email, and ORCID identifiers in the author-side title page.
2. Confirm author order, contributions, funding statement, and conflict-of-interest statement with every author.
3. Create a public release from the final Git tag at `https://github.com/HarshitTaneja006/QRC-Volatility-Forecasting`.
4. Deposit code plus permitted replication artifacts in Zenodo, OSF, Figshare, or equivalent; obtain a persistent DOI.
5. Replace the manuscript Data Availability Statement with the archive URL and DOI. Keep the archive link out of a blinded manuscript if double-anonymous review is required; provide it through the journal's supplementary-material workflow instead.
6. Upload the anonymous manuscript, title page, any separate figures requested by the portal, and cover letter.
7. Recheck current journal instructions on submission date, including data-sharing policy and ORCID requirements.

## Release contents

Include source code, `requirements.txt`, `config_q2.yaml`, result tables, forecast panels, selection logs, and a README reproducing commands. Do not distribute third-party market data unless license terms permit it; include retrieval code and checksums instead.

## Known author-only fields

These cannot be inferred safely and must be supplied by the authors:

- Full legal author names and order
- Institutional affiliations and country
- Corresponding author email and postal address
- ORCID identifiers
- Funding / grant numbers or confirmation of no external funding
- Conflicts of interest declaration
- Public archive DOI and permanent URL
