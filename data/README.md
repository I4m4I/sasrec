# External experiment inputs

No experiment data are included in this repository. This directory contains
documentation only. Training, evaluation, background preparation and checkpoint
loading require separately supplied inputs and artifacts.

For a domain selected in the configuration, supply these files under
`data/<domain>/`:

- `catalog.csv`
- `interactions.csv`
- `evaluation_histories.csv`
- `val.jsonl`
- `test.jsonl`
- `test2.jsonl`

See [the data contract](../docs/data_contract.md) for the schemas and invariants.
The loaders contain no dataset URLs, credentials, automatic download mechanism,
or fallback experiment. `.gitignore` excludes these input files and generated
artifacts from version control.
