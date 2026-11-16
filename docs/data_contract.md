# Data contract

The published source uses normalized CSV and JSONL inputs. The original data
exports, user selections, item mappings, and query banks are not included.
Schema descriptions below define interfaces only; they are not sample records.
These inputs must be prepared from the actual experiment data to reproduce an
experiment.

## Shared identifiers

Use the same raw user ID string in every file. Convert any internal user IDs to
that namespace before export. Use an existing, frozen item catalog; do not
rebuild it independently for each split. Item indices are contiguous positive
integers starting at 1. Index 0 is reserved for padding.

Timestamps must be finite numbers using the same unit. Sorting is stable; keep
the source order for equal timestamps. The saved selected-user history determines
which event is the held-out event when timestamps tie.

## `catalog.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `raw_item_id` | String | Item identifier in the background interaction source |
| `item_id` | Integer | Frozen model index, contiguous from 1 |

The mapping must be one-to-one. Keep it consistent with the original catalog
and all saved query files. The loader verifies uniqueness and index continuity;
the data export must establish the mapping's correspondence to the source data.

## `interactions.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | String | Raw user identifier |
| `raw_item_id` | String | Item identifier mapped through the catalog |
| `timestamp` | Number | Interaction timestamp |

For expanded-background domains, this file must contain the full available
background interaction stream, not just the earlier selected-user subset.
For Books, supply the original background. The configuration records this choice;
the code cannot infer whether a supplied file is the full source.

The builder filters this stream to the frozen catalog, removes every selected
validation/test user, then restores those users' verified prefixes. Other users'
eligible interactions remain available. It does not deduplicate repeated purchases.

## `evaluation_histories.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `user_id` | String | Selected validation/test user in the raw namespace |
| `item_id` | Integer | Item index in the frozen catalog |
| `timestamp` | Number | Timestamp from the saved chronological sequence |

Include exactly the users present in the three query files. For each user, include
the saved history followed by the single held-out target event. The loader checks
this complete sequence against the query, then excludes its last event before
restoring the prefix to training. A user's additional raw events are not restored.

## `val.jsonl`, `test.jsonl`, `test2.jsonl`

Each nonempty line is a JSON object with these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `query_id` | String | Globally unique query identifier |
| `user_id` | String | Selected user in the raw namespace |
| `history` | List of integers | Chronological saved history, excluding the target event |
| `target` | Integer | Held-out item index |
| `candidates` | List of integers | Original ordered candidate set, including the target |

Users and query identifiers must be unique across all three files. Histories must
be nonempty. Candidates must be distinct and match the configured count. Every
item must belong to the frozen catalog. Candidate order and membership are
preserved; there is no candidate resampling in this repository.

Validation alone determines the training checkpoint. The final evaluation combines
`test` and `test2`. Test targets are inspected during preprocessing only to check
prefix alignment and held-out event exclusion; their accuracy is not used to
select epochs or configure training.

## Prepared artifacts

The `prepare` command writes a padded training sequence matrix and interaction
degree vector to `background.npz`. Its manifest records the input fingerprints,
sequence length, catalog size and background checksum. Training checks those
fingerprints before starting. These artifacts are generated locally from supplied
data and are excluded from this source repository.
