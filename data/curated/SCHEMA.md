# Curated validation entry schema

Each `*.yaml` file is one validation system. Fields:

| field | meaning |
|---|---|
| `name` | compound name |
| `organism` | producing fungus |
| `bgc` | MIBiG id / synthase name |
| `subclass` | HR / NR / PR |
| `citation` | literature source for the program (the trace), not just the structure |
| `program` | the explicit biosynthetic program: `starter`, ordered `cycles`, `release` |
| `program.cycles[]` | each: `reduction` (keto/kr/dh/er), `c_methyl` (bool), `extender` (malonyl/methylmalonyl) |
| `program.release` | hydrolysis / lactonization / aldol_aromatic / none |
| `scored_level` | `linear` (released chain, pre-cyclization) or `cyclized` (final metabolite) |
| `expected_linear_smiles` | released linear chain (carboxylic acid) — the checkpoint |
| `expected_final_smiles` | final isolated metabolite |
| `tier` | `gate` (counts toward the 80% go/no-go) / `sanity` / `bonus` |
| `ground_truth` | provenance of the expected structure: |
| | • `independent` — taken from PubChem/literature for the named compound (non-circular test) |
| | • `self_consistent` — hand-derived from biosynthetic reasoning (documents the program; weaker) |
| | • `pending` — blocked on an authoritative structure / operator not yet available |
| `program_confidence` | high / medium / low |
| `needs_expert_review` | true if the program assignment needs Brage's confirmation (also listed in QUESTIONS.md) |
| `notes` | free text |

## Scoring

The round-trip runner executes `program`, then compares the executor's output at
`scored_level` (canonical SMILES OR InChI match) against `expected_*_smiles`.
A `pending` entry with no usable expected structure is reported as **PENDING**, not
PASS — it is not silently counted as a success.
