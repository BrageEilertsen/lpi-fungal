# EX3 hand-off — Tier-1 conformational pilot

Phase A scaffold + Phase B1/B2 sequence prep are done locally. This dir holds the
scripts that must run on the EX3 GPU cluster to land AF2 structures + Vina poses,
after which Phase C (policy training + LOSO eval) picks up locally.

## What's already prepared (local)

| Artifact | Path |
|---|---|
| 30-row HR+PR intermediates (24 conformational + 6 substrate-only) | `data/policy/intermediates.parquet` |
| 30 docking-prep bundles with stubbed AF2 paths | `data/policy/docking/*/cycle_*/manifest.json` |
| 6 KR+ACP didomain FASTAs (315–371 aa each) | `data/policy/sequences/*_kr_acp.fasta` |
| KR-domain index (HMM scores, alignment ranges) | `data/policy/sequences/_kr_index.parquet` |

## Architectural split — only AF2 goes to EX3

| Step | Where | Why |
|---|---|---|
| AF2 KR+ACP folds | **EX3** | only GPU-required step |
| Vina covalent docking | **local** | CPU; 24 runs total |
| Descriptor extraction | **local** | CPU; validated in Phase A |
| Phase C policy eval | **local** | tiny |

Transfer footprint: ~15 KB up (6 FASTAs + SLURM), ~1 MB down (6 PDB folds + scores).

## What to run

### 1. Upload + AF2 on EX3

```bash
# from FungalFold root, set your EX3 credentials and push:
EX3_USER=brageei EX3_HOST=login.ex3.simula.no bash scripts/policy/ex3/transfer.sh up
ssh $EX3_USER@$EX3_HOST 'cd ~/lpi-af2 && sbatch af2_run.slurm'
```

Resources are set: `--partition=a100q --gres=gpu:1 --cpus-per-task=4 --mem-per-gpu=32G --time=08:00:00`. Expected runtime: ~5–10 min per sequence (315–371 aa, 1 model, 3 recycles); total ~30–60 min.

**Adjust the `module load colabfold/1.5.5` line in `af2_run.slurm`** to whatever EX3 uses (check `module avail colabfold` or `which colabfold_batch` on a login node first).

### 2. Pull AF2 outputs + normalize

```bash
EX3_USER=brageei EX3_HOST=login.ex3.simula.no bash scripts/policy/ex3/transfer.sh down
PYTHONPATH=src .venv/bin/python scripts/policy/ex3/normalize_af2_outputs.py
```

`normalize_af2_outputs.py` picks the rank-1 relaxed PDB per synthase and writes `data/policy/structures/{slug}/apo.pdb` + `_structures_index.parquet` with pLDDT means.

### 3. Identify the ACP active-site serine (local)

```bash
PYTHONPATH=src .venv/bin/python scripts/policy/ex3/identify_acp_serine.py
```

Finds the GxDS / DSL motif in the C-terminal half of each KR+ACP didomain, picks the candidate serine with the highest pLDDT (B-factor proxy), records its residue number into every cycle's `manifest.json` under `acp_serine_resid`.

Spot-check before docking: pLDDT > 70 means the ACP region is well-folded; lower than that (the ACP linker is intrinsically flexible) means the covalent anchor may be unreliable — Phase C will flag this in the bootstrap CI.

### 4. Covalent Vina docking — 24 (synthase, cycle) bundles (local)

```bash
bash scripts/policy/ex3/vina_run.sh
```

Per cycle: prepare receptor (covalent serine), prepare ligand (sentinel `*` flagged as covalent endpoint via Meeko's `--covalent_smarts '[#0]'`), box centered on ACP-Ser:OG, exhaustiveness 16, top-5 poses. Output `data/policy/poses/{syn_slug}/cycle_{t}.pdb`.

Local dependencies (install once via brew/pip):
```bash
brew install autodock-vina open-babel
pip install meeko    # also pulls AutoDockTools deps
```

If the macOS install gives trouble (legacy AutoDockTools is the usual headache), the same script runs unchanged on EX3 — just upload the FungalFold tree there too and run from a CPU partition. Trade-off is your call; I'll prep either path if you want.

The 6 substrate-only proxy cycles (3-HB, hexanoic ×2, octanoic ×3) are skipped automatically — their manifests have null `protein_pdb_path`.

### 5. Batch descriptor extraction (local)

```bash
PYTHONPATH=src .venv/bin/python scripts/policy/ex3/run_descriptors_batch.py
```

Walks all 30 manifests, runs `scripts/policy/descriptors.extract` over the docked complex, writes `data/policy/descriptors.parquet`. The 6 substrate-only rows carry NaN features + `arm="substrate_only"` so Phase C joins cleanly.

## What comes back to local

```
data/policy/structures/{slug}/apo.pdb           # 6 KR+ACP folds
data/policy/structures/_structures_index.parquet
data/policy/structures/_acp_serine.parquet
data/policy/poses/{slug}/cycle_{t}.pdb          # 24 docked complexes
data/policy/descriptors.parquet                 # 30 rows (24 conformational + 6 substrate-only)
```

Phase C harness then joins descriptors + intermediates and runs LOSO over the 9 synthases (4-way reduction state), bootstrapping over folds.

## Honest scope reminders

- **KR+ACP didomain ≠ full synthase.** AF2 will give a plausible local fold but the inter-domain context (KS, AT, DH) is missing. Justification: Brage's decision 2 (KR-only scope); per-cycle reduction signal lives at the KR pocket.
- **6-OH-mellein assignment is provisional.** AUW31183.1 (the small "PKS-like" protein) carries the KR domain; AUW31184.1 (the larger "type I PKS" in BGC0001489) does not. Treating AUW31183 as the 6-OH-mellein synthase pending paper-level confirmation.
- **3 proxy synthases (3-HB, hexanoic, octanoic) have no real protein.** They stay in the substrate-only arm. Conformational scope = 24 / 30 HR+PR cycles.
- **Single-pose Vina, not ensemble.** Static covalent dock; the dynamic-conformation hypothesis (Tier 3 MD) is the next escalation if Tier 1 is null.
