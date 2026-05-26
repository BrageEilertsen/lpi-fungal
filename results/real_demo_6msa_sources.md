# Real Mining Demo v1 -- 6-MSA: data provenance

Every input to `scripts/real_demo_6msa.py` is a real, citable value. Nothing is simulated; no MS/MS
peaks were fabricated.

## Cluster and product
- **MIBiG accession:** `BGC0001275` (release 4.0). Producer ***Glarea lozoyensis*** (fungal,
  NCBI taxId 101852). Genomic locus: GenBank `AY941322.1`.
- **Product:** 6-methylsalicylic acid (2-hydroxy-6-methylbenzoic acid).
  - SMILES (MIBiG compound record): `CC1=C(C(=CC=C1)O)C(=O)O`
  - Formula **C8H8O3**, monoisotopic mass **152.047344116**, PubChem **CID 11279**.
- **Domain annotation:** MIBiG 4.0 lists this cluster only as biosynthesis class `"PKS"`, subclass
  `"Unknown"` -- no per-module domains. (This is exactly the gap the manuscript documents: fungal
  iterative PKS are not module-annotated in MIBiG.)

## Domain alphabet used (the grammar input)
Because MIBiG gives no domains, the alphabet is the **canonical 6-MSAS architecture**:
**KS-AT-DH-KR-ACP** -- 6-methylsalicylic acid synthase, the prototypical *partially-reducing*
fungal iterative PKS and one of the first cloned fungal PKS. It carries a ketoreductase and a
dehydratase (used selectively across cycles) and **no** thioesterase, product-template,
enoylreductase, or C-methyltransferase domain. This is a literature-curated domain set, recorded
here as the provenance of the grammar input.

## Mass observable
- Real neutral monoisotopic mass **152.047344116** (MIBiG / PubChem CID 11279), supplied as the
  **[M-H]-** ion: computed m/z **151.0401** (electron-mass-corrected). 5 ppm tolerance.

## Real spectrum -- MASS sanity reference only (NOT the MS/MS rung)
- **MassBank `MSBNK-Fac_Eng_Univ_Tokyo-JP008039`**, 6-methylsalicylic acid.
  - `AC$INSTRUMENT_TYPE: EI-B`; `MS_TYPE: MS` (electron ionization); `ION_MODE: POSITIVE`;
    ion type `[M]+*`.
  - Key real peaks (m/z / relative intensity): **152/482** ([M]+•), **134/999** (base, [M-H2O]+•),
    **106/412**, **105/337**, **77/174**, **51/111**.
  - Use: the molecular ion at m/z 152 corroborates the neutral monoisotopic mass used above. This is
    an **EI** spectrum, so per the sourcing policy it is a documented real-spectrum **mass** sanity
    reference, **not** an LC-ESI-MS/MS validation rung.

## What was and was not validated
- **Validated:** a blind run on a real fungal cluster (real domain architecture + real accurate
  mass) reconstructs the real product **uniquely and correctly** (genome 1121 -> +mass 1, VERIFIED,
  true product rank 1); a real EI spectrum corroborates the molecular mass.
- **Not validated (future work):** a real LC-ESI-MS/MS fragmentation rung. None was sourced in this
  pass, and it is not load-bearing here (the mass alone already yields a unique verified core). We do
  not fabricate peaks, so the MS/MS rung is reported as *not applied* rather than simulated.
