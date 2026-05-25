"""Deterministic chemical executor (the verifier core)."""

from rdkit import RDLogger

# Operators intentionally delete atoms (the OH lost in DH, the S sentinel lost on
# release); RDKit logs these as warnings. Silence to keep executor output clean.
RDLogger.DisableLog("rdApp.*")
