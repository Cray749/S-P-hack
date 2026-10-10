"""Tests run in 'light' mode: hashing embedder + lexicon sentiment, so no model downloads and sub-second start."""
import os

os.environ["TREMOR_LIGHT"] = "1"
