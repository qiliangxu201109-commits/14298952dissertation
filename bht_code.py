#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Complete Bayesian analysis script for participant-level wide activity-diary exposure metrics.

This version uses the supervisor-provided ``bht_IndividualsData.csv`` as the formal input.
The source file is already in participant-level wide format: each row represents one
analysis participant, and the 12 settings provide participation, Activities, contacts,
duration, PCM, HHcontacts, and household PCM. Within the code,
outputs, and manuscript terminology, PCM consistently means person-contact minutes;
no numerical conversion is applied.

This script **does not perform activity-row data cleaning, duplicate-activity removal,
or participant×setting aggregation**. The input stage performs only data-contract and
integrity validation, then losslessly reshapes the wide file into a 49×12
participant-setting analysis grid. Source values are not imputed, corrected, or clipped.

By default, ``--mode full --steps all`` executes the following stages in order:

1. Validate the new 85-column wide-data contract, non-negative/finite values, binary
   participation, consistency with Activities, structural zeros, 49 participants
   (20 positive / 29 negative), and the three measured all-zero positive records at
   source rows 1/3/7;
2. Perform overall Bayesian predictive comparison of four common-metric representations;
3. Evaluate 36 controlled one-setting-at-a-time metric substitutions (without searching 4^12);
4. Compare one post-review fixed, outcome-independent, equal-complexity hybrid against
   the common models;
5. Extend the analysis with household occurrence / intensity / setting-overlap features.
   The project brief defines Contacts as a non-household contact measure and HHcontacts
   as a household-contact measure; the two are stored and modeled as parallel fields and
   are never subtracted;
6. Run robustness analyses using direct standardization without log1p, stronger/weaker
   priors, and removal of completely separated settings;
7. Run the post-review continuous household-contact intensity auxiliary analysis;
8. Run an independent MAP+Laplace approximate robustness validation (participant-level
   LOPO) and the nested D module;
9. Generate manuscript figures/tables, diagnostic gates, privacy partitions, and artifact hashes.

Important semantic boundaries:
- Source rows 1, 3, and 7 are retained as measured-zero and distinct observations based on
  external project confirmation obtained before analysis. The source CSV has no participant-ID,
  so the file itself cannot independently prove that the three records belong to different people;
- The new file has no source participant-ID column. The script therefore uses the 1-based
  source-row number as the restricted-analysis ``participant_id`` for LOO alignment only;
  it is not claimed to equal any external original identity number;
- The project brief defines ``Contacts`` as a non-household contact measure and ``HHcontacts``
  as a household-contact measure. Because the current wide CSV contains no activity-level coding
  dictionary, the script does not infer the exact coding mechanism. The two fields are used in
  parallel and ``Contacts - HHcontacts`` is never computed;
- For multi-activity cells, the sum of products generally differs from the product of marginal 
  sums, so PCM is not universally required to equal aggregate Contacts × Duration. The data 
  contract audits only single-activity cells (``Activities == 1``):
  all 83/83 such cells in the frozen data satisfy PCM = Contacts × Duration(minutes). Among the
  67 cells with ``Activities > 1``, only 7 satisfy the aggregate marginal product.

The frozen manuscript values correspond to the 3.0.0 environment (Python 3.14.0) below.
For a full rerun that matches the frozen manifest as closely as possible, install these
pinned versions in an isolated environment:

    python -m pip install "pymc==6.2.0" "pytensor==3.2.4" "arviz==1.2.0" \
      "arviz-base==1.2.0" "arviz-plots==1.2.0" "arviz-stats==1.2.0" \
      "xarray==2026.7.0" "xarray-einstats==0.11.0" "numba==0.66.0" \
      "numpy==2.3.5" "pandas==3.0.1" "scipy==1.18.0" \
      "scikit-learn==1.9.0" "matplotlib==3.11.1" "seaborn==0.13.2"

Formal run example:

    python "C:\\Users\\86180\\Desktop\\bht_code.py" `
      --data "C:\\Users\\86180\\Desktop\\bht_IndividualsData.csv" `
      --output "C:\\Users\\86180\\Desktop\\output3" `
      --strict-known-data

``quick`` is only for program/pipeline checks
Use ``--mode full`` for formal reporting. The output directory must be empty.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import sys
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_VERSION = "3.0.3"
__version__ = SCRIPT_VERSION

# Version 3.0.3 is a nomenclature-only revision of the frozen 3.0.2 analysis:
# the supplied person-contact-minute values are reported as person-contact minutes (PCM).
# No exposure value, transformation, prior, likelihood, seed, or validation rule is changed.


def _dependency_error(error: ImportError) -> SystemExit:
    return SystemExit(
        "Missing scientific-computing dependencies. To reproduce the frozen manuscript environment, install the pinned versions shown at the top of this file and retry."
        f"\nOriginal error: {error}"
    )


try:
    import numpy as np
    import pandas as pd
    import scipy
    import xarray as xr
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
except ImportError as _error:
    raise _dependency_error(_error) from _error

# =============================================================================
# 1. Runtime cache and thread settings
# =============================================================================

def _set_flag(current: str, flag: str, key: str) -> str:
    parts = [item.strip() for item in current.split(",") if item.strip()]
    parts = [item for item in parts if item.split("=", 1)[0] != key]
    parts.append(flag)
    return ",".join(parts)


def configure_runtime(cache_root: str | Path, blas_threads: int = 1) -> Path:
    """Configure writable caches before importing matplotlib, PyTensor, or PyMC."""
    root = Path(cache_root).resolve()
    for child in ("matplotlib", "numba", "pytensor"):
        (root / child).mkdir(parents=True, exist_ok=True)

    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
    os.environ["NUMBA_CACHE_DIR"] = str(root / "numba")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(blas_threads))

    flags = os.environ.get("PYTENSOR_FLAGS", "")
    # PyTensor's environment parser can misread Windows backslashes and create a
    # flattened relative directory. Forward slashes preserve the absolute drive path.
    pytensor_cache = (root / "pytensor").as_posix()
    flags = _set_flag(flags, f"base_compiledir={pytensor_cache}", "base_compiledir")
    if shutil.which("g++") is None:
        flags = _set_flag(flags, "cxx=", "cxx")
    os.environ["PYTENSOR_FLAGS"] = flags
    return root

# =============================================================================
# 2. Built-in configuration
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    mode: str
    draws: int
    tune: int
    chains: int
    cores: int
    target_accept: float
    sensitivity_draws: int
    sensitivity_tune: int
    sensitivity_chains: int
    exact_reloo: bool
    exact_reloo_sensitivity: bool
    bootstrap_draws: int
    seed: int
    progressbar: bool = False
    save_idata: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def derived_seed(self, label: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{label}".encode("utf-8")).digest()
        return 1 + int.from_bytes(digest[:4], "big") % 2_147_483_000


DEFAULTS = {
    "quick": RunConfig(
        mode="quick",
        draws=500,
        tune=500,
        chains=2,
        cores=int(1),
        target_accept=0.95,
        sensitivity_draws=300,
        sensitivity_tune=300,
        sensitivity_chains=2,
        exact_reloo=False,
        exact_reloo_sensitivity=False,
        bootstrap_draws=2_000,
        seed=20_260_804,
    ),
    "full": RunConfig(
        mode="full",
        draws=2_000,
        tune=2_000,
        chains=4,
        cores=int(1),
        target_accept=0.95,
        sensitivity_draws=1_000,
        sensitivity_tune=1_000,
        sensitivity_chains=4,
        exact_reloo=True,
        exact_reloo_sensitivity=True,
        bootstrap_draws=10_000,
        seed=20_260_804,
    ),
}


def load_config(
    mode: str,
    config_path: str | Path | None = None,
    **overrides: Any,
) -> RunConfig:
    if mode not in DEFAULTS:
        raise ValueError("mode must be 'quick' or 'full'")
    config = DEFAULTS[mode]
    values: dict[str, Any] = {}
    if config_path is not None:
        payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        values.update(payload.get("analysis", payload))
    configured_mode = values.pop("mode", mode)
    if configured_mode != mode:
        raise ValueError(
            f"CLI mode {mode!r} conflicts with config mode {configured_mode!r}"
        )
    values.update({key: value for key, value in overrides.items() if value is not None})
    valid = set(config.to_dict())
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {unknown}")
    return replace(config, **values)

# =============================================================================
# 3. participant-level wide-data loading and validation (no data cleaning)
# =============================================================================

WIDE_SETTINGS = (
    "Holiday",
    "Campus",
    "Exercise",
    "Hospitality",
    "Work",
    "Travel",
    "Other",
    "Research",
    "Retail",
    "Social",
    "Teaching",
    "Testing",
)

WIDE_FIELD_SUFFIXES = {
    "participation": "",
    "activity_count": "Activities",
    "contacts": "contacts",
    "duration": "duration",
    "pcm": "pcm",
    "household_contacts": "HHcontacts",
    "household_pcm": "HHpcm",
}

EXPECTED_WIDE_COLUMNS = ["Result"] + [
    setting if field == "participation" else f"{setting}{suffix}"
    for field, suffix in WIDE_FIELD_SUFFIXES.items()
    for setting in WIDE_SETTINGS
]
# The source file orders columns by field block, exactly as constructed above.
KNOWN_SHA256 = "fa93cbeff9735b5c4c5c01d7779b08cfeadcacf0cbf639ddcf2ce9baef28111c"
CONFIRMED_ZERO_EXPOSURE_ROWS = (1, 3, 7)


@dataclass
class CleanData:
    # Canonical 49×12 participant-setting grid.  This is a deterministic reshape,
    # not a cleaning or aggregation step.
    frame: pd.DataFrame
    report: dict[str, Any]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _wide_column(setting: str, field: str) -> str:
    suffix = WIDE_FIELD_SUFFIXES[field]
    return setting if field == "participation" else f"{setting}{suffix}"


def _wide_to_participant_setting(wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape the supplied wide analysis table to a balanced participant-setting grid.

    No values are imputed, corrected, deduplicated, clipped or aggregated here.
    ``participant_id`` is the 1-based source-row number because the supplied file has
    no explicit participant-ID column.
    """
    outcomes = wide["Result"].map({"Negative": 0, "Positive": 1}).to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for source_index, outcome in enumerate(outcomes, start=1):
        row = wide.iloc[source_index - 1]
        for setting in WIDE_SETTINGS:
            rows.append(
                {
                    "participant_id": source_index,
                    "source_row": source_index,
                    "outcome": int(outcome),
                    "setting_group": setting,
                    "setting": setting,
                    "participation": float(row[_wide_column(setting, "participation")]),
                    "activity_count": float(row[_wide_column(setting, "activity_count")]),
                    "contacts": float(row[_wide_column(setting, "contacts")]),
                    "duration": float(row[_wide_column(setting, "duration")]),
                    "pcm": float(row[_wide_column(setting, "pcm")]),
                    "household_contacts": float(row[_wide_column(setting, "household_contacts")]),
                    "household_pcm": float(row[_wide_column(setting, "household_pcm")]),
                }
            )
    return pd.DataFrame(rows).sort_values(["participant_id", "setting_group"]).reset_index(drop=True)


def identical_profile_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Restricted audit of identical participant profiles; profiles are never removed."""
    source = wide.copy()
    source.insert(0, "participant_id", np.arange(1, len(source) + 1))
    value_columns = [column for column in source.columns if column != "participant_id"]
    duplicated = source.duplicated(subset=value_columns, keep=False)
    if not duplicated.any():
        return pd.DataFrame(columns=["profile_group", "participant_id", "Result"])
    subset = source.loc[duplicated].copy()
    hashes = pd.util.hash_pandas_object(subset[value_columns], index=False)
    subset.insert(0, "profile_group", pd.factorize(hashes)[0] + 1)
    return subset[["profile_group", "participant_id", "Result"]]


def load_and_validate(path: str | Path, strict_known_data: bool = False) -> CleanData:
    """Validate the new participant-level dataset without performing data cleaning."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    sha = _sha256(source)
    wide = pd.read_csv(source)

    missing_columns = [name for name in EXPECTED_WIDE_COLUMNS if name not in wide.columns]
    extra_columns = [name for name in wide.columns if name not in EXPECTED_WIDE_COLUMNS]
    if missing_columns:
        raise ValueError(f"Missing required new-data columns: {missing_columns}")
    if extra_columns:
        raise ValueError(
            "Unexpected columns in the participant-level analysis file; no silent schema "
            f"expansion is allowed: {extra_columns}"
        )
    if list(wide.columns) != EXPECTED_WIDE_COLUMNS:
        raise ValueError(
            "The 85-column schema is present but column order differs from the supplied "
            "analysis contract. Refusing to reorder silently."
        )

    if wide["Result"].isna().any():
        raise ValueError("Result contains missing values")
    result_values = set(wide["Result"].astype(str).unique())
    if result_values != {"Positive", "Negative"}:
        raise ValueError(f"Unexpected Result coding: {sorted(result_values)}")

    numeric_columns = EXPECTED_WIDE_COLUMNS[1:]
    numeric = wide[numeric_columns].apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any():
        raise ValueError("Exposure fields contain missing values")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Exposure fields contain non-finite values")
    if (numeric < 0).any().any():
        bad = [column for column in numeric.columns if (numeric[column] < 0).any()]
        raise ValueError(f"Exposure fields contain negative values: {bad}")
    # From this point the numeric copy is used only to guarantee deterministic dtypes.
    # No values are altered.
    wide = wide.copy()
    wide[numeric_columns] = numeric

    indicator_mismatches: list[dict[str, Any]] = []
    noninteger_activity_cells: list[dict[str, Any]] = []
    structural_zero_violations: list[dict[str, Any]] = []
    household_exceeds_contacts: list[dict[str, Any]] = []
    for setting in WIDE_SETTINGS:
        p_col = _wide_column(setting, "participation")
        a_col = _wide_column(setting, "activity_count")
        c_col = _wide_column(setting, "contacts")
        d_col = _wide_column(setting, "duration")
        pcm_col = _wide_column(setting, "pcm")
        hh_col = _wide_column(setting, "household_contacts")
        hhpcm_col = _wide_column(setting, "household_pcm")

        if not wide[p_col].isin([0, 1]).all():
            bad_rows = (np.flatnonzero(~wide[p_col].isin([0, 1]).to_numpy()) + 1).tolist()
            raise ValueError(f"{p_col} must be binary 0/1; bad source rows={bad_rows}")

        activity = wide[a_col].to_numpy(dtype=float)
        noninteger = ~np.isclose(activity, np.round(activity), rtol=0.0, atol=1e-12)
        for row in (np.flatnonzero(noninteger) + 1):
            noninteger_activity_cells.append({"participant_id": int(row), "setting": setting})

        mismatch = wide[p_col].to_numpy(dtype=int) != (activity > 0).astype(int)
        for row in (np.flatnonzero(mismatch) + 1):
            indicator_mismatches.append({"participant_id": int(row), "setting": setting})

        zero_mask = wide[p_col].to_numpy(dtype=int) == 0
        for field_name, column in (
            ("activity_count", a_col),
            ("contacts", c_col),
            ("duration", d_col),
            ("pcm", pcm_col),
            ("household_contacts", hh_col),
            ("household_pcm", hhpcm_col),
        ):
            violating = zero_mask & (wide[column].to_numpy(dtype=float) != 0.0)
            for row in (np.flatnonzero(violating) + 1):
                structural_zero_violations.append(
                    {"participant_id": int(row), "setting": setting, "field": field_name}
                )

        
        exceeds = wide[hh_col].to_numpy(dtype=float) > wide[c_col].to_numpy(dtype=float)
        for row in (np.flatnonzero(exceeds) + 1):
            household_exceeds_contacts.append(
                {
                    "participant_id": int(row),
                    "setting": setting,
                    "contacts": float(wide.loc[row - 1, c_col]),
                    "household_contacts": float(wide.loc[row - 1, hh_col]),
                }
            )

    if noninteger_activity_cells:
        raise ValueError(f"Activities fields must be integer counts: {noninteger_activity_cells[:10]}")
    if indicator_mismatches:
        raise ValueError(
            "Participation indicator must equal 1(Activities>0): "
            f"{indicator_mismatches[:10]}"
        )
    if structural_zero_violations:
        raise ValueError(
            "Participation=0 cells must be structural zeros in the supplied wide data: "
            f"{structural_zero_violations[:10]}"
        )

    outcome = wide["Result"].map({"Negative": 0, "Positive": 1}).to_numpy(dtype=int)
    all_exposure_zero = (wide[numeric_columns].to_numpy(dtype=float) == 0.0).all(axis=1)
    all_zero_rows = tuple(int(index + 1) for index in np.flatnonzero(all_exposure_zero))
    all_zero_positive_rows = tuple(
        int(index + 1) for index in np.flatnonzero(all_exposure_zero & (outcome == 1))
    )

    expected_zero_rows = tuple(CONFIRMED_ZERO_EXPOSURE_ROWS)
    zero_row_contract = {
        "all_zero_profile_count": len(all_zero_rows),
        "all_zero_positive_count": len(all_zero_positive_rows),
        "confirmed_measured_zero_count": len(expected_zero_rows),
        "confirmed_rows_match": all_zero_positive_rows == expected_zero_rows,
        "interpretation": (
            "Three measured-zero positive records are retained under an external pre-analysis "
            "project confirmation. The supplied CSV has no participant ID, so distinct identity "
            "cannot be established from the file alone; restricted row labels are not published."
        ),
    }
    if strict_known_data and not zero_row_contract["confirmed_rows_match"]:
        raise ValueError(f"Confirmed zero-exposure row contract changed: {zero_row_contract}")

    # Internal PCM semantic audit.  The supplied wide file has no activity rows,
    # so the script does not reconstruct multi-activity PCM.  For a participant-setting
    # cell with exactly one reported activity, however, the aggregate equals the
    # activity-level value and the project person-contact-minute definition is directly auditable.
    pcm_single_activity_cells = 0
    pcm_single_activity_product_matches = 0
    pcm_multi_activity_cells = 0
    pcm_multi_activity_aggregate_product_matches = 0
    for setting in WIDE_SETTINGS:
        a = wide[_wide_column(setting, "activity_count")].to_numpy(dtype=float)
        c = wide[_wide_column(setting, "contacts")].to_numpy(dtype=float)
        d = wide[_wide_column(setting, "duration")].to_numpy(dtype=float)
        pcm = wide[_wide_column(setting, "pcm")].to_numpy(dtype=float)
        single = a == 1
        multi = a > 1
        product_match = np.isclose(pcm, c * d, rtol=0.0, atol=1e-9)
        pcm_single_activity_cells += int(single.sum())
        pcm_single_activity_product_matches += int(np.count_nonzero(single & product_match))
        pcm_multi_activity_cells += int(multi.sum())
        pcm_multi_activity_aggregate_product_matches += int(np.count_nonzero(multi & product_match))

    pcm_semantic_audit = {
        "single_activity_cells": pcm_single_activity_cells,
        "single_activity_pcm_equals_contacts_times_duration": pcm_single_activity_product_matches,
        "single_activity_match_fraction": (
            pcm_single_activity_product_matches / pcm_single_activity_cells
            if pcm_single_activity_cells else float("nan")
        ),
        "multi_activity_cells": pcm_multi_activity_cells,
        "multi_activity_pcm_equals_aggregate_contacts_times_aggregate_duration": (
            pcm_multi_activity_aggregate_product_matches
        ),
        "interpretation": (
            "Single-activity cells directly support the project person-contact-minute-product "
            "interpretation. Multi-activity participant-setting PCM is retained as supplied "
            "because an aggregate sum of activity-level products generally differs from the "
            "product of aggregate Contacts and aggregate Duration."
        ),
    }

    long = _wide_to_participant_setting(wide)
    if len(long) != len(wide) * len(WIDE_SETTINGS):
        raise AssertionError("Wide-to-grid reshape did not produce n_participants × 12 rows")
    cell_counts = long.groupby("participant_id")["setting_group"].nunique()
    if not cell_counts.eq(len(WIDE_SETTINGS)).all():
        raise AssertionError("Participant-setting grid is not balanced")

    visit = (
        long.loc[long["participation"] > 0]
        .groupby(["setting_group", "outcome"])["participant_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=list(WIDE_SETTINGS), columns=[0, 1], fill_value=0)
    )
    separation = [
        {
            "setting_group": setting,
            "negative_visitors": int(visit.loc[setting, 0]),
            "positive_visitors": int(visit.loc[setting, 1]),
            "complete_separation": bool(visit.loc[setting, 0] == 0 or visit.loc[setting, 1] == 0),
        }
        for setting in WIDE_SETTINGS
    ]

    profile_audit = identical_profile_table(wide)
    profile_group_sizes = (
        profile_audit.groupby("profile_group").size().astype(int).tolist()
        if not profile_audit.empty else []
    )
    warnings: list[str] = []
    if sha != KNOWN_SHA256:
        warnings.append("Input SHA-256 differs from the audited bht_IndividualsData.csv copy")
    if household_exceeds_contacts:
        warnings.append(
            "Some participant-setting cells have HHcontacts > Contacts. This is retained as supplied; "
            "it is compatible with the project brief treating Contacts and HHcontacts as parallel "
            "non-household and household measures, rather than a subset relationship."
        )

    totals = {
        "reported_activities": int(sum(wide[_wide_column(s, "activity_count")].sum() for s in WIDE_SETTINGS)),
        "contacts": float(sum(wide[_wide_column(s, "contacts")].sum() for s in WIDE_SETTINGS)),
        "duration": float(sum(wide[_wide_column(s, "duration")].sum() for s in WIDE_SETTINGS)),
        "pcm": float(sum(wide[_wide_column(s, "pcm")].sum() for s in WIDE_SETTINGS)),
        "household_contacts": float(sum(wide[_wide_column(s, "household_contacts")].sum() for s in WIDE_SETTINGS)),
        "household_pcm": float(sum(wide[_wide_column(s, "household_pcm")].sum() for s in WIDE_SETTINGS)),
    }

    expected = {
        "source_rows": len(wide) == 49,
        "source_columns": len(wide.columns) == 85,
        "participants": len(wide) == 49,
        "positive_participants": int((outcome == 1).sum()) == 20,
        "negative_participants": int((outcome == 0).sum()) == 29,
        "settings": len(WIDE_SETTINGS) == 12,
        "reported_activities": totals["reported_activities"] == 358,
        "confirmed_zero_rows": zero_row_contract["confirmed_rows_match"],
        "pcm_single_activity_contract": (
            pcm_single_activity_cells == 83
            and pcm_single_activity_product_matches == 83
            and pcm_multi_activity_cells == 67
            and pcm_multi_activity_aggregate_product_matches == 7
        ),
    }
    failed_expected = [name for name, passed in expected.items() if not passed]
    if failed_expected:
        message = f"Known-data expectations differ: {failed_expected}"
        if strict_known_data:
            raise ValueError(message)
        warnings.append(message)
    if strict_known_data and sha != KNOWN_SHA256:
        raise ValueError("Input SHA-256 differs under --strict-known-data")

    report = {
        "source_file": source.name,
        "source_path_redacted": True,
        "sha256": sha,
        "known_sha256_match": sha == KNOWN_SHA256,
        "source_format": "participant_level_wide",
        "source_rows": int(len(wide)),
        "source_columns": int(len(wide.columns)),
        "participant_setting_cells": int(len(long)),
        "participants": int(len(wide)),
        "positive_participants": int((outcome == 1).sum()),
        "negative_participants": int((outcome == 0).sum()),
        "settings": len(WIDE_SETTINGS),
        "setting_names": list(WIDE_SETTINGS),
        "participant_id_definition": (
            "1-based source-row number used as a restricted analysis identifier; "
            "the source file contains no participant-ID column"
        ),
        "cleaning_performed": False,
        "validation_only": True,
        "wide_to_long_operation": "deterministic reshape only; no aggregation or imputation",
        "missing_numeric_values": int(numeric.isna().sum().sum()),
        "nonfinite_numeric_values": 0,
        "negative_numeric_values": 0,
        "participation_activity_mismatch_count": len(indicator_mismatches),
        "noninteger_activity_count_cells": len(noninteger_activity_cells),
        "structural_zero_violation_count": len(structural_zero_violations),
        "all_zero_profile_count": len(all_zero_rows),
        "all_zero_positive_count": len(all_zero_positive_rows),
        "zero_exposure_contract": zero_row_contract,
        "identical_profile_group_sizes": profile_group_sizes,
        "identical_profiles_are_removed": False,
        "household_contacts_exceed_contacts_cell_count": len(household_exceeds_contacts),
        "contacts_household_semantics": (
            "project brief: Contacts=non-household contact measure and HHcontacts=household-contact measure; "
            "parallel supplied fields; no subtraction; precise activity-level encoding not available"
        ),
        "contacts_definition": (
            "Project-defined non-household contact measure; the wide CSV does not expose the finer "
            "activity-level coding mechanism, so it is not interpreted as an exact raw head count."
        ),
        "pcm_validation_rule": (
            "supplied participant-setting aggregate is retained; single-activity cells "
            "are audited against Contacts × Duration, while no universal equality is "
            "imposed on multi-activity aggregate cells"
        ),
        "pcm_semantic_audit": pcm_semantic_audit,
        "totals": totals,
        "setting_separation": separation,
        "warnings": warnings,
    }
    return CleanData(frame=long, report=report, sha256=sha)


# =============================================================================
# 4. Build four design matrices from the already aggregated participant-setting grid
# =============================================================================

METRICS = ("participation", "contacts", "duration", "pcm")


@dataclass
class Design:
    metric: str
    X: np.ndarray
    y: np.ndarray
    participant_ids: np.ndarray
    setting_names: list[str]
    household_any: np.ndarray
    household_center: float
    household_contact_intensity_z: np.ndarray
    household_contact_log1p_mean: float
    household_contact_log1p_scale: float
    transform: str
    means: np.ndarray
    scales: np.ndarray
    variant: str
    auxiliary_predictors: dict[str, np.ndarray] = field(default_factory=dict)
    auxiliary_definitions: dict[str, str] = field(default_factory=dict)

    def subset(self, keep: np.ndarray, variant: str | None = None) -> "Design":
        return replace(
            self,
            X=self.X[keep],
            y=self.y[keep],
            participant_ids=self.participant_ids[keep],
            household_any=self.household_any[keep],
            household_contact_intensity_z=self.household_contact_intensity_z[keep],
            auxiliary_predictors={
                name: np.asarray(values)[keep].copy()
                for name, values in self.auxiliary_predictors.items()
            },
            variant=variant or self.variant,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "variant": self.variant,
            "transform": self.transform,
            "participants": int(len(self.participant_ids)),
            "settings": self.setting_names,
            "household_center": self.household_center,
            "household_contact_intensity_definition": (
                "z(log1p(sum of supplied setting-specific HHcontacts per participant))"
            ),
            "household_contact_log1p_mean": self.household_contact_log1p_mean,
            "household_contact_log1p_scale": self.household_contact_log1p_scale,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "auxiliary_definitions": self.auxiliary_definitions,
        }


def _setting_visit_counts(frame: pd.DataFrame) -> pd.DataFrame:
    visits = (
        frame.loc[frame["participation"] > 0]
        .groupby(["setting_group", "outcome"])["participant_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=list(WIDE_SETTINGS), columns=[0, 1], fill_value=0)
    )
    return visits


def build_designs(
    frame: pd.DataFrame,
    *,
    continuous_transform: str = "log1p",
    drop_sparse: bool = False,
    variant: str = "main",
) -> tuple[dict[str, Design], pd.DataFrame]:
    """Build four 49×setting matrices from the already aggregated wide dataset.

    ``frame`` must be the canonical balanced participant-setting grid returned by
    :func:`load_and_validate`.  No deduplication or activity-level aggregation occurs.
    """
    if continuous_transform not in {"log1p", "raw"}:
        raise ValueError("continuous_transform must be 'log1p' or 'raw'")
    required = {
        "participant_id", "outcome", "setting_group", "participation",
        "activity_count", "contacts", "duration", "pcm",
        "household_contacts", "household_pcm",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Canonical participant-setting grid missing columns: {missing}")

    used = frame.copy()
    participants = np.asarray(sorted(used["participant_id"].unique()), dtype=int)
    outcomes = (
        used[["participant_id", "outcome"]].drop_duplicates()
        .set_index("participant_id").reindex(participants)["outcome"]
        .to_numpy(dtype=int)
    )
    settings = list(WIDE_SETTINGS)
    if drop_sparse:
        visits = _setting_visit_counts(used)
        sparse = set(visits.index[(visits[0] == 0) | (visits[1] == 0)])
        settings = [setting for setting in settings if setting not in sparse]
    if not settings:
        raise ValueError("No settings remain after sparse-setting rule")

    expected_cells = len(participants) * len(settings)
    selected = used.loc[used["setting_group"].isin(settings)].copy()
    if len(selected) != expected_cells:
        raise ValueError(
            f"Canonical participant-setting grid is incomplete: {len(selected)} != {expected_cells}"
        )

    hh_total = (
        used.groupby("participant_id")["household_contacts"].sum()
        .reindex(participants).to_numpy(dtype=float)
    )
    household_any = (hh_total > 0).astype(float)
    household_log = np.log1p(hh_total)
    household_mean = float(household_log.mean())
    household_scale = float(household_log.std(ddof=0))
    if not np.isfinite(household_scale) or household_scale <= 0:
        raise ValueError("Continuous household-contact intensity has zero or invalid variance")
    household_z = (household_log - household_mean) / household_scale

    designs: dict[str, Design] = {}
    for metric in METRICS:
        pivot = selected.pivot(index="participant_id", columns="setting_group", values=metric)
        pivot = pivot.reindex(index=participants, columns=settings)
        if pivot.isna().any().any():
            raise ValueError(f"Missing values in {metric} design")
        raw = pivot.to_numpy(dtype=float)
        if metric != "participation" and continuous_transform == "log1p":
            transformed = np.log1p(raw)
            transform = "log1p_then_zscore"
        else:
            transformed = raw.copy()
            transform = "zscore" if metric == "participation" else "raw_then_zscore"
        means = transformed.mean(axis=0)
        scales = transformed.std(axis=0, ddof=0)
        invalid = (~np.isfinite(scales)) | (scales <= 0)
        if np.any(invalid):
            bad = [settings[i] for i in np.flatnonzero(invalid)]
            raise ValueError(f"Zero/invalid variance columns for {metric}: {bad}")
        X = (transformed - means) / scales
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite standardized values in {metric}")
        designs[metric] = Design(
            metric=metric,
            X=X,
            y=outcomes.copy(),
            participant_ids=participants.copy(),
            setting_names=settings.copy(),
            household_any=household_any.copy(),
            household_center=float(household_any.mean()),
            household_contact_intensity_z=household_z.copy(),
            household_contact_log1p_mean=household_mean,
            household_contact_log1p_scale=household_scale,
            transform=transform,
            means=means,
            scales=scales,
            variant=variant,
        )

    exposure_long = selected[
        [
            "participant_id", "outcome", "setting_group", "participation",
            "activity_count", "contacts", "duration", "pcm",
            "household_contacts", "household_pcm",
        ]
    ].copy()
    return designs, exposure_long


# =============================================================================
# 4A. Setting-specific fixed outcome-independent post-review hybrid and structured household construction
# =============================================================================

PLANNED_HYBRID_DURATION_SETTINGS = ("Exercise", "Research", "Teaching")
PLANNED_HYBRID_SCIENTIFIC_RATIONALE = (
    "Fixed outcome-independent post-review exploratory representation contrast: duration is used for "
    "Exercise, Research and Teaching because these are structured activities for which "
    "time spent may represent sustained opportunity for exposure; contact count is used "
    "for the other settings. The mapping is not selected from local LOO winners and is "
    "kept separate from the nested data-adaptive D module."
)
CONTACTS_HOUSEHOLD_RELATIONSHIP = (
    "Project brief defines Contacts as a non-household contact measure and HHcontacts as a "
    "household-contact measure. The supplied wide CSV does not expose the finer activity-level "
    "encoding, so the two fields are treated as parallel recorded measures and are never subtracted."
)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _validate_design_alignment(designs: dict[str, Design], required: set[str]) -> Design:
    missing = sorted(required.difference(designs))
    if missing:
        raise ValueError(f"Missing required metric designs: {missing}")
    anchor = designs[next(iter(sorted(required)))]
    n, settings = anchor.X.shape
    if settings != len(anchor.setting_names):
        raise ValueError("Design X columns and setting_names are inconsistent")
    for metric in required:
        design = designs[metric]
        if design.X.shape != (n, settings):
            raise ValueError(f"Misaligned X shape for {metric}")
        if design.setting_names != anchor.setting_names:
            raise ValueError(f"Misaligned setting names for {metric}")
        if not np.array_equal(design.participant_ids, anchor.participant_ids):
            raise ValueError(f"Misaligned participants for {metric}")
        if not np.array_equal(design.y, anchor.y):
            raise ValueError(f"Misaligned outcomes for {metric}")
        if not np.isfinite(design.X).all():
            raise ValueError(f"Non-finite X values for {metric}")
    return anchor


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "setting"


def build_controlled_local_designs(
    designs: dict[str, Design], anchor: str = "contacts"
) -> tuple[dict[str, Design], pd.DataFrame]:
    """Equal-complexity local substitutions; outcome is not used in construction."""
    required = set(METRICS)
    _validate_design_alignment(designs, required)
    if anchor not in designs:
        raise ValueError(f"Unknown anchor metric: {anchor!r}")
    anchor_design = designs[anchor]
    output: dict[str, Design] = {}
    rows: list[dict[str, Any]] = []
    for column, setting in enumerate(anchor_design.setting_names):
        for metric in METRICS:
            source = designs[metric]
            is_anchor = metric == anchor
            key = f"controlled_local__{_slug(setting)}__{metric}"
            X = anchor_design.X.copy()
            X[:, column] = source.X[:, column]
            means = anchor_design.means.copy()
            scales = anchor_design.scales.copy()
            means[column] = source.means[column]
            scales[column] = source.scales[column]
            definitions = dict(anchor_design.auxiliary_definitions)
            definitions["controlled_local_design"] = (
                f"All settings use {anchor}; only {setting} uses {metric}. "
                "Construction does not inspect the outcome."
            )
            candidate = replace(
                anchor_design,
                metric="controlled_local",
                X=X,
                transform="columnwise_source_standardization_retained",
                means=means,
                scales=scales,
                variant=key,
                auxiliary_definitions=definitions,
            )
            output[key] = candidate
            rows.append(
                {
                    "design_key": key,
                    "setting": setting,
                    "candidate_metric": metric,
                    "anchor_metric": anchor,
                    "is_anchor": is_anchor,
                    "is_true_substitution": not is_anchor,
                    "maps_to_existing_main_model": is_anchor,
                    "existing_main_model_metric": anchor if is_anchor else "",
                    "number_of_model_columns": int(anchor_design.X.shape[1]),
                    "number_of_replaced_columns": int(not is_anchor),
                    "selection_rule": "fixed one-setting-at-a-time contrast",
                    "outcome_used_for_construction": False,
                }
            )
    return output, pd.DataFrame(rows)


def build_planned_hybrid_design(
    designs: dict[str, Design],
) -> tuple[Design, pd.DataFrame]:
    """Build the fixed outcome-independent post-review duration/contact hybrid with 12 coefficient columns."""
    _validate_design_alignment(designs, {"contacts", "duration"})
    contacts = designs["contacts"]
    duration = designs["duration"]
    missing = sorted(set(PLANNED_HYBRID_DURATION_SETTINGS) - set(contacts.setting_names))
    if missing:
        raise ValueError(f"Fixed post-review hybrid settings absent from designs: {missing}")
    X = contacts.X.copy()
    means = contacts.means.copy()
    scales = contacts.scales.copy()
    rows: list[dict[str, Any]] = []
    for column, setting in enumerate(contacts.setting_names):
        selected_metric = (
            "duration" if setting in PLANNED_HYBRID_DURATION_SETTINGS else "contacts"
        )
        source = designs[selected_metric]
        X[:, column] = source.X[:, column]
        means[column] = source.means[column]
        scales[column] = source.scales[column]
        rows.append(
            {
                "setting": setting,
                "selected_metric": selected_metric,
                "number_of_model_columns": int(contacts.X.shape[1]),
                "mapping_fixed_for_extension_stage": True,
                "mapping_preregistered": False,
                "outcome_used_for_construction": False,
                "scientific_rationale": PLANNED_HYBRID_SCIENTIFIC_RATIONALE,
            }
        )
    definitions = dict(contacts.auxiliary_definitions)
    definitions["planned_hybrid_design"] = PLANNED_HYBRID_SCIENTIFIC_RATIONALE
    hybrid = replace(
        contacts,
        metric="planned_hybrid",
        X=X,
        transform="columnwise_source_standardization_retained",
        means=means,
        scales=scales,
        variant="planned_hybrid_duration_exercise_research_teaching_contacts_elsewhere",
        auxiliary_definitions=definitions,
    )
    return hybrid, pd.DataFrame(rows)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.std(ddof=0) <= 0 or right.std(ddof=0) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def build_household_structured_predictors(
    frame: pd.DataFrame, anchor_design: Design
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Build household predictors without assuming a Contacts/HHcontacts subset relation."""
    required = {
        "participant_id", "setting_group", "household_contacts",
        "household_pcm", "participation",
    }
    _require_columns(frame, required)
    participants = list(anchor_design.participant_ids)
    settings = list(anchor_design.setting_names)
    n_participants = len(participants)
    n_settings = len(settings)

    selected = frame.loc[
        frame["participant_id"].isin(participants)
        & frame["setting_group"].isin(settings)
    ].copy()
    if len(selected) != n_participants * n_settings:
        raise ValueError("Household feature source is not a balanced participant-setting grid")

    count_matrix = (
        selected.pivot(index="participant_id", columns="setting_group", values="household_contacts")
        .reindex(index=participants, columns=settings).to_numpy(dtype=float)
    )
    time_matrix = (
        selected.pivot(index="participant_id", columns="setting_group", values="household_pcm")
        .reindex(index=participants, columns=settings).to_numpy(dtype=float)
    )
    if not np.isfinite(count_matrix).all() or not np.isfinite(time_matrix).all():
        raise ValueError("Household predictor inputs must be finite")
    if (count_matrix < 0).any() or (time_matrix < 0).any():
        raise ValueError("Household predictor inputs must be non-negative")

    household_any_matrix = (count_matrix > 0).astype(float)
    household_occurrence = (household_any_matrix.sum(axis=1) > 0).astype(float)
    H_centered = household_occurrence - household_occurrence.mean()
    household_total_count = count_matrix.sum(axis=1)
    log_total = np.log1p(household_total_count)
    occurring = household_occurrence > 0
    if np.count_nonzero(occurring) < 2:
        raise ValueError("At least two household-occurrence participants are required")
    J_mean = float(log_total[occurring].mean())
    J_scale = float(log_total[occurring].std(ddof=0))
    if not np.isfinite(J_scale) or J_scale <= 0:
        raise ValueError("Household intensity among H=1 has zero or invalid variance")
    J = np.zeros(n_participants, dtype=float)
    J[occurring] = (log_total[occurring] - J_mean) / J_scale

    modification_raw = np.sum(anchor_design.X * household_any_matrix, axis=1)
    residualization_matrix = np.column_stack(
        [np.ones(n_participants, dtype=float), H_centered, J]
    )
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        residualization_matrix, modification_raw, rcond=None
    )
    modification_residual = modification_raw - residualization_matrix @ coefficients
    modification_scale = float(modification_residual.std(ddof=0))
    if not np.isfinite(modification_scale) or modification_scale <= 0:
        raise ValueError("Residualized household modification score has zero variance")
    modification_z = (modification_residual - modification_residual.mean()) / modification_scale

    anonymous_participants = [
        f"analysis_row_{index:02d}" for index in range(1, n_participants + 1)
    ]
    long_summary = pd.DataFrame(
        {
            "analysis_participant": np.repeat(anonymous_participants, n_settings),
            "setting": np.tile(settings, n_participants),
            "household_any": household_any_matrix.astype(int).ravel(),
            "household_contact_count": count_matrix.ravel(),
            "household_pcm": time_matrix.ravel(),
        }
    )
    predictors = {
        "household_any_matrix": household_any_matrix,
        "household_contact_count_matrix": count_matrix,
        "household_pcm_matrix": time_matrix,
        "household_occurrence_centered": H_centered,
        "household_intensity_within_occurrence_z": J,
        "setting_overlap_modification_raw": modification_raw,
        "setting_overlap_modification_z": modification_z,
    }
    audit = {
        "outcome_used_for_construction": False,
        "contacts_household_relationship": CONTACTS_HOUSEHOLD_RELATIONSHIP,
        "participant_count": n_participants,
        "setting_count": n_settings,
        "matrix_shapes": {
            "household_any": list(household_any_matrix.shape),
            "household_contact_count": list(count_matrix.shape),
            "household_pcm": list(time_matrix.shape),
        },
        "nonzero_counts": {
            "household_occurrence_participants": int(np.count_nonzero(occurring)),
            "household_any_cells": int(np.count_nonzero(household_any_matrix)),
            "household_contact_count_cells": int(np.count_nonzero(count_matrix)),
            "household_pcm_cells": int(np.count_nonzero(time_matrix)),
        },
        "residualization_matrix_rank": int(rank),
        "residualization_matrix_condition_number": float(np.linalg.cond(residualization_matrix)),
        "residualization_singular_values": singular_values.tolist(),
        "pearson_correlations_after_residualization": {
            "modification_z_with_H_centered": _pearson(modification_z, H_centered),
            "modification_z_with_J": _pearson(modification_z, J),
        },
        "household_intensity_H1_log1p_mean": J_mean,
        "household_intensity_H1_log1p_scale_ddof0": J_scale,
        "modification_residual_scale_ddof0": modification_scale,
        "definitions": {
            "household_any_matrix": "1 if supplied HHcontacts > 0 for participant-setting",
            "household_contact_count_matrix": "supplied setting-specific HHcontacts",
            "household_pcm_matrix": "supplied setting-specific HHpcm",
            "household_occurrence_centered": "participant any-HHcontacts indicator minus sample mean",
            "household_intensity_within_occurrence_z": (
                "z(log1p participant total supplied HHcontacts) among H=1; set to 0 for H=0"
            ),
            "setting_overlap_modification_raw": (
                "sum over settings of anchor standardized exposure × 1(HHcontacts>0)"
            ),
            "setting_overlap_modification_z": (
                "setting-overlap score residualized on intercept, household occurrence, "
                "and household intensity, then standardized"
            ),
            "deidentification": "source-row identifiers and outcomes are excluded from public long summary",
        },
    }
    return predictors, long_summary, audit


# =============================================================================
# 5. Bayesian logistic regression and sampling diagnostics
# =============================================================================

@dataclass(frozen=True)
class FitSettings:
    draws: int
    tune: int
    chains: int
    cores: int
    target_accept: float
    progressbar: bool = False


@dataclass
class FitResult:
    name: str
    design: Design
    idata: Any
    prior: Any | None
    include_household: bool
    household_mode: str | None
    prior_scale: float
    seed: int
    settings: FitSettings
    auxiliary_prior_scales: dict[str, float] = field(default_factory=dict)


def fit_model(
    design: Design,
    *,
    name: str,
    prior_scale: float,
    settings: FitSettings,
    seed: int,
    include_household: bool = False,
    household_mode: str = "binary",
    auxiliary_prior_scales: dict[str, float] | None = None,
    sample_predictive: bool = True,
) -> FitResult:
    import pymc as pm

    auxiliary_prior_scales = dict(auxiliary_prior_scales or {})
    predictor_names = set(design.auxiliary_predictors)
    prior_names = set(auxiliary_prior_scales)
    if predictor_names != prior_names:
        raise ValueError(
            "Auxiliary predictor/prior names differ: "
            f"predictors={sorted(predictor_names)}, priors={sorted(prior_names)}"
        )

    coords = {
        "participant": design.participant_ids,
        "setting": design.setting_names,
    }
    with pm.Model(coords=coords) as model:
        x_data = pm.Data("X", design.X, dims=("participant", "setting"))
        z = pm.Normal("z", 0.0, 1.0, dims="setting")
        tau = pm.HalfNormal("tau", sigma=prior_scale)
        beta = pm.Deterministic("beta", z * tau, dims="setting")
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        eta = alpha + pm.math.dot(x_data, beta)
        if include_household:
            if household_mode == "binary":
                household_predictor = design.household_any - design.household_center
            elif household_mode == "continuous":
                household_predictor = design.household_contact_intensity_z
            else:
                raise ValueError("household_mode must be 'binary' or 'continuous'")
            household_data = pm.Data(
                "household_predictor", household_predictor, dims="participant"
            )
            gamma_household = pm.Normal("gamma_household", mu=0.0, sigma=0.75)
            eta = eta + gamma_household * household_data
        for parameter, prior_sd in auxiliary_prior_scales.items():
            values = np.asarray(design.auxiliary_predictors[parameter], dtype=float)
            if values.shape != (len(design.y),):
                raise ValueError(
                    f"Auxiliary predictor {parameter} has shape {values.shape}; "
                    f"expected {(len(design.y),)}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Auxiliary predictor {parameter} contains non-finite values")
            if not np.isfinite(prior_sd) or prior_sd <= 0:
                raise ValueError(f"Auxiliary prior scale for {parameter} must be positive")
            predictor_data = pm.Data(
                f"x_{parameter}", values, dims="participant"
            )
            coefficient = pm.Normal(parameter, mu=0.0, sigma=float(prior_sd))
            eta = eta + coefficient * predictor_data
        probability = pm.Deterministic(
            "p", pm.math.sigmoid(eta), dims="participant"
        )
        pm.Bernoulli(
            "outcome", p=probability, observed=design.y, dims="participant"
        )

        prior = None
        if sample_predictive:
            prior = pm.sample_prior_predictive(
                draws=min(1_000, max(200, settings.draws * settings.chains)),
                random_seed=seed + 1,
                var_names=["outcome"],
            )
        idata = pm.sample(
            draws=settings.draws,
            tune=settings.tune,
            chains=settings.chains,
            cores=settings.cores,
            target_accept=settings.target_accept,
            random_seed=seed,
            progressbar=settings.progressbar,
            compute_convergence_checks=settings.chains >= 2,
        )
        idata = pm.compute_log_likelihood(
            idata,
            model=model,
            progressbar=settings.progressbar,
        )
        if sample_predictive:
            idata = pm.sample_posterior_predictive(
                idata,
                model=model,
                var_names=["outcome"],
                random_seed=seed + 2,
                progressbar=settings.progressbar,
                extend_inferencedata=True,
            )

    return FitResult(
        name=name,
        design=design,
        idata=idata,
        prior=prior,
        include_household=include_household,
        household_mode=household_mode if include_household else None,
        prior_scale=prior_scale,
        seed=seed,
        settings=settings,
        auxiliary_prior_scales=auxiliary_prior_scales,
    )


def diagnostic_row(fit: FitResult) -> dict[str, Any]:
    import arviz as az

    sample_stats = fit.idata.sample_stats
    divergences = int(sample_stats["diverging"].sum().item())
    max_tree_depth = (
        int(sample_stats["tree_depth"].max().item())
        if "tree_depth" in sample_stats
        else None
    )
    reached_max_treedepth = (
        int(sample_stats["reached_max_treedepth"].sum().item())
        if "reached_max_treedepth" in sample_stats
        else 0
    )
    # ArviZ 1.2's top-level bfmi helper does not consistently accept the
    # xarray.DataTree returned by PyMC 6.2. Compute the standard E-BFMI ratio
    # directly from each chain's Hamiltonian energy instead of silently
    # dropping the diagnostic.
    if "energy" in sample_stats:
        energy = sample_stats["energy"].transpose("chain", "draw").to_numpy()
        numerator = np.square(np.diff(energy, axis=1)).mean(axis=1)
        denominator = np.var(energy, axis=1, ddof=1)
        bfmi = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0,
        )
        min_bfmi = float(np.nanmin(bfmi)) if np.isfinite(bfmi).any() else np.nan
    else:
        min_bfmi = np.nan
    auxiliary_names = list(fit.auxiliary_prior_scales)
    summary = az.summary(
        fit.idata,
        var_names=["alpha", "tau", "beta"]
        + (["gamma_household"] if fit.include_household else [])
        + auxiliary_names,
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    )
    rhat = summary["r_hat"].to_numpy(dtype=float) if "r_hat" in summary else np.array([np.nan])
    bulk = (
        summary["ess_bulk"].to_numpy(dtype=float)
        if "ess_bulk" in summary
        else np.array([np.nan])
    )
    tail = (
        summary["ess_tail"].to_numpy(dtype=float)
        if "ess_tail" in summary
        else np.array([np.nan])
    )
    return {
        "model": fit.name,
        "variant": fit.design.variant,
        "metric": fit.design.metric,
        "include_household": fit.include_household,
        "household_mode": fit.household_mode,
        "auxiliary_parameters": ";".join(auxiliary_names) or None,
        "draws": fit.settings.draws,
        "tune": fit.settings.tune,
        "chains": fit.settings.chains,
        "divergences": divergences,
        "max_tree_depth": max_tree_depth,
        "reached_max_treedepth_count": reached_max_treedepth,
        "max_rhat": float(np.nanmax(rhat)) if np.isfinite(rhat).any() else np.nan,
        "min_ess_bulk": float(np.nanmin(bulk)) if np.isfinite(bulk).any() else np.nan,
        "min_ess_tail": float(np.nanmin(tail)) if np.isfinite(tail).any() else np.nan,
        "min_bfmi": min_bfmi,
    }


def coefficient_table(fit: FitResult) -> pd.DataFrame:
    import arviz as az

    var_names = ["alpha", "tau", "beta"]
    if fit.include_household:
        var_names.append("gamma_household")
    var_names.extend(fit.auxiliary_prior_scales)
    summary = az.summary(
        fit.idata,
        var_names=var_names,
        ci_prob=0.95,
        ci_kind="hdi",
        round_to="none",
    ).reset_index(names="parameter")
    summary.insert(0, "model", fit.name)
    summary.insert(1, "variant", fit.design.variant)
    summary.insert(2, "metric", fit.design.metric)
    summary["posterior_probability_positive"] = np.nan
    posterior = fit.idata.posterior
    for row_index, parameter in summary["parameter"].items():
        if parameter.startswith("beta[") and parameter.endswith("]"):
            setting = parameter[5:-1]
            draws = posterior["beta"].sel(setting=setting).to_numpy().reshape(-1)
            summary.loc[row_index, "posterior_probability_positive"] = float(
                np.mean(draws > 0)
            )
        elif parameter == "gamma_household" and "gamma_household" in posterior:
            draws = posterior["gamma_household"].to_numpy().reshape(-1)
            summary.loc[row_index, "posterior_probability_positive"] = float(
                np.mean(draws > 0)
            )
        elif parameter in fit.auxiliary_prior_scales and parameter in posterior:
            draws = posterior[parameter].to_numpy().reshape(-1)
            summary.loc[row_index, "posterior_probability_positive"] = float(
                np.mean(draws > 0)
            )
   
    summary["exp_coefficient_posterior_mean"] = np.where(
        summary["parameter"].str.startswith(("beta", "gamma_", "psi_")),
        np.exp(summary["mean"]),
        np.nan,
    )
    if "hdi95_lb" in summary and "hdi95_ub" in summary:
        summary["exp_coefficient_hdi95_lb"] = np.where(
            summary["parameter"].str.startswith(("beta", "gamma_", "psi_")),
            np.exp(summary["hdi95_lb"]),
            np.nan,
        )
        summary["exp_coefficient_hdi95_ub"] = np.where(
            summary["parameter"].str.startswith(("beta", "gamma_", "psi_")),
            np.exp(summary["hdi95_ub"]),
            np.nan,
        )
    return summary


def predictive_count_arrays(fit: FitResult) -> tuple[np.ndarray, np.ndarray]:
    prior_counts = np.array([], dtype=float)
    posterior_counts = np.array([], dtype=float)
    if fit.prior is not None and "prior_predictive" in fit.prior.children:
        prior_counts = (
            fit.prior.prior_predictive["outcome"]
            .sum("participant")
            .to_numpy()
            .reshape(-1)
        )
    if "posterior_predictive" in fit.idata.children:
        posterior_counts = (
            fit.idata.posterior_predictive["outcome"]
            .sum("participant")
            .to_numpy()
            .reshape(-1)
        )
    return prior_counts, posterior_counts

# =============================================================================
# 6. PSIS-LOO, exact reloo, and predictive scoring
# =============================================================================

@dataclass
class Evaluation:
    model: str
    loo: Any
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    exact_refit_count: int
    reloo_diagnostics: list[dict[str, Any]]


def _loo_probabilities(fit: FitResult, loo: Any) -> np.ndarray:
    participant_ids = fit.design.participant_ids
    posterior_p = (
        fit.idata.posterior["p"]
        .sel(participant=participant_ids)
        .transpose("participant", "chain", "draw")
        .to_numpy()
    )
    log_weights = (
        loo.log_weights.sel(participant=participant_ids)
        .transpose("participant", "chain", "draw")
        .to_numpy()
    )
    weights = np.exp(log_weights)
    totals = weights.sum(axis=(1, 2), keepdims=True)
    if not np.allclose(totals, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("PSIS log weights are not normalized by participant")
    probabilities = np.sum(weights * posterior_p, axis=(1, 2))
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("LOO probabilities outside [0, 1]")
    return probabilities


def _heldout_from_refit(
    fit: FitResult,
    heldout_x: np.ndarray,
    heldout_y: int,
    heldout_household: float,
    heldout_auxiliary: dict[str, float] | None = None,
) -> tuple[float, float]:
    posterior = fit.idata.posterior
    beta = posterior["beta"].transpose("chain", "draw", "setting").to_numpy()
    alpha = posterior["alpha"].transpose("chain", "draw").to_numpy()
    eta = alpha + np.sum(beta * heldout_x[None, None, :], axis=2)
    if fit.include_household:
        gamma = posterior["gamma_household"].transpose("chain", "draw").to_numpy()
        if fit.household_mode == "continuous":
            household_predictor = heldout_household
        else:
            household_predictor = heldout_household - fit.design.household_center
        eta = eta + gamma * household_predictor
    heldout_auxiliary = heldout_auxiliary or {}
    if set(heldout_auxiliary) != set(fit.auxiliary_prior_scales):
        raise ValueError("Held-out auxiliary predictors do not match fitted parameters")
    for parameter, value in heldout_auxiliary.items():
        coefficient = posterior[parameter].transpose("chain", "draw").to_numpy()
        eta = eta + coefficient * float(value)
    probabilities = expit(eta).reshape(-1)
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    log_lik = heldout_y * np.log(clipped) + (1 - heldout_y) * np.log1p(-clipped)
    return float(logsumexp(log_lik) - math.log(len(log_lik))), float(probabilities.mean())


def _correct_loo(loo: Any, fit: FitResult, replacements: dict[int, float]) -> Any:
    from arviz_stats.utils import ELPDData

    corrected_i = loo.elpd_i.sel(participant=fit.design.participant_ids).copy(deep=True)
    pareto = loo.pareto_k.sel(participant=fit.design.participant_ids).copy(deep=True)
    for index, value in replacements.items():
        participant_id = fit.design.participant_ids[index]
        corrected_i.loc[{"participant": participant_id}] = value
        pareto.loc[{"participant": participant_id}] = 0.0

    log_lik = (
        fit.idata.log_likelihood["outcome"]
        .sel(participant=fit.design.participant_ids)
        .transpose("chain", "draw", "participant")
        .to_numpy()
    )
    samples = log_lik.shape[0] * log_lik.shape[1]
    lppd_i = logsumexp(log_lik.reshape(samples, -1), axis=0) - math.log(samples)
    values = corrected_i.to_numpy().astype(float)
    elpd = float(values.sum())
    # Match ArviZ's ELPD standard-error convention (population variance).
    se = float(np.sqrt(len(values) * np.var(values, ddof=0)))
    p_loo = float(np.sum(lppd_i - values))
    return ELPDData(
        kind="loo",
        elpd=elpd,
        se=se,
        p=p_loo,
        n_samples=loo.n_samples,
        n_data_points=loo.n_data_points,
        scale="log",
        warning=bool(np.any(pareto.to_numpy() > loo.good_k)),
        good_k=loo.good_k,
        elpd_i=corrected_i,
        pareto_k=pareto,
        approx_posterior=False,
        log_weights=loo.log_weights,
    )


def evaluate_fit(
    fit: FitResult,
    *,
    exact_reloo: bool,
    reloo_seed: int,
) -> Evaluation:
    import arviz as az

    loo = az.loo(fit.idata, pointwise=True, var_name="outcome")
    probabilities = _loo_probabilities(fit, loo)
    pareto = (
        loo.pareto_k.sel(participant=fit.design.participant_ids).to_numpy().astype(float)
    )
    high_indices = np.flatnonzero(pareto > float(loo.good_k))
    exact_flags = np.zeros(len(fit.design.y), dtype=bool)
    replacements: dict[int, float] = {}
    reloo_diagnostics: list[dict[str, Any]] = []

    if exact_reloo:
        for order, index in enumerate(high_indices):
            keep = np.ones(len(fit.design.y), dtype=bool)
            keep[index] = False
            training = fit.design.subset(keep, variant=f"{fit.design.variant}_reloo")
            anonymous_fold = order + 1
            refit = fit_model(
                training,
                name=f"{fit.name}_reloo_fold_{anonymous_fold:02d}",
                prior_scale=fit.prior_scale,
                settings=fit.settings,
                seed=reloo_seed + order,
                include_household=fit.include_household,
                household_mode=fit.household_mode or "binary",
                auxiliary_prior_scales=fit.auxiliary_prior_scales,
                sample_predictive=False,
            )
            refit_diagnostic = diagnostic_row(refit)
            refit_diagnostic.update(
                {
                    "fit_role": "exact_reloo",
                    "parent_model": fit.name,
                    "heldout_fold": anonymous_fold,
                }
            )
            reloo_diagnostics.append(refit_diagnostic)
            elpd_i, probability = _heldout_from_refit(
                refit,
                fit.design.X[index],
                int(fit.design.y[index]),
                float(
                    fit.design.household_contact_intensity_z[index]
                    if fit.household_mode == "continuous"
                    else fit.design.household_any[index]
                ),
                {
                    parameter: float(fit.design.auxiliary_predictors[parameter][index])
                    for parameter in fit.auxiliary_prior_scales
                },
            )
            replacements[index] = elpd_i
            probabilities[index] = probability
            exact_flags[index] = True
        if replacements:
            loo = _correct_loo(loo, fit, replacements)

    elpd_values = (
        loo.elpd_i.sel(participant=fit.design.participant_ids).to_numpy().astype(float)
    )
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    y = fit.design.y
    brier = float(np.mean((probabilities - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    auc = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else np.nan
    predictions = pd.DataFrame(
        {
            "model": fit.name,
            "variant": fit.design.variant,
            "metric": fit.design.metric,
            "participant_id": fit.design.participant_ids,
            "outcome": y,
            "loo_probability": probabilities,
            "elpd_i": elpd_values,
            "pareto_k": pareto,
            "pareto_good_k": float(loo.good_k),
            "exact_refit": exact_flags,
        }
    )
    metrics = {
        "model": fit.name,
        "variant": fit.design.variant,
        "metric": fit.design.metric,
        "elpd": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
        "good_k": float(loo.good_k),
        "high_k_count_initial": int(len(high_indices)),
        "exact_refit_count": int(exact_flags.sum()),
        "brier": brier,
        "log_loss": log_loss,
        "roc_auc": auc,
    }
    return Evaluation(
        model=fit.name,
        loo=loo,
        predictions=predictions,
        metrics=metrics,
        exact_refit_count=int(exact_flags.sum()),
        reloo_diagnostics=reloo_diagnostics,
    )


def compare_evaluations(evaluations: dict[str, Evaluation]) -> pd.DataFrame:
    import arviz as az

    comparison = az.compare(
        {name: evaluation.loo for name, evaluation in evaluations.items()},
        method="stacking",
        round_to="none",
    ).reset_index(names="model")
    return comparison


def bayesian_bootstrap_ranks(
    evaluations: dict[str, Evaluation], *, draws: int, seed: int
) -> pd.DataFrame:
    names = list(evaluations)
    if not names:
        raise ValueError("At least one evaluation is required")
    reference_ids = evaluations[names[0]].predictions["participant_id"].to_numpy()
    pointwise_rows: list[np.ndarray] = []
    for name in names:
        evaluation = evaluations[name]
        ids = evaluation.predictions["participant_id"].to_numpy()
        if not np.array_equal(ids, reference_ids):
            raise ValueError(f"Participant order differs for Bayesian bootstrap: {name}")
        values = (
            evaluation.loo.elpd_i
            .sel(participant=reference_ids)
            .to_numpy()
            .astype(float)
        )
        pointwise_rows.append(values)
    pointwise = np.vstack(pointwise_rows)
    n_obs = pointwise.shape[1]
    rng = np.random.default_rng(seed)
    weights = rng.exponential(1.0, size=(draws, n_obs))
    weights = weights / weights.sum(axis=1, keepdims=True) * n_obs
    totals = weights @ pointwise.T
    order = np.argsort(-totals, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(draws)[:, None], order] = np.arange(len(names))[None, :]
    return pd.DataFrame(
        {
            "model": names,
            "probability_rank_1": (ranks == 0).mean(axis=0),
            "mean_rank": ranks.mean(axis=0) + 1,
            "median_bootstrap_elpd": np.median(totals, axis=0),
            "bootstrap_elpd_q025": np.quantile(totals, 0.025, axis=0),
            "bootstrap_elpd_q975": np.quantile(totals, 0.975, axis=0),
        }
    ).sort_values("mean_rank")

# =============================================================================
# 7. Result tables and English-language figures
# =============================================================================

def ensure_output_tree(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).resolve()
    paths = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _pyplot():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return plt


def plot_data_flow(validation: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    labels = ["Participant rows", "Participant-setting cells", "Positive", "Negative"]
    values = [
        validation["source_rows"],
        validation["participant_setting_cells"],
        validation["positive_participants"],
        validation["negative_participants"],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#E45756", "#54A24B"])
    ax.set_ylabel("Count")
    ax.set_title("Validated participant-level dataset and outcome counts")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison(comparison: pd.DataFrame, output: Path) -> None:
    plt = _pyplot()
    ordered = comparison.sort_values("elpd")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        ordered["elpd"],
        ordered["model"],
        xerr=ordered["se"],
        fmt="o",
        color="#4C78A8",
        ecolor="#9EC1DF",
        capsize=3,
    )
    ax.set_xlabel("PSIS-LOO ELPD (± 1 SE)")
    ax.set_ylabel("")
    ax.set_title("Participant-level predictive comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _interval_columns(frame: pd.DataFrame) -> tuple[str, str]:
    lower = [
        column
        for column in frame
        if column.startswith(("hdi_", "eti_", "hdi95_", "eti95_"))
        and (column.endswith(("lb", "lower")) or "2.5" in column)
    ]
    upper = [
        column
        for column in frame
        if column.startswith(("hdi_", "eti_", "hdi95_", "eti95_"))
        and (column.endswith(("ub", "upper")) or "97.5" in column)
    ]
    if not lower or not upper:
        interval = [column for column in frame if column.startswith(("hdi", "eti"))]
        if len(interval) >= 2:
            return interval[0], interval[-1]
        raise ValueError("Posterior summary lacks interval columns")
    return lower[0], upper[0]


def plot_setting_forest(coefficients: pd.DataFrame, output: Path) -> None:
    plt = _pyplot()
    subset = coefficients[
        coefficients["parameter"].str.startswith("beta[")
        & coefficients["model"].str.startswith("main__")
    ].copy()
    if subset.empty:
        return
    low, high = _interval_columns(subset)
    subset["label"] = subset["model"].str.replace("main__", "", regex=False) + ": " + subset[
        "parameter"
    ].str.replace("beta[", "", regex=False).str.rstrip("]")
    subset = subset.sort_values("mean").reset_index(drop=True)
    height = max(6.0, 0.22 * len(subset))
    fig, ax = plt.subplots(figsize=(8.5, height))
    y = np.arange(len(subset))
    ax.errorbar(
        subset["mean"],
        y,
        xerr=np.vstack([subset["mean"] - subset[low], subset[high] - subset["mean"]]),
        fmt="o",
        color="#4C78A8",
        ecolor="#A9C9E2",
        capsize=2,
        markersize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, subset["label"])
    ax.set_xlabel("Posterior log-odds coefficient (95% HDI)")
    ax.set_title("Shrunk setting coefficients")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ppc(
    prior_counts: np.ndarray,
    posterior_counts: np.ndarray,
    observed_positive: int,
    output: Path,
) -> None:
    if len(posterior_counts) == 0:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    if len(prior_counts):
        axes[0].hist(prior_counts, bins=np.arange(-0.5, 47.5, 1), color="#BAB0AC")
    axes[0].axvline(observed_positive, color="#E45756", linestyle="--")
    axes[0].set_title("Prior predictive")
    axes[1].hist(posterior_counts, bins=np.arange(-0.5, 47.5, 1), color="#4C78A8")
    axes[1].axvline(observed_positive, color="#E45756", linestyle="--")
    axes[1].set_title("Posterior predictive")
    for ax in axes:
        ax.set_xlabel("Positive participants")
        ax.set_ylabel("Draws")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(results: pd.DataFrame, output: Path) -> None:
    if results.empty:
        return
    import seaborn as sns

    plt = _pyplot()
    pivot = results.pivot(index="scenario", columns="metric", values="elpd")
    fig, ax = plt.subplots(figsize=(7.6, max(3.8, 0.65 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax, cbar_kws={"label": "ELPD"})
    ax.set_xlabel("Exposure metric")
    ax.set_ylabel("Sensitivity scenario")
    ax.set_title("Predictive performance across analysis-choice perturbations")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_household(comparison: pd.DataFrame, output: Path) -> None:
    plot_main_comparison(comparison, output)


def _require_plot_columns(
    frame: pd.DataFrame, required: set[str], *, figure_name: str
) -> None:
    """Raise a concise error when a plotting table does not meet its contract."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{figure_name} requires a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{figure_name} received an empty table")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{figure_name} is missing columns: {', '.join(missing)}")


def _validated_numeric(
    frame: pd.DataFrame, column: str, *, figure_name: str, nonnegative: bool = False
) -> pd.Series:
    """Return a finite numeric series without silently discarding observations."""
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{figure_name}: {column} must contain only finite numbers")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{figure_name}: {column} must be non-negative")
    return values.astype(float)


def _save_research_figure(fig: Any, output: str | Path, plt: Any) -> None:
    """Save with the established reporting defaults and release the figure."""
    path = Path(output)
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    base = path.with_suffix("")
    if path.suffix.casefold() != ".svg":
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    if path.suffix.casefold() != ".pdf":
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _display_metric(metric: object) -> str:
    labels = {
        "participation": "Participation",
        "contacts": "Contact count",
        "duration": "Duration",
        "pcm": "Person-contact minutes",
    }
    token = str(metric)
    return labels.get(token, token.replace("_", " ").strip().title())


def plot_setting_metric_support(
    summary: pd.DataFrame,
    output: str | Path,
    *,
    value: str | None = None,
) -> None:
    """Plot setting-by-metric rank support or paired delta ELPD as a heatmap.

    ``summary`` must contain one row per ``setting`` and ``metric``. When
    ``value`` is omitted, ``rank1_probability`` is preferred, followed by the
    legacy ``probability_rank_1`` spelling and then ``delta_elpd``.
    """
    figure_name = "Setting-by-metric support heatmap"
    _require_plot_columns(summary, {"setting", "metric"}, figure_name=figure_name)
    if value is None:
        value = next(
            (
                candidate
                for candidate in ("rank1_probability", "probability_rank_1", "delta_elpd")
                if candidate in summary.columns
            ),
            None,
        )
    if value is None:
        raise ValueError(
            f"{figure_name} requires rank1_probability, probability_rank_1, or delta_elpd"
        )
    _require_plot_columns(summary, {value}, figure_name=figure_name)

    table = summary[["setting", "metric", value]].copy()
    table["setting"] = table["setting"].astype(str)
    table["metric"] = table["metric"].astype(str)
    table[value] = _validated_numeric(table, value, figure_name=figure_name)
    duplicate = table.duplicated(["setting", "metric"], keep=False)
    if duplicate.any():
        pairs = table.loc[duplicate, ["setting", "metric"]].drop_duplicates()
        raise ValueError(
            f"{figure_name} requires unique setting-metric rows; found {len(pairs)} duplicates"
        )

    setting_order = list(dict.fromkeys(table["setting"]))
    observed_metrics = list(dict.fromkeys(table["metric"]))
    canonical = ["participation", "contacts", "duration", "pcm"]
    metric_order = [item for item in canonical if item in observed_metrics]
    metric_order.extend(item for item in observed_metrics if item not in metric_order)
    pivot = table.pivot(index="setting", columns="metric", values=value).reindex(
        index=setting_order, columns=metric_order
    )
    if pivot.isna().any().any():
        missing_cells = int(pivot.isna().sum().sum())
        raise ValueError(f"{figure_name} has {missing_cells} missing setting-metric cells")

    values = pivot.to_numpy(dtype=float)
    is_probability = value in {"rank1_probability", "probability_rank_1"}
    if is_probability and ((values < 0).any() or (values > 1).any()):
        raise ValueError(f"{figure_name}: rank probabilities must lie in [0, 1]")

    plt = _pyplot()
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(7.4, max(4.2, 0.42 * len(setting_order) + 1.5)))
    if is_probability:
        image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        colorbar_label = "Bayesian-bootstrap rank-1 probability"
        annotation_format = ".2f"
    else:
        limit = max(float(np.max(np.abs(values))), 1e-9)
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        colorbar_label = "Paired delta ELPD"
        annotation_format = ".1f"
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([_display_metric(item) for item in metric_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(setting_order)))
    ax.set_yticklabels(setting_order)
    ax.set_xlabel("Exposure representation")
    ax.set_ylabel("Activity setting")
    ax.set_title("Setting-specific predictive support")
    threshold = 0.58 if is_probability else 0.58 * max(float(np.max(np.abs(values))), 1e-9)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            shown = values[row, column]
            dark_cell = shown > threshold if is_probability else abs(shown) > threshold
            ax.text(
                column,
                row,
                format(shown, annotation_format),
                ha="center",
                va="center",
                color="white" if dark_cell else "#272727",
                fontsize=7.5,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save_research_figure(fig, output, plt)


def plot_common_vs_hybrid_delta(
    comparison: pd.DataFrame, output: str | Path
) -> None:
    """Plot paired delta ELPD for common-metric and the fixed post-review hybrid models.

    Required columns are ``model``, ``delta_elpd`` and either ``dse`` or
    ``delta_elpd_se``. An optional ``reference_model`` column is incorporated
    into the displayed contrast label.
    """
    figure_name = "Common-versus-hybrid comparison"
    _require_plot_columns(
        comparison, {"model", "delta_elpd"}, figure_name=figure_name
    )
    uncertainty = next(
        (column for column in ("dse", "delta_elpd_se", "paired_dse") if column in comparison),
        None,
    )
    if uncertainty is None:
        raise ValueError(f"{figure_name} requires dse, delta_elpd_se, or paired_dse")

    table = comparison.copy()
    table["delta_elpd"] = _validated_numeric(
        table, "delta_elpd", figure_name=figure_name
    )
    table[uncertainty] = _validated_numeric(
        table, uncertainty, figure_name=figure_name, nonnegative=True
    )
    if table["model"].astype(str).duplicated().any():
        raise ValueError(f"{figure_name} requires unique model rows")
    labels = table["model"].astype(str)
    if "reference_model" in table:
        labels = labels + " vs " + table["reference_model"].astype(str)

    plt = _pyplot()
    height = max(3.5, 0.55 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(7.4, height))
    y = np.arange(len(table))[::-1]
    ax.errorbar(
        table["delta_elpd"],
        y,
        xerr=table[uncertainty],
        fmt="o",
        color="#0F4D92",
        ecolor="#9EC1DF",
        capsize=3,
        markersize=5,
    )
    ax.axvline(0.0, color="#4D4D4D", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired delta ELPD (+/- dSE)")
    ax.set_ylabel("")
    ax.set_title("Common-metric and fixed post-review hybrid prediction")
    _save_research_figure(fig, output, plt)


def plot_household_extension_comparison(
    comparison: pd.DataFrame, output: str | Path
) -> None:
    """Compare baseline, additive and modification household specifications.

    Pairwise input uses ``delta_elpd`` plus ``dse`` (or its supported aliases)
    and labels from ``contrast`` or ``model``. If pairwise columns are absent,
    absolute ``elpd`` with ``se`` is plotted instead.
    """
    figure_name = "Household extension comparison"
    _require_plot_columns(comparison, set(), figure_name=figure_name)
    table = comparison.copy()
    label_column = next(
        (column for column in ("contrast", "role", "model") if column in table), None
    )
    if label_column is None:
        raise ValueError(f"{figure_name} requires contrast, role, or model")
    if table[label_column].astype(str).duplicated().any():
        raise ValueError(f"{figure_name} requires unique comparison labels")

    delta_column = next(
        (
            column
            for column in ("delta_elpd", "delta_elpd_vs_reference", "delta_elpd_vs_baseline")
            if column in table
        ),
        None,
    )
    if delta_column is not None:
        uncertainty = next(
            (column for column in ("dse", "delta_elpd_se", "paired_dse") if column in table),
            None,
        )
        if uncertainty is None:
            raise ValueError(f"{figure_name} pairwise input requires paired uncertainty")
        x_label = "Paired delta ELPD (+/- dSE)"
        reference_line = 0.0
    else:
        _require_plot_columns(table, {"elpd", "se"}, figure_name=figure_name)
        delta_column = "elpd"
        uncertainty = "se"
        x_label = "Participant-level PSIS-LOO ELPD (+/- 1 SE)"
        reference_line = None

    table[delta_column] = _validated_numeric(
        table, delta_column, figure_name=figure_name
    )
    table[uncertainty] = _validated_numeric(
        table, uncertainty, figure_name=figure_name, nonnegative=True
    )
    labels = table[label_column].astype(str).str.replace("_", " ", regex=False)
    colors = [
        "#767676"
        if "baseline" in label.casefold()
        else "#42949E"
        if "modif" in label.casefold()
        else "#0F4D92"
        for label in labels
    ]

    plt = _pyplot()
    height = max(3.5, 0.58 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(7.4, height))
    y = np.arange(len(table))[::-1]
    for index, (estimate, error, color) in enumerate(
        zip(table[delta_column], table[uncertainty], colors)
    ):
        ax.errorbar(
            estimate,
            y[index],
            xerr=error,
            fmt="o",
            color=color,
            ecolor=color,
            alpha=0.9,
            capsize=3,
            markersize=5,
        )
    if reference_line is not None:
        ax.axvline(reference_line, color="#4D4D4D", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, labels.str.title())
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.set_title("Household additive and modification comparisons")
    _save_research_figure(fig, output, plt)


def plot_setting_outcome_descriptives(
    summary: pd.DataFrame,
    output: str | Path,
    *,
    value_column: str = "mean_exposure",
    metric: str | None = None,
    xlabel: str = "Mean standardized exposure",
) -> None:
    """Plot positive-versus-negative mean exposure for each activity setting.

    The tidy input requires ``setting``, ``outcome_group`` and ``value_column``.
    Outcome groups may be Positive/Negative or 1/0. If a ``metric`` column
    contains multiple representations, select one through the ``metric``
    argument so unlike exposure scales are never combined silently.
    """
    figure_name = "Positive-versus-negative setting descriptives"
    _require_plot_columns(
        summary, {"setting", "outcome_group", value_column}, figure_name=figure_name
    )
    table = summary.copy()
    if "metric" in table:
        available = list(dict.fromkeys(table["metric"].astype(str)))
        if metric is None and len(available) > 1:
            raise ValueError(
                f"{figure_name}: choose metric from {', '.join(available)}"
            )
        selected = str(metric) if metric is not None else available[0]
        table = table[table["metric"].astype(str) == selected].copy()
        if table.empty:
            raise ValueError(f"{figure_name}: metric {selected!r} is not present")
        metric_title = _display_metric(selected)
    elif metric is not None:
        raise ValueError(f"{figure_name}: metric was supplied but the table has no metric column")
    else:
        metric_title = "Exposure"

    def outcome_label(value: object) -> str:
        token = str(value).strip().casefold()
        if token in {"1", "1.0", "positive", "case", "cases", "infected"}:
            return "Positive"
        if token in {"0", "0.0", "negative", "control", "controls", "uninfected"}:
            return "Negative"
        raise ValueError(f"{figure_name}: unrecognised outcome group {value!r}")

    table["setting"] = table["setting"].astype(str)
    table["_outcome"] = table["outcome_group"].map(outcome_label)
    table["_value"] = _validated_numeric(
        table, value_column, figure_name=figure_name
    )
    if table.duplicated(["setting", "_outcome"], keep=False).any():
        raise ValueError(f"{figure_name} requires one row per setting and outcome group")
    pivot = table.pivot(index="setting", columns="_outcome", values="_value")
    missing_groups = {"Positive", "Negative"}.difference(pivot.columns)
    if missing_groups or pivot[["Positive", "Negative"]].isna().any().any():
        raise ValueError(f"{figure_name} requires Positive and Negative values for every setting")
    pivot["difference"] = pivot["Positive"] - pivot["Negative"]
    pivot = pivot.sort_values("difference")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.4, max(4.2, 0.42 * len(pivot) + 1.5)))
    y = np.arange(len(pivot))
    for row, (_, values) in enumerate(pivot.iterrows()):
        ax.plot(
            [values["Negative"], values["Positive"]],
            [row, row],
            color="#CFCECE",
            linewidth=1.4,
            zorder=1,
        )
    ax.scatter(
        pivot["Negative"], y, color="#767676", s=30, label="Negative", zorder=2
    )
    ax.scatter(
        pivot["Positive"], y, color="#B64342", s=30, label="Positive", zorder=3
    )
    ax.set_yticks(y, pivot.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Activity setting")
    ax.set_title(f"{metric_title} by participant outcome")
    ax.legend(loc="best")
    _save_research_figure(fig, output, plt)

# =============================================================================
# 8. RQ1--RQ4 main workflow and robustness analyses
# =============================================================================

@dataclass(frozen=True)
class Scenario:
    name: str
    continuous_transform: str = "log1p"
    prior_scale: float = 0.5
    drop_sparse: bool = False


def _versions() -> dict[str, str]:
    packages = [
        "pymc",
        "pytensor",
        "arviz",
        "arviz-base",
        "arviz-plots",
        "arviz-stats",
        "xarray",
        "xarray-einstats",
        "numba",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint() -> tuple[str, int]:
    """Hash this standalone script instead of a multi-file package."""
    source = Path(__file__).resolve()
    return _file_sha256(source), 1


def _settings(config: RunConfig, sensitivity: bool = False) -> FitSettings:
    return FitSettings(
        draws=config.sensitivity_draws if sensitivity else config.draws,
        tune=config.sensitivity_tune if sensitivity else config.tune,
        chains=config.sensitivity_chains if sensitivity else config.chains,
        cores=config.cores,
        target_accept=config.target_accept,
        progressbar=config.progressbar,
    )


def _sample_audit_table(report: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("source_rows", report["source_rows"]),
        ("source_columns", report["source_columns"]),
        ("participant_setting_cells", report["participant_setting_cells"]),
        ("participants", report["participants"]),
        ("positive_participants", report["positive_participants"]),
        ("negative_participants", report["negative_participants"]),
        ("settings", report["settings"]),
        ("reported_activities", report["totals"]["reported_activities"]),
        ("all_zero_positive_count", report["all_zero_positive_count"]),
        ("participation_activity_mismatch_count", report["participation_activity_mismatch_count"]),
        ("structural_zero_violation_count", report["structural_zero_violation_count"]),
        ("household_contacts_exceed_contacts_cell_count", report["household_contacts_exceed_contacts_cell_count"]),
        ("pcm_single_activity_cells", report["pcm_semantic_audit"]["single_activity_cells"]),
        ("pcm_single_activity_product_matches", report["pcm_semantic_audit"]["single_activity_pcm_equals_contacts_times_duration"]),
        ("pcm_multi_activity_cells", report["pcm_semantic_audit"]["multi_activity_cells"]),
        ("pcm_multi_activity_aggregate_product_matches", report["pcm_semantic_audit"]["multi_activity_pcm_equals_aggregate_contacts_times_aggregate_duration"]),
        ("cleaning_performed", report["cleaning_performed"]),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def _save_idata(fit: FitResult, destination: Path) -> None:
    try:
        import h5netcdf  # noqa: F401
    except ImportError as error:
        raise RuntimeError("--save-idata requires h5netcdf and h5py") from error
    fit.idata.to_netcdf(destination, engine="h5netcdf")


def _model_task(
    design,
    *,
    name: str,
    prior_scale: float,
    settings: FitSettings,
    seed: int,
    include_household: bool,
    exact_reloo: bool,
    auxiliary_prior_scales: dict[str, float] | None = None,
    sample_predictive: bool = True,
) -> tuple[FitResult, Evaluation, dict[str, Any], pd.DataFrame]:
    fit = fit_model(
        design,
        name=name,
        prior_scale=prior_scale,
        settings=settings,
        seed=seed,
        include_household=include_household,
        auxiliary_prior_scales=auxiliary_prior_scales,
        sample_predictive=sample_predictive,
    )
    evaluation = evaluate_fit(
        fit,
        exact_reloo=exact_reloo,
        reloo_seed=seed + 100_000,
    )
    diagnostic = diagnostic_row(fit)
    diagnostic.update(
        {"fit_role": "full_data", "parent_model": None, "heldout_fold": None}
    )
    return fit, evaluation, diagnostic, coefficient_table(fit)


def _paired_elpd_summary(
    candidate: Evaluation,
    reference: Evaluation,
    *,
    comparison: str,
) -> dict[str, Any]:
    """Return an aligned participant-level paired ELPD contrast.

    Positive values favour ``candidate``.  The uncertainty follows ArviZ's
    population-variance convention: sqrt(n * var(delta_i, ddof=0)).
    """

    candidate_frame = candidate.predictions.set_index("participant_id")
    reference_frame = reference.predictions.set_index("participant_id")
    if not candidate_frame.index.equals(reference_frame.index):
        raise ValueError(f"Participant order differs for paired contrast: {comparison}")
    delta_i = (
        candidate_frame["elpd_i"].to_numpy(dtype=float)
        - reference_frame["elpd_i"].to_numpy(dtype=float)
    )
    n = len(delta_i)
    delta = float(delta_i.sum())
    dse = float(np.sqrt(n * np.var(delta_i, ddof=0)))
    return {
        "comparison": comparison,
        "candidate_model": candidate.model,
        "reference_model": reference.model,
        "delta_elpd_candidate_minus_reference": delta,
        "paired_dse": dse,
        "delta_over_dse": delta / dse if dse > 0 else np.nan,
        "participant_count": n,
        "candidate_brier": candidate.metrics["brier"],
        "reference_brier": reference.metrics["brier"],
        "delta_brier_candidate_minus_reference": (
            candidate.metrics["brier"] - reference.metrics["brier"]
        ),
        "candidate_roc_auc": candidate.metrics["roc_auc"],
        "reference_roc_auc": reference.metrics["roc_auc"],
        "delta_auc_candidate_minus_reference": (
            candidate.metrics["roc_auc"] - reference.metrics["roc_auc"]
        ),
        "interpretation_rule": (
            "ELPD is primary; Brier and AUC are supplemental. Improvement smaller "
            "than paired uncertainty is not stable predictive evidence."
        ),
    }


def run_pipeline(
    data_path: str | Path,
    output_dir: str | Path,
    config: RunConfig,
    *,
    strict_known_data: bool = False,
    steps: Iterable[str] = ("all",),
) -> dict[str, Any]:
    selected = set(steps)
    if "all" in selected:
        selected = {"audit", "main", "sensitivity", "household", "report"}
    paths = ensure_output_tree(output_dir)
    started = datetime.now(timezone.utc)
    clean = load_and_validate(data_path, strict_known_data=strict_known_data)
    code_sha256, fingerprinted_files = _source_fingerprint()
    preliminary = config.mode == "quick"
    manifest: dict[str, Any] = {
        "status": "running",
        "preliminary": preliminary,
        "warning": "PRELIMINARY: quick mode is not for thesis reporting" if preliminary else None,
        "started_utc": started.isoformat(),
        "input_sha256": clean.sha256,
        "analysis_package_version": __version__,
        "analysis_code_sha256": code_sha256,
        "analysis_code_files_hashed": fingerprinted_files,
        "strict_known_data": strict_known_data,
        "requested_steps": sorted(selected),
        "preprocessing_validation_scope": (
            "Main/RQ2/RQ3/sensitivity/household PyMC designs are constructed on the full current "
            "analysis scenario before PSIS-LOO/reloo; their LOO estimates are conditional on frozen "
            "preprocessing. The independent MAP+Laplace and nested D paths use fold-local scaling."
        ),
        "config": config.to_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": _versions(),
        "completed_models": [],
    }
    write_json(paths["root"] / "manifest.json", manifest)
    write_json(paths["root"] / "validation.json", clean.report)
    # Identical participant profiles are an audit fact, not a deletion rule.
    source_wide = pd.read_csv(Path(data_path).resolve())
    write_csv(
        identical_profile_table(source_wide),
        paths["tables"] / "identical_participant_profiles_restricted.csv",
    )
    hh_gt = clean.frame.loc[
        clean.frame["household_contacts"] > clean.frame["contacts"],
        ["participant_id", "setting_group", "contacts", "household_contacts"],
    ].copy()
    write_csv(
        hh_gt,
        paths["tables"] / "household_contacts_gt_contacts_restricted.csv",
    )
    zero_profiles = (
        clean.frame.groupby(["participant_id", "outcome"], as_index=False)[
            ["participation", "activity_count", "contacts", "duration", "pcm",
             "household_contacts", "household_pcm"]
        ].sum()
    )
    zero_profiles = zero_profiles.loc[
        zero_profiles[["participation", "activity_count", "contacts", "duration",
                       "pcm", "household_contacts", "household_pcm"]]
        .eq(0).all(axis=1)
    ]
    write_csv(
        zero_profiles,
        paths["tables"] / "confirmed_zero_exposure_profiles_restricted.csv",
    )
    write_csv(_sample_audit_table(clean.report), paths["tables"] / "table_1_sample_audit.csv")
    plot_data_flow(clean.report, paths["figures"] / "fig_1_data_flow.png")
    if selected == {"audit"}:
        manifest["status"] = "complete"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(paths["root"] / "manifest.json", manifest)
        return manifest

    main_designs, exposure_long = build_designs(clean.frame, variant="main")
    write_csv(exposure_long, paths["tables"] / "participant_setting_exposure_main.csv")
    write_json(
        paths["root"] / "feature_metadata.json",
        {metric: design.metadata() for metric, design in main_designs.items()},
    )

    main_fits: dict[str, FitResult] = {}
    main_evaluations: dict[str, Evaluation] = {}
    diagnostics: list[dict[str, Any]] = []
    coefficients: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    ppc_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prior_predictive_rows: list[dict[str, Any]] = []
    main_settings = _settings(config, sensitivity=False)

    for metric in METRICS:
        name = f"main__{metric}"
        fit, evaluation, diagnostic, coefficient = _model_task(
            main_designs[metric],
            name=name,
            prior_scale=0.5,
            settings=main_settings,
            seed=config.derived_seed(name),
            include_household=False,
            exact_reloo=config.exact_reloo,
        )
        main_fits[name] = fit
        main_evaluations[name] = evaluation
        diagnostics.append(diagnostic)
        diagnostics.extend(evaluation.reloo_diagnostics)
        coefficients.append(coefficient)
        predictions.append(evaluation.predictions)
        ppc_by_model[name] = predictive_count_arrays(fit)
        prior_counts, posterior_counts = ppc_by_model[name]
        prior_predictive_rows.append(
            {
                "model": name,
                "metric": metric,
                "observed_positive": int(clean.report["positive_participants"]),
                "prior_count_q05": float(np.quantile(prior_counts, 0.05)),
                "prior_count_median": float(np.quantile(prior_counts, 0.50)),
                "prior_count_q95": float(np.quantile(prior_counts, 0.95)),
                "prior_probability_all_negative_or_all_positive": float(
                    np.mean((prior_counts == 0) | (prior_counts == len(main_designs[metric].y)))
                ),
                "posterior_count_q05": float(np.quantile(posterior_counts, 0.05)),
                "posterior_count_median": float(np.quantile(posterior_counts, 0.50)),
                "posterior_count_q95": float(np.quantile(posterior_counts, 0.95)),
            }
        )
        manifest["completed_models"].append(name)
        write_json(paths["root"] / "manifest.json", manifest)
        if config.save_idata:
            _save_idata(fit, paths["models"] / f"{name}.nc")

    main_comparison = compare_evaluations(main_evaluations)
    main_metrics = pd.DataFrame([value.metrics for value in main_evaluations.values()])
    main_comparison = main_comparison.merge(main_metrics, on="model", how="left", suffixes=("", "_metric"))
    ranks = bayesian_bootstrap_ranks(
        main_evaluations,
        draws=config.bootstrap_draws,
        seed=config.derived_seed("main_bootstrap"),
    )
    write_csv(main_comparison, paths["tables"] / "table_2_main_model_comparison.csv")
    write_csv(ranks, paths["tables"] / "main_rank_probabilities.csv")
    write_csv(
        pd.DataFrame(prior_predictive_rows),
        paths["tables"] / "prior_posterior_predictive_counts.csv",
    )
    plot_main_comparison(main_comparison, paths["figures"] / "fig_2_model_comparison.png")
    anchor_prior, anchor_posterior = ppc_by_model["main__participation"]
    plot_ppc(
        anchor_prior,
        anchor_posterior,
        observed_positive=int(clean.report["positive_participants"]),
        output=paths["figures"] / "fig_4_ppc_participation.png",
    )

    # ------------------------------------------------------------------
    # RQ2: controlled one-setting-at-a-time representation comparisons.
    # The Contacts design is the fixed anchor.  No locally preferred columns
    # are combined into a data-selected model.
    # ------------------------------------------------------------------
    manifest["research_question_contracts"] = {
        "RQ1": "Four common-metric participant-level models retained unchanged.",
        "RQ2": (
            "For each setting, replace only that Contacts-anchor column with each "
            "candidate representation; 36 new equal-complexity fits, no 4^12 search."
        ),
        "RQ3": (
            "Compare one outcome-independent extension-stage hybrid with each common-"
            "metric model using participant-level paired ELPD differences."
        ),
        "RQ4": (
            "Assess the separately supplied household-contact measure (HHcontacts) as additive "
            "occurrence/intensity information and one strongly shrunk setting-overlap modification score; "
            "Contacts is the project-defined non-household measure, and no subtraction is performed."
        ),
    }
    loo_gate_evaluations: list[Evaluation] = list(main_evaluations.values())
    controlled_designs, controlled_config = build_controlled_local_designs(
        main_designs, anchor="contacts"
    )
    write_csv(
        controlled_config,
        paths["tables"] / "table_7_setting_metric_candidate_configurations.csv",
    )
    controlled_lookup = {
        (row.setting, row.candidate_metric): row.design_key
        for row in controlled_config.itertuples(index=False)
    }
    setting_metric_rows: list[dict[str, Any]] = []
    setting_metric_comparisons: list[pd.DataFrame] = []
    setting_metric_rank_tables: list[pd.DataFrame] = []
    setting_preference_rows: list[dict[str, Any]] = []
    for setting_index, setting in enumerate(main_designs["contacts"].setting_names):
        setting_evaluations: dict[str, Evaluation] = {
            "contacts": main_evaluations["main__contacts"]
        }
        for metric in METRICS:
            if metric == "contacts":
                continue
            design_key = controlled_lookup[(setting, metric)]
            design = controlled_designs[design_key]
            name = f"rq2__{design_key.replace('controlled_local__', '')}"
            fit, evaluation, diagnostic, coefficient = _model_task(
                design,
                name=name,
                prior_scale=0.5,
                settings=main_settings,
                seed=config.derived_seed(name),
                include_household=False,
                exact_reloo=config.exact_reloo,
                sample_predictive=False,
            )
            setting_evaluations[metric] = evaluation
            diagnostics.append(diagnostic)
            diagnostics.extend(evaluation.reloo_diagnostics)
            coefficients.append(coefficient.assign(rq2_setting=setting))
            predictions.append(
                evaluation.predictions.assign(
                    analysis="rq2_controlled_local",
                    rq2_setting=setting,
                    candidate_metric=metric,
                )
            )
            loo_gate_evaluations.append(evaluation)
            manifest["completed_models"].append(name)
            write_json(paths["root"] / "manifest.json", manifest)
            if config.save_idata:
                _save_idata(fit, paths["models"] / f"{name}.nc")
            del fit
            gc.collect()

        local_comparison = compare_evaluations(setting_evaluations).assign(
            setting=setting
        )
        setting_metric_comparisons.append(local_comparison)
        local_ranks = bayesian_bootstrap_ranks(
            setting_evaluations,
            draws=config.bootstrap_draws,
            seed=config.derived_seed(f"rq2_bootstrap__{setting_index:02d}"),
        ).rename(columns={"model": "candidate_metric"})
        local_ranks["setting"] = setting
        setting_metric_rank_tables.append(local_ranks)

        ordered_metrics = sorted(
            METRICS,
            key=lambda value: setting_evaluations[value].metrics["elpd"],
            reverse=True,
        )
        point_best, runner_up = ordered_metrics[:2]
        best_vs_runner = _paired_elpd_summary(
            setting_evaluations[point_best],
            setting_evaluations[runner_up],
            comparison=f"{setting}: {point_best} minus {runner_up}",
        )
        rank_map = local_ranks.set_index("candidate_metric")[
            "probability_rank_1"
        ].to_dict()
        stable_preference = bool(
            rank_map[point_best] >= 0.80
            and best_vs_runner["delta_elpd_candidate_minus_reference"]
            > best_vs_runner["paired_dse"]
        )
        setting_preference_rows.append(
            {
                "setting": setting,
                "point_best_metric": point_best,
                "runner_up_metric": runner_up,
                "point_best_rank1_probability": rank_map[point_best],
                "delta_elpd_best_minus_runner_up": best_vs_runner[
                    "delta_elpd_candidate_minus_reference"
                ],
                "paired_dse_best_minus_runner_up": best_vs_runner["paired_dse"],
                "stable_preference": stable_preference,
                "stability_rule": (
                    "rank-1 probability >= 0.80 and delta ELPD > paired dSE"
                ),
            }
        )
        for metric in METRICS:
            evaluation = setting_evaluations[metric]
            versus_anchor = _paired_elpd_summary(
                evaluation,
                setting_evaluations["contacts"],
                comparison=f"{setting}: {metric} minus contacts anchor",
            )
            setting_metric_rows.append(
                {
                    "setting": setting,
                    "candidate_metric": metric,
                    "fitted_model": evaluation.model,
                    **evaluation.metrics,
                    "probability_rank_1": rank_map[metric],
                    "mean_rank": float(
                        local_ranks.set_index("candidate_metric").loc[
                            metric, "mean_rank"
                        ]
                    ),
                    "delta_elpd_vs_contacts": versus_anchor[
                        "delta_elpd_candidate_minus_reference"
                    ],
                    "paired_dse_vs_contacts": versus_anchor["paired_dse"],
                    "point_best_for_setting": metric == point_best,
                    "stable_preference_for_setting": (
                        stable_preference and metric == point_best
                    ),
                }
            )

    setting_metric_summary = pd.DataFrame(setting_metric_rows)
    setting_preference_summary = pd.DataFrame(setting_preference_rows)
    write_csv(
        setting_metric_summary,
        paths["tables"] / "table_8_setting_metric_predictive_support.csv",
    )
    write_csv(
        setting_preference_summary,
        paths["tables"] / "table_9_setting_metric_preference_summary.csv",
    )
    write_csv(
        pd.concat(setting_metric_comparisons, ignore_index=True),
        paths["tables"] / "setting_metric_arviz_comparisons.csv",
    )
    write_csv(
        pd.concat(setting_metric_rank_tables, ignore_index=True),
        paths["tables"] / "setting_metric_rank_probabilities.csv",
    )
    plot_setting_metric_support(
        setting_metric_summary[
            ["setting", "candidate_metric", "probability_rank_1"]
        ].rename(columns={"candidate_metric": "metric"}),
        paths["figures"] / "fig_7_setting_metric_support.png",
        value="probability_rank_1",
    )

    # RQ3: one scientifically defined, outcome-independent construction.  It
    # has the same 12 coefficient columns as every common-metric model and was
    # not assembled from the local LOO winners above.
    hybrid_design, hybrid_mapping = build_planned_hybrid_design(main_designs)
    write_csv(
        hybrid_mapping,
        paths["tables"] / "table_10_planned_hybrid_mapping.csv",
    )
    hybrid_name = "rq3__planned_hybrid"
    hybrid_fit, hybrid_evaluation, hybrid_diagnostic, hybrid_coefficient = _model_task(
        hybrid_design,
        name=hybrid_name,
        prior_scale=0.5,
        settings=main_settings,
        seed=config.derived_seed(hybrid_name),
        include_household=False,
        exact_reloo=config.exact_reloo,
        sample_predictive=False,
    )
    diagnostics.append(hybrid_diagnostic)
    diagnostics.extend(hybrid_evaluation.reloo_diagnostics)
    coefficients.append(hybrid_coefficient)
    predictions.append(
        hybrid_evaluation.predictions.assign(analysis="rq3_planned_hybrid")
    )
    loo_gate_evaluations.append(hybrid_evaluation)
    manifest["completed_models"].append(hybrid_name)
    hybrid_evaluations = {
        metric: evaluation for metric, evaluation in main_evaluations.items()
    }
    # Use readable keys rather than main__ prefixes in the RQ3 comparison.
    hybrid_evaluations = {
        key.replace("main__", ""): value
        for key, value in hybrid_evaluations.items()
    }
    hybrid_evaluations["planned_hybrid"] = hybrid_evaluation
    hybrid_comparison = compare_evaluations(hybrid_evaluations)
    hybrid_metric_rows = []
    for label, evaluation in hybrid_evaluations.items():
        hybrid_metric_rows.append({**evaluation.metrics, "model": label})
    hybrid_comparison = hybrid_comparison.merge(
        pd.DataFrame(hybrid_metric_rows), on="model", how="left", suffixes=("", "_metric")
    )
    hybrid_ranks = bayesian_bootstrap_ranks(
        hybrid_evaluations,
        draws=config.bootstrap_draws,
        seed=config.derived_seed("rq3_hybrid_bootstrap"),
    )
    hybrid_pairs = pd.DataFrame(
        [
            _paired_elpd_summary(
                hybrid_evaluation,
                main_evaluations[f"main__{metric}"],
                comparison=f"fixed_post_review_hybrid minus common_{metric}",
            )
            for metric in METRICS
        ]
    )
    write_csv(
        hybrid_comparison,
        paths["tables"] / "table_11_common_vs_planned_hybrid_comparison.csv",
    )
    write_csv(
        hybrid_pairs,
        paths["tables"] / "table_12_common_vs_hybrid_paired_differences.csv",
    )
    write_csv(
        hybrid_ranks,
        paths["tables"] / "common_vs_hybrid_rank_probabilities.csv",
    )
    hybrid_plot_frame = hybrid_pairs[
        ["delta_elpd_candidate_minus_reference", "paired_dse"]
    ].copy()
    hybrid_plot_frame.insert(
        0,
        "model",
        [
            "Fixed post-review hybrid minus Participation",
            "Fixed post-review hybrid minus Contact count",
            "Fixed post-review hybrid minus Duration",
            "Fixed post-review hybrid minus Person-contact minutes",
        ],
    )
    hybrid_plot_frame = hybrid_plot_frame.rename(
        columns={"delta_elpd_candidate_minus_reference": "delta_elpd"}
    )
    plot_common_vs_hybrid_delta(
        hybrid_plot_frame,
        paths["figures"] / "fig_8_common_vs_planned_hybrid.png",
    )
    if config.save_idata:
        _save_idata(hybrid_fit, paths["models"] / f"{hybrid_name}.nc")
    write_json(paths["root"] / "manifest.json", manifest)
    del hybrid_fit
    gc.collect()

    # Contacts/HHcontacts semantic boundary. The project brief defines Contacts as the
    # non-household contact measure and HHcontacts as the household-contact measure.
   
    write_json(
        paths["root"] / "contacts_household_semantic_contract.json",
        {
            "relationship": CONTACTS_HOUSEHOLD_RELATIONSHIP,
            "subtraction_performed": False,
            "hhcontacts_gt_contacts_cell_count": clean.report[
                "household_contacts_exceed_contacts_cell_count"
            ],
            "interpretation": (
                "Contacts is used as the project-defined non-household contact measure and HHcontacts "
                "as the household-contact measure. The wide file does not expose finer activity-level "
                "encoding, so the two supplied fields are kept parallel and are not subtracted."
            ),
        },
    )

    # De-identified outcome-stratified setting summaries support interpretation
    # of the model results without exposing participant-level records.
    setting_descriptive = (
        exposure_long.groupby(["setting_group", "outcome"], as_index=False)[
            list(METRICS)
        ]
        .agg(["mean", "median"])
    )
    setting_descriptive.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in setting_descriptive.columns
    ]
    write_csv(
        setting_descriptive,
        paths["tables"] / "setting_exposure_by_outcome_descriptive.csv",
    )
    descriptive_source = exposure_long.copy()
    descriptive_source["contacts_log1p"] = np.log1p(
        descriptive_source["contacts"].to_numpy(dtype=float)
    )
    setting_descriptive_tidy = (
        descriptive_source.groupby(["setting_group", "outcome"], as_index=False)[
            "contacts_log1p"
        ]
        .mean()
        .rename(
            columns={
                "setting_group": "setting",
                "outcome": "outcome_group",
                "contacts_log1p": "mean_exposure",
            }
        )
    )
    setting_descriptive_tidy["metric"] = "contacts"
    write_csv(
        setting_descriptive_tidy,
        paths["tables"] / "setting_contact_count_by_outcome_plot_data.csv",
    )
    plot_setting_outcome_descriptives(
        setting_descriptive_tidy,
        paths["figures"] / "fig_10_contact_count_by_outcome.png",
        value_column="mean_exposure",
        metric="contacts",
        xlabel="Mean log(1 + participant contact-count exposure)",
    )

    # The updated source is already participant-level.  There is no activity-row
    # duplicate-removal sensitivity.  Robustness now perturbs transformation, prior,
    # and separated-setting inclusion only.
    sensitivity_scenarios = [Scenario("raw_scale", continuous_transform="raw")]
    if config.mode == "full":
        sensitivity_scenarios.extend(
            [
                Scenario("stronger_prior", prior_scale=0.25),
                Scenario("weaker_prior", prior_scale=1.0),
                Scenario("drop_sparse", drop_sparse=True),
            ]
        )
    sensitivity_rows = [
        {
            "scenario": "main",
            "continuous_transform": "log1p",
            "prior_scale": 0.5,
            "drop_sparse": False,
            **evaluation.metrics,
        }
        for evaluation in main_evaluations.values()
    ]
    sensitivity_comparisons: list[pd.DataFrame] = [
        main_comparison.assign(scenario="main")
    ]
    sensitivity_rank_tables: list[pd.DataFrame] = [ranks.assign(scenario="main")]
    sensitivity_prediction_tables: list[pd.DataFrame] = [
        evaluation.predictions.assign(scenario="main")
        for evaluation in main_evaluations.values()
    ]
    sensitivity_settings = _settings(config, sensitivity=True)
    for scenario in sensitivity_scenarios:
        designs, _ = build_designs(
            clean.frame,
            continuous_transform=scenario.continuous_transform,
            drop_sparse=scenario.drop_sparse,
            variant=scenario.name,
        )
        scenario_evaluations: dict[str, Evaluation] = {}
        for metric in METRICS:
            name = f"{scenario.name}__{metric}"
            fit, evaluation, diagnostic, _ = _model_task(
                designs[metric],
                name=name,
                prior_scale=scenario.prior_scale,
                settings=sensitivity_settings,
                seed=config.derived_seed(name),
                include_household=False,
                exact_reloo=config.exact_reloo_sensitivity,
            )
            sensitivity_rows.append(
                {
                    "scenario": scenario.name,
                    "continuous_transform": scenario.continuous_transform,
                    "prior_scale": scenario.prior_scale,
                    "drop_sparse": scenario.drop_sparse,
                    **evaluation.metrics,
                }
            )
            scenario_evaluations[name] = evaluation
            sensitivity_prediction_tables.append(
                evaluation.predictions.assign(scenario=scenario.name)
            )
            diagnostics.append(diagnostic)
            diagnostics.extend(evaluation.reloo_diagnostics)
            manifest["completed_models"].append(name)
            write_json(paths["root"] / "manifest.json", manifest)
            if config.save_idata:
                _save_idata(fit, paths["models"] / f"{name}.nc")
            del fit
            gc.collect()
        scenario_comparison = compare_evaluations(scenario_evaluations)
        scenario_metrics = pd.DataFrame(
            [value.metrics for value in scenario_evaluations.values()]
        )
        scenario_comparison = scenario_comparison.merge(
            scenario_metrics, on="model", how="left", suffixes=("", "_metric")
        )
        sensitivity_comparisons.append(
            scenario_comparison.assign(scenario=scenario.name)
        )
        sensitivity_rank_tables.append(
            bayesian_bootstrap_ranks(
                scenario_evaluations,
                draws=config.bootstrap_draws,
                seed=config.derived_seed(f"{scenario.name}_bootstrap"),
            ).assign(scenario=scenario.name)
        )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["rank_within_scenario"] = sensitivity.groupby("scenario")["elpd"].rank(
        method="min", ascending=False
    )
    sensitivity["delta_elpd_to_scenario_best"] = sensitivity["elpd"] - sensitivity.groupby(
        "scenario"
    )["elpd"].transform("max")
    write_csv(sensitivity, paths["tables"] / "table_5_sensitivity.csv")
    write_csv(
        pd.concat(sensitivity_comparisons, ignore_index=True),
        paths["tables"] / "sensitivity_model_comparisons.csv",
    )
    write_csv(
        pd.concat(sensitivity_rank_tables, ignore_index=True),
        paths["tables"] / "sensitivity_rank_probabilities.csv",
    )
    write_csv(
        pd.concat(sensitivity_prediction_tables, ignore_index=True),
        paths["tables"] / "sensitivity_loo_predictions.csv",
    )
    plot_sensitivity(sensitivity, paths["figures"] / "fig_5_sensitivity.png")

    anchor_name = "main__participation"
    household_name = "main__participation_household"
    household_fit, household_evaluation, household_diagnostic, household_coefficient = _model_task(
        main_designs["participation"],
        name=household_name,
        prior_scale=0.5,
        settings=main_settings,
        seed=config.derived_seed(household_name),
        include_household=True,
        exact_reloo=config.exact_reloo,
    )
    household_evaluations = {
        anchor_name: main_evaluations[anchor_name],
        household_name: household_evaluation,
    }
    household_comparison = compare_evaluations(household_evaluations)
    household_metrics = pd.DataFrame(
        [evaluation.metrics for evaluation in household_evaluations.values()]
    )
    household_comparison = household_comparison.merge(
        household_metrics, on="model", how="left", suffixes=("", "_metric")
    )
    diagnostics.append(household_diagnostic)
    diagnostics.extend(household_evaluation.reloo_diagnostics)
    coefficients.append(household_coefficient)
    predictions.append(household_evaluation.predictions)
    manifest["completed_models"].append(household_name)
    write_csv(household_comparison, paths["tables"] / "table_6_household_extension.csv")
    plot_household(household_comparison, paths["figures"] / "fig_6_household_extension.png")
    if config.save_idata:
        _save_idata(household_fit, paths["models"] / f"{household_name}.nc")

    # ------------------------------------------------------------------
    # RQ4: setting-aware household co-participation.  The additive model asks
    # whether household occurrence/intensity add predictive information.  The
    # modification model asks the separate, narrower question of whether a
    # single outcome-blind setting-overlap score modifies the duration-based
    # activity association.  It is not 12 unrestricted interactions.
    # ------------------------------------------------------------------
    household_predictors, household_setting_long, household_structured_audit = (
        build_household_structured_predictors(
            clean.frame, main_designs["duration"]
        )
    )
    write_csv(
        household_setting_long,
        paths["tables"] / "household_participant_setting_deidentified.csv",
    )
    write_json(
        paths["root"] / "household_structured_feature_audit.json",
        household_structured_audit,
    )
    additive_auxiliary = {
        "gamma_household_occurrence": household_predictors[
            "household_occurrence_centered"
        ],
        "gamma_household_intensity": household_predictors[
            "household_intensity_within_occurrence_z"
        ],
    }
    additive_definitions = {
        key: household_structured_audit["definitions"][source]
        for key, source in {
            "gamma_household_occurrence": "household_occurrence_centered",
            "gamma_household_intensity": (
                "household_intensity_within_occurrence_z"
            ),
        }.items()
    }
    additive_design = replace(
        main_designs["duration"],
        metric="duration_household_additive",
        variant="rq4_household_additive",
        auxiliary_predictors={
            key: np.asarray(value, dtype=float).copy()
            for key, value in additive_auxiliary.items()
        },
        auxiliary_definitions=additive_definitions,
    )
    modification_design = replace(
        additive_design,
        metric="duration_household_modification",
        variant="rq4_household_additive_plus_overlap_modification",
        auxiliary_predictors={
            **additive_design.auxiliary_predictors,
            "psi_household_overlap": household_predictors[
                "setting_overlap_modification_z"
            ].copy(),
        },
        auxiliary_definitions={
            **additive_design.auxiliary_definitions,
            "psi_household_overlap": household_structured_audit["definitions"][
                "setting_overlap_modification_z"
            ],
        },
    )
    additive_name = "rq4__duration_household_additive"
    additive_fit, additive_evaluation, additive_diagnostic, additive_coefficient = (
        _model_task(
            additive_design,
            name=additive_name,
            prior_scale=0.5,
            settings=main_settings,
            seed=config.derived_seed(additive_name),
            include_household=False,
            exact_reloo=config.exact_reloo,
            auxiliary_prior_scales={
                "gamma_household_occurrence": 0.75,
                "gamma_household_intensity": 0.75,
            },
            sample_predictive=False,
        )
    )
    modification_name = "rq4__duration_household_modification"
    (
        modification_fit,
        modification_evaluation,
        modification_diagnostic,
        modification_coefficient,
    ) = _model_task(
        modification_design,
        name=modification_name,
        prior_scale=0.5,
        settings=main_settings,
        seed=config.derived_seed(modification_name),
        include_household=False,
        exact_reloo=config.exact_reloo,
        auxiliary_prior_scales={
            "gamma_household_occurrence": 0.75,
            "gamma_household_intensity": 0.75,
            "psi_household_overlap": 0.35,
        },
        sample_predictive=False,
    )
    for evaluation, diagnostic, coefficient, role in (
        (
            additive_evaluation,
            additive_diagnostic,
            additive_coefficient,
            "additive",
        ),
        (
            modification_evaluation,
            modification_diagnostic,
            modification_coefficient,
            "modification",
        ),
    ):
        diagnostics.append(diagnostic)
        diagnostics.extend(evaluation.reloo_diagnostics)
        coefficients.append(coefficient.assign(rq4_model_role=role))
        predictions.append(
            evaluation.predictions.assign(
                analysis="rq4_household_structured", rq4_model_role=role
            )
        )
        loo_gate_evaluations.append(evaluation)
        manifest["completed_models"].append(evaluation.model)
    rq4_evaluations = {
        "duration_baseline": main_evaluations["main__duration"],
        "household_additive": additive_evaluation,
        "household_modification": modification_evaluation,
    }
    rq4_comparison = compare_evaluations(rq4_evaluations)
    rq4_metric_rows = pd.DataFrame(
        [
            {**evaluation.metrics, "model": label}
            for label, evaluation in rq4_evaluations.items()
        ]
    )
    rq4_comparison = rq4_comparison.merge(
        rq4_metric_rows, on="model", how="left", suffixes=("", "_metric")
    )
    rq4_paired = pd.DataFrame(
        [
            _paired_elpd_summary(
                additive_evaluation,
                main_evaluations["main__duration"],
                comparison="household_additive minus duration_baseline",
            ),
            _paired_elpd_summary(
                modification_evaluation,
                additive_evaluation,
                comparison="household_modification minus household_additive",
            ),
            _paired_elpd_summary(
                modification_evaluation,
                main_evaluations["main__duration"],
                comparison="household_modification minus duration_baseline",
            ),
        ]
    )
    rq4_ranks = bayesian_bootstrap_ranks(
        rq4_evaluations,
        draws=config.bootstrap_draws,
        seed=config.derived_seed("rq4_household_bootstrap"),
    )
    rq4_effects = pd.concat(
        [additive_coefficient, modification_coefficient], ignore_index=True
    )
    rq4_effects = rq4_effects[
        rq4_effects["parameter"].str.startswith(("gamma_household", "psi_household"))
    ].copy()
    write_csv(
        rq4_comparison,
        paths["tables"] / "table_14_household_additive_modification_comparison.csv",
    )
    write_csv(
        rq4_paired,
        paths["tables"] / "table_15_household_paired_differences.csv",
    )
    write_csv(
        rq4_effects,
        paths["tables"] / "table_16_household_structured_effects.csv",
    )
    write_csv(
        rq4_ranks,
        paths["tables"] / "household_structured_rank_probabilities.csv",
    )
    plot_household_extension_comparison(
        rq4_paired.rename(
            columns={
                "comparison": "contrast",
                "delta_elpd_candidate_minus_reference": "delta_elpd",
            }
        ),
        paths["figures"] / "fig_9_household_additive_modification.png",
    )

    # Strong-prior sensitivity applies only to the single modification
    # coefficient; the activity slopes and additive household priors are fixed.
    modification_prior_rows: list[dict[str, Any]] = []
    modification_prior_coefficients: list[pd.DataFrame] = []
    modification_prior_evaluations: list[Evaluation] = []
    if config.mode == "full":
        for psi_scale in (0.25, 0.50):
            scale_label = str(psi_scale).replace(".", "p")
            name = f"rq4_sensitivity__psi_sd_{scale_label}"
            fit, evaluation, diagnostic, coefficient = _model_task(
                modification_design,
                name=name,
                prior_scale=0.5,
                settings=sensitivity_settings,
                seed=config.derived_seed(name),
                include_household=False,
                exact_reloo=config.exact_reloo_sensitivity,
                auxiliary_prior_scales={
                    "gamma_household_occurrence": 0.75,
                    "gamma_household_intensity": 0.75,
                    "psi_household_overlap": psi_scale,
                },
                sample_predictive=False,
            )
            diagnostics.append(diagnostic)
            diagnostics.extend(evaluation.reloo_diagnostics)
            coefficients.append(coefficient.assign(rq4_model_role="prior_sensitivity"))
            predictions.append(
                evaluation.predictions.assign(
                    analysis="rq4_modification_prior_sensitivity",
                    psi_prior_sd=psi_scale,
                )
            )
            modification_prior_evaluations.append(evaluation)
            modification_prior_rows.append(
                {"psi_prior_sd": psi_scale, **evaluation.metrics}
            )
            modification_prior_coefficients.append(
                coefficient.assign(psi_prior_sd=psi_scale)
            )
            manifest["completed_models"].append(name)
            if config.save_idata:
                _save_idata(fit, paths["models"] / f"{name}.nc")
            del fit
            gc.collect()
    write_csv(
        pd.DataFrame(modification_prior_rows),
        paths["tables"] / "household_modification_prior_sensitivity.csv",
    )
    write_csv(
        (
            pd.concat(modification_prior_coefficients, ignore_index=True)
            if modification_prior_coefficients
            else pd.DataFrame()
        ),
        paths["tables"] / "household_modification_prior_coefficients.csv",
    )
    if config.save_idata:
        _save_idata(additive_fit, paths["models"] / f"{additive_name}.nc")
        _save_idata(modification_fit, paths["models"] / f"{modification_name}.nc")
    write_json(paths["root"] / "manifest.json", manifest)
    del additive_fit, modification_fit
    gc.collect()

    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    write_csv(coefficient_frame, paths["tables"] / "table_3_setting_coefficients.csv")
    write_csv(prediction_frame, paths["tables"] / "table_4_loo_predictions.csv")
    write_csv(diagnostic_frame, paths["tables"] / "model_diagnostics.csv")
    plot_setting_forest(coefficient_frame, paths["figures"] / "fig_3_setting_forest.png")

    if config.mode == "full":
        checks = {
            "zero_divergences": bool((diagnostic_frame["divergences"] == 0).all()),
            "zero_max_treedepth_hits": bool(
                (diagnostic_frame["reached_max_treedepth_count"] == 0).all()
            ),
            "max_rhat_le_1_01": bool(
                diagnostic_frame["max_rhat"].notna().all()
                and (diagnostic_frame["max_rhat"] <= 1.01).all()
            ),
            "min_ess_bulk_ge_400": bool(
                diagnostic_frame["min_ess_bulk"].notna().all()
                and (diagnostic_frame["min_ess_bulk"] >= 400).all()
            ),
            "min_ess_tail_ge_400": bool(
                diagnostic_frame["min_ess_tail"].notna().all()
                and (diagnostic_frame["min_ess_tail"] >= 400).all()
            ),
            "min_bfmi_gt_0_3": bool(
                diagnostic_frame["min_bfmi"].notna().all()
                and (diagnostic_frame["min_bfmi"] > 0.3).all()
            ),
        }
        manifest["diagnostic_gate"] = {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "rule": "Full results are thesis-ready only when this gate passes.",
        }
        primary_loo_rows = pd.DataFrame(
            [evaluation.metrics for evaluation in loo_gate_evaluations]
            + [household_evaluation.metrics]
        )
        extension_sensitivity_loo_rows = pd.DataFrame(
            [evaluation.metrics for evaluation in modification_prior_evaluations]
        )
        extension_sensitivity_repaired = (
            True
            if extension_sensitivity_loo_rows.empty
            else bool(
                (
                    extension_sensitivity_loo_rows["high_k_count_initial"]
                    == extension_sensitivity_loo_rows["exact_refit_count"]
                ).all()
            )
        )
        loo_checks = {
            "all_primary_high_k_repaired": bool(
                (
                    primary_loo_rows["high_k_count_initial"]
                    == primary_loo_rows["exact_refit_count"]
                ).all()
            ),
            "all_sensitivity_high_k_repaired": bool(
                (
                    sensitivity["high_k_count_initial"]
                    == sensitivity["exact_refit_count"]
                ).all()
            ),
            "all_household_prior_sensitivity_high_k_repaired": (
                extension_sensitivity_repaired
            ),
        }
        manifest["loo_reliability_gate"] = {
            "status": "pass" if all(loo_checks.values()) else "fail",
            "checks": loo_checks,
            "rule": "Every Pareto-k value above good_k must be replaced by exact reloo in full mode.",
        }
    else:
        manifest["diagnostic_gate"] = {
            "status": "not_evaluated",
            "rule": "Quick/smoke chains are not inferential diagnostics.",
        }
        manifest["loo_reliability_gate"] = {
            "status": "not_evaluated",
            "rule": "Quick/smoke mode may leave high Pareto-k observations unrepaired.",
        }

    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["model_count"] = len(manifest["completed_models"])
    manifest["model_count_scope"] = {
        "core_bayesian_full_data_models": len(manifest["completed_models"]),
        "includes_household_modification_prior_sensitivity_models": 2,
        "continuous_household_auxiliary_stage_counted_separately": True,
        "note": (
            "The core Bayesian stage count includes two psi-prior sensitivity fits. "
            "The post-review continuous-household auxiliary stage runs under a separate manifest."
        ),
    }
    manifest["outputs"] = {
        "main_comparison": "tables/table_2_main_model_comparison.csv",
        "coefficients": "tables/table_3_setting_coefficients.csv",
        "loo_predictions": "tables/table_4_loo_predictions.csv",
        "sensitivity": "tables/table_5_sensitivity.csv",
        "household": "tables/table_6_household_extension.csv",
        "main_rank_probabilities": "tables/main_rank_probabilities.csv",
        "prior_posterior_predictive_counts": "tables/prior_posterior_predictive_counts.csv",
        "model_diagnostics": "tables/model_diagnostics.csv",
        "sensitivity_model_comparisons": "tables/sensitivity_model_comparisons.csv",
        "sensitivity_rank_probabilities": "tables/sensitivity_rank_probabilities.csv",
        "sensitivity_loo_predictions": "tables/sensitivity_loo_predictions.csv",
        "setting_metric_configurations": (
            "tables/table_7_setting_metric_candidate_configurations.csv"
        ),
        "setting_metric_support": (
            "tables/table_8_setting_metric_predictive_support.csv"
        ),
        "setting_metric_preferences": (
            "tables/table_9_setting_metric_preference_summary.csv"
        ),
        "planned_hybrid_mapping": "tables/table_10_planned_hybrid_mapping.csv",  # legacy filename; mapping is fixed post-review, not preregistered
        "common_vs_hybrid": (
            "tables/table_11_common_vs_planned_hybrid_comparison.csv"
        ),
        "common_vs_hybrid_paired": (
            "tables/table_12_common_vs_hybrid_paired_differences.csv"
        ),
        "contacts_household_semantic_contract": (
            "contacts_household_semantic_contract.json"
        ),
        "household_additive_modification": (
            "tables/table_14_household_additive_modification_comparison.csv"
        ),
        "household_paired_differences": (
            "tables/table_15_household_paired_differences.csv"
        ),
        "household_structured_effects": (
            "tables/table_16_household_structured_effects.csv"
        ),
        "contact_semantic_audit": "contacts_household_semantic_contract.json",
        "household_structured_feature_audit": (
            "household_structured_feature_audit.json"
        ),
    }
    artifact_files = sorted(
        [path for folder in (paths["tables"], paths["figures"], paths["models"]) for path in folder.rglob("*") if path.is_file()],
        key=lambda path: path.relative_to(paths["root"]).as_posix(),
    )
    manifest["artifact_sha256"] = {
        path.relative_to(paths["root"]).as_posix(): _file_sha256(path)
        for path in artifact_files
    }
    write_json(paths["root"] / "manifest.json", manifest)
    return manifest

# =============================================================================
# 8A. Core independent MAP+Laplace approximate robustness validation
# =============================================================================

INTERCEPT_SD = 1.5
SLOPE_SD = 0.5
# Robustness-path model note: unlike the main PyMC model, this validator uses
# independent fixed Normal slope priors (no shared hierarchical tau), centres
# participation, scales continuous log1p predictors by 2 training-fold SDs, and
# recomputes all transformations within each LOPO training fold. Its purpose is
# robustness checking across a deliberately different prior/scaling/approximation path.
CONTINUOUS_MODELS = ("contacts", "duration_minutes", "pcm")
EXPOSURE_MODELS = ("participation",) + CONTINUOUS_MODELS
ALL_MODELS = EXPOSURE_MODELS + ("intercept_only",)
N_GAUSS_HERMITE = 40
GH_NODES, GH_WEIGHTS = np.polynomial.hermite.hermgauss(N_GAUSS_HERMITE)


@dataclass(frozen=True)
class ExposureData:
    variant: str
    source: Path
    participant_ids: tuple[str, ...]
    settings: tuple[str, ...]
    outcomes: np.ndarray
    matrices: dict[str, np.ndarray]


@dataclass(frozen=True)
class ScaleSpec:
    model: str
    transform: str
    mean: np.ndarray
    sd: np.ndarray
    denominator: np.ndarray
    constant: np.ndarray


@dataclass(frozen=True)
class LaplaceFit:
    mean: np.ndarray
    covariance: np.ndarray
    objective: float
    optimizer_method: str
    optimizer_reported_success: bool
    accepted: bool
    status: int
    message: str
    iterations: int
    gradient_norm: float
    hessian_min_eigenvalue: float
    hessian_condition_number: float


def _participant_sort_key(value: str) -> tuple[int, float | str, str]:
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, value, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exposure_data(path: Path, variant: str) -> ExposureData:
    required = {
        "participant_id",
        "outcome",
        "setting_group",
        "participation",
        "contacts",
        "duration_minutes",
        "pcm",
    }
    records: dict[str, dict[str, dict[str, float]]] = {}
    outcomes_by_id: dict[str, int] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            participant = row["participant_id"].strip()
            setting = row["setting_group"].strip()
            if not participant or not setting:
                raise ValueError(f"{path}:{line_number}: blank participant or setting")
            try:
                outcome_float = float(row["outcome"])
                values = {model: float(row[model]) for model in EXPOSURE_MODELS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from exc
            outcome = int(outcome_float)
            if outcome_float != outcome or outcome not in (0, 1):
                raise ValueError(f"{path}:{line_number}: outcome must be 0 or 1")
            if any((not math.isfinite(value) or value < 0.0) for value in values.values()):
                raise ValueError(f"{path}:{line_number}: exposures must be finite and non-negative")
            if values["participation"] not in (0.0, 1.0):
                raise ValueError(f"{path}:{line_number}: participation must be binary")
            if participant in outcomes_by_id and outcomes_by_id[participant] != outcome:
                raise ValueError(f"{path}: inconsistent outcomes for participant {participant}")
            outcomes_by_id[participant] = outcome
            participant_records = records.setdefault(participant, {})
            if setting in participant_records:
                raise ValueError(f"{path}: duplicate participant-setting cell {participant!r}, {setting!r}")
            participant_records[setting] = values

    if not records:
        raise ValueError(f"{path}: no data rows")
    participant_ids = tuple(sorted(records, key=_participant_sort_key))
    settings = tuple(sorted(next(iter(records.values()))))
    expected_settings = set(settings)
    for participant in participant_ids:
        observed_settings = set(records[participant])
        if observed_settings != expected_settings:
            missing = sorted(expected_settings - observed_settings)
            extra = sorted(observed_settings - expected_settings)
            raise ValueError(
                f"{path}: participant {participant} has an unbalanced setting grid; "
                f"missing={missing}, extra={extra}"
            )

    outcomes = np.asarray([outcomes_by_id[participant] for participant in participant_ids], dtype=float)
    matrices = {
        model: np.asarray(
            [
                [records[participant][setting][model] for setting in settings]
                for participant in participant_ids
            ],
            dtype=float,
        )
        for model in EXPOSURE_MODELS
    }
    return ExposureData(
        variant=variant,
        source=path.resolve(),
        participant_ids=participant_ids,
        settings=settings,
        outcomes=outcomes,
        matrices=matrices,
    )


def verify_data_pair(raw: ExposureData, dedup: ExposureData) -> None:
    if raw.participant_ids != dedup.participant_ids:
        raise ValueError("raw and exact-dedup participant IDs do not match")
    if raw.settings != dedup.settings:
        raise ValueError("raw and exact-dedup setting grids do not match")
    if not np.array_equal(raw.outcomes, dedup.outcomes):
        raise ValueError("raw and exact-dedup outcomes do not match")
    if not np.array_equal(raw.matrices["participation"], dedup.matrices["participation"]):
        raise ValueError("participation should be invariant to exact-row deduplication")


def fit_scale_spec(train_raw: np.ndarray, model: str) -> ScaleSpec:
    if train_raw.ndim != 2:
        raise ValueError("training exposure matrix must be two-dimensional")
    if model == "participation":
        transformed = train_raw.astype(float, copy=False)
        transform = "center_only"
    elif model in CONTINUOUS_MODELS:
        transformed = np.log1p(train_raw)
        transform = "log1p_center_div_2sd"
    else:
        raise ValueError(f"unknown exposure model: {model}")

    mean = transformed.mean(axis=0)
    sd = transformed.std(axis=0, ddof=1)
    constant = (~np.isfinite(sd)) | (sd <= np.finfo(float).eps)
    if model == "participation":
        denominator = np.ones_like(sd)
    else:
        denominator = 2.0 * sd
        denominator = np.where(constant, 1.0, denominator)
    return ScaleSpec(model, transform, mean, sd, denominator, constant)


def apply_scale(raw: np.ndarray, spec: ScaleSpec) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=float)
    transformed = values if spec.model == "participation" else np.log1p(values)
    scaled = (transformed - spec.mean) / spec.denominator
    if np.any(spec.constant):
        # A training-constant column contributes exactly zero in training.  A novel
        # held-out value is left on the transformed unit scale, so its uncertainty
        # is propagated through the proper slope prior instead of being magnified.
        scaled = np.where(spec.constant, transformed - spec.mean, scaled)
    return np.asarray(transformed, dtype=float), np.asarray(scaled, dtype=float)


def negative_log_posterior(
    theta: np.ndarray, design: np.ndarray, outcomes: np.ndarray, precision: np.ndarray
) -> float:
    eta = design @ theta
    return float(np.logaddexp(0.0, eta).sum() - outcomes @ eta + 0.5 * theta @ (precision * theta))


def negative_log_posterior_gradient(
    theta: np.ndarray, design: np.ndarray, outcomes: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    eta = design @ theta
    return design.T @ (expit(eta) - outcomes) + precision * theta


def negative_log_posterior_hessian(
    theta: np.ndarray, design: np.ndarray, outcomes: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    del outcomes  # the Bernoulli Hessian does not depend directly on y
    probabilities = expit(design @ theta)
    weights = probabilities * (1.0 - probabilities)
    return design.T @ (design * weights[:, None]) + np.diag(precision)


def fit_map_laplace(design: np.ndarray, outcomes: np.ndarray) -> LaplaceFit:
    n_parameters = design.shape[1]
    precision = np.full(n_parameters, 1.0 / (SLOPE_SD**2), dtype=float)
    precision[0] = 1.0 / (INTERCEPT_SD**2)
    positives = float(outcomes.sum())
    initial = np.zeros(n_parameters, dtype=float)
    initial[0] = math.log((positives + 0.5) / (len(outcomes) - positives + 0.5))
    args = (design, outcomes, precision)

    result = minimize(
        negative_log_posterior,
        initial,
        args=args,
        method="trust-exact",
        jac=negative_log_posterior_gradient,
        hess=negative_log_posterior_hessian,
        options={"gtol": 1e-7, "maxiter": 1000},
    )
    method = "trust-exact"
    gradient = negative_log_posterior_gradient(result.x, *args)
    if not result.success:
        fallback = minimize(
            negative_log_posterior,
            result.x,
            args=args,
            method="L-BFGS-B",
            jac=negative_log_posterior_gradient,
            options={"ftol": 1e-14, "gtol": 1e-7, "maxiter": 2000, "maxls": 100},
        )
        fallback_gradient = negative_log_posterior_gradient(fallback.x, *args)
        if (
            fallback.success
            or fallback.fun < result.fun
            or np.linalg.norm(fallback_gradient, ord=np.inf) < np.linalg.norm(gradient, ord=np.inf)
        ):
            result = fallback
            gradient = fallback_gradient
            method = "L-BFGS-B_fallback"

    hessian = negative_log_posterior_hessian(result.x, *args)
    eigenvalues = np.linalg.eigvalsh(hessian)
    min_eigenvalue = float(eigenvalues[0])
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    accepted = bool(min_eigenvalue > 0.0 and gradient_norm <= 1e-6 and np.all(np.isfinite(result.x)))
    if not accepted:
        raise RuntimeError(
            f"MAP optimization failed acceptance: success={result.success}, "
            f"gradient_inf={gradient_norm:.3g}, min_eigenvalue={min_eigenvalue:.3g}, "
            f"message={result.message}"
        )
    covariance = np.linalg.solve(hessian, np.eye(n_parameters))
    covariance = 0.5 * (covariance + covariance.T)
    return LaplaceFit(
        mean=np.asarray(result.x, dtype=float),
        covariance=covariance,
        objective=float(result.fun),
        optimizer_method=method,
        optimizer_reported_success=bool(result.success),
        accepted=accepted,
        status=int(result.status),
        message=str(result.message),
        iterations=int(getattr(result, "nit", -1)),
        gradient_norm=gradient_norm,
        hessian_min_eigenvalue=min_eigenvalue,
        hessian_condition_number=condition_number,
    )


def laplace_predictive_probability(
    design_row: np.ndarray, fit: LaplaceFit
) -> tuple[float, float, float]:
    mean = float(design_row @ fit.mean)
    variance = float(design_row @ fit.covariance @ design_row)
    if variance < 0.0 and variance > -1e-12:
        variance = 0.0
    if variance < 0.0 or not math.isfinite(variance):
        raise RuntimeError(f"invalid Laplace predictive variance {variance}")
    if variance == 0.0:
        probability = float(expit(mean))
    else:
        logits = mean + math.sqrt(2.0 * variance) * GH_NODES
        probability = float(np.dot(GH_WEIGHTS, expit(logits)) / math.sqrt(math.pi))
    probability = float(np.clip(probability, 1e-15, 1.0 - 1e-15))
    return probability, mean, variance


def binary_auc(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    positives = outcomes == 1.0
    n_positive = int(positives.sum())
    n_negative = int((~positives).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(probabilities, method="average")
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def run_loo_for_dataset(data: ExposureData) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions: list[dict[str, object]] = []
    scaling_rows: list[dict[str, object]] = []
    n = len(data.participant_ids)

    for model in ALL_MODELS:
        for heldout_index, participant in enumerate(data.participant_ids):
            train_mask = np.arange(n) != heldout_index
            train_outcomes = data.outcomes[train_mask]
            if model == "intercept_only":
                train_design = np.ones((n - 1, 1), dtype=float)
                heldout_design = np.ones(1, dtype=float)
            else:
                raw_matrix = data.matrices[model]
                spec = fit_scale_spec(raw_matrix[train_mask], model)
                _, train_scaled = apply_scale(raw_matrix[train_mask], spec)
                heldout_transformed, heldout_scaled = apply_scale(raw_matrix[heldout_index], spec)
                train_design = np.column_stack((np.ones(n - 1, dtype=float), train_scaled))
                heldout_design = np.concatenate(([1.0], heldout_scaled))
                for setting_index, setting in enumerate(data.settings):
                    scaling_rows.append(
                        {
                            "data_variant": data.variant,
                            "model": model,
                            "heldout_participant_id": participant,
                            "setting_group": setting,
                            "transform": spec.transform,
                            "train_mean_transformed": float(spec.mean[setting_index]),
                            "train_sd_transformed": float(spec.sd[setting_index]),
                            "scale_denominator": float(spec.denominator[setting_index]),
                            "training_constant": int(spec.constant[setting_index]),
                            "heldout_raw": float(raw_matrix[heldout_index, setting_index]),
                            "heldout_transformed": float(heldout_transformed[setting_index]),
                            "heldout_scaled": float(heldout_scaled[setting_index]),
                        }
                    )

            fit = fit_map_laplace(train_design, train_outcomes)
            probability, logit_mean, logit_variance = laplace_predictive_probability(heldout_design, fit)
            outcome = int(data.outcomes[heldout_index])
            log_predictive_density = math.log(probability if outcome == 1 else 1.0 - probability)
            predictions.append(
                {
                    "data_variant": data.variant,
                    "model": model,
                    "participant_id": participant,
                    "outcome": outcome,
                    "predicted_probability": probability,
                    "log_predictive_density": log_predictive_density,
                    "brier_contribution": (outcome - probability) ** 2,
                    "heldout_logit_mean": logit_mean,
                    "heldout_logit_variance": logit_variance,
                    "n_train": n - 1,
                    "train_positives": int(train_outcomes.sum()),
                    "train_negatives": int(len(train_outcomes) - train_outcomes.sum()),
                    "optimizer_method": fit.optimizer_method,
                    "optimizer_reported_success": int(fit.optimizer_reported_success),
                    "fit_accepted": int(fit.accepted),
                    "optimizer_status": fit.status,
                    "optimizer_message": fit.message,
                    "optimizer_iterations": fit.iterations,
                    "map_negative_log_posterior": fit.objective,
                    "gradient_inf_norm": fit.gradient_norm,
                    "hessian_min_eigenvalue": fit.hessian_min_eigenvalue,
                    "hessian_condition_number": fit.hessian_condition_number,
                }
            )
    return predictions, scaling_rows


def _group_predictions(
    predictions: Sequence[dict[str, object]], variant: str, model: str
) -> list[dict[str, object]]:
    rows = [
        row
        for row in predictions
        if row["data_variant"] == variant and row["model"] == model
    ]
    return sorted(rows, key=lambda row: _participant_sort_key(str(row["participant_id"])))


def calculate_metrics(
    predictions: Sequence[dict[str, object]], variants: Sequence[str]
) -> list[dict[str, object]]:
    metrics: list[dict[str, object]] = []
    for variant in variants:
        variant_rows: list[dict[str, object]] = []
        for model in ALL_MODELS:
            rows = _group_predictions(predictions, variant, model)
            outcomes = np.asarray([row["outcome"] for row in rows], dtype=float)
            probabilities = np.asarray([row["predicted_probability"] for row in rows], dtype=float)
            log_density = np.asarray([row["log_predictive_density"] for row in rows], dtype=float)
            brier = np.asarray([row["brier_contribution"] for row in rows], dtype=float)
            variant_rows.append(
                {
                    "data_variant": variant,
                    "model": model,
                    "n": len(rows),
                    "positives": int(outcomes.sum()),
                    "negatives": int(len(outcomes) - outcomes.sum()),
                    "elpd": float(log_density.sum()),
                    "elpd_se": float(math.sqrt(len(rows) * np.var(log_density, ddof=1))),
                    "mean_log_score": float(log_density.mean()),
                    "brier_score": float(brier.mean()),
                    # A pooled leave-one-out intercept probability changes slightly
                    # with the held-out outcome because the training prevalence changes.
                    # The intercept-only model has no discriminating covariate, so its
                    # structural discrimination benchmark is defined as AUC=0.5.
                    "auc": 0.5 if model == "intercept_only" else binary_auc(outcomes, probabilities),
                    "nominal_rank_exposure": "",
                    "nominal_rank_all": "",
                    "optimizer_reported_failures": sum(
                        int(row["optimizer_reported_success"]) == 0 for row in rows
                    ),
                    "fit_acceptance_failures": sum(int(row["fit_accepted"]) == 0 for row in rows),
                    "max_gradient_inf_norm": max(float(row["gradient_inf_norm"]) for row in rows),
                    "min_hessian_eigenvalue": min(
                        float(row["hessian_min_eigenvalue"]) for row in rows
                    ),
                    "max_hessian_condition_number": max(
                        float(row["hessian_condition_number"]) for row in rows
                    ),
                }
            )
        all_order = sorted(variant_rows, key=lambda row: float(row["elpd"]), reverse=True)
        for rank, row in enumerate(all_order, start=1):
            row["nominal_rank_all"] = rank
        exposure_order = sorted(
            [row for row in variant_rows if row["model"] in EXPOSURE_MODELS],
            key=lambda row: float(row["elpd"]),
            reverse=True,
        )
        for rank, row in enumerate(exposure_order, start=1):
            row["nominal_rank_exposure"] = rank
        metrics.extend(variant_rows)
    return metrics


def bayesian_bootstrap_outputs(
    predictions: Sequence[dict[str, object]],
    variants: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rank_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    n = len({str(row["participant_id"]) for row in predictions})

    for variant_index, variant in enumerate(variants):
        rng = np.random.default_rng(seed + variant_index)
        weights = rng.dirichlet(np.ones(n), size=n_bootstrap)
        pointwise = {
            model: np.asarray(
                [row["log_predictive_density"] for row in _group_predictions(predictions, variant, model)],
                dtype=float,
            )
            for model in ALL_MODELS
        }

        for scope, models in (
            ("exposure_models", EXPOSURE_MODELS),
            ("all_models_with_baseline", ALL_MODELS),
        ):
            score_matrix = n * weights @ np.column_stack([pointwise[model] for model in models])
            order = np.argsort(-score_matrix, axis=1, kind="stable")
            ranks = np.empty_like(order)
            ranks[np.arange(n_bootstrap)[:, None], order] = np.arange(1, len(models) + 1)
            for model_index, model in enumerate(models):
                for rank in range(1, len(models) + 1):
                    rank_rows.append(
                        {
                            "data_variant": variant,
                            "scope": scope,
                            "model": model,
                            "rank": rank,
                            "probability": float(np.mean(ranks[:, model_index] == rank)),
                            "mean_rank": float(ranks[:, model_index].mean()),
                            "n_bootstrap": n_bootstrap,
                            "seed": seed + variant_index,
                        }
                    )

        for first_index, model_a in enumerate(ALL_MODELS):
            for model_b in ALL_MODELS[first_index + 1 :]:
                difference = pointwise[model_a] - pointwise[model_b]
                bootstrap_delta = n * (weights @ difference)
                pairwise_rows.append(
                    {
                        "data_variant": variant,
                        "model_a": model_a,
                        "model_b": model_b,
                        "delta_elpd_a_minus_b": float(difference.sum()),
                        "paired_se": float(math.sqrt(n * np.var(difference, ddof=1))),
                        "mean_delta_per_participant": float(difference.mean()),
                        "bootstrap_q025": float(np.quantile(bootstrap_delta, 0.025)),
                        "bootstrap_median": float(np.quantile(bootstrap_delta, 0.5)),
                        "bootstrap_q975": float(np.quantile(bootstrap_delta, 0.975)),
                        "bootstrap_probability_delta_gt_0": float(np.mean(bootstrap_delta > 0.0)),
                        "n_bootstrap": n_bootstrap,
                        "seed": seed + variant_index,
                    }
                )
    return rank_rows, pairwise_rows

# =============================================================================
# 8A-D. D module: nested participant-level setting-specific exposure metric selection
# =============================================================================

D_NESTED_MODEL = "setting_specific_nested"
D_ANCHOR_MODEL = "contacts"
D_SELECTION_TOLERANCE = 1e-10


def _bernoulli_log_score(outcome: float, probability: float) -> float:
    """Return the Bernoulli log predictive density for one held-out participant."""
    probability = float(np.clip(probability, 1e-15, 1.0 - 1e-15))
    return math.log(probability if int(outcome) == 1 else 1.0 - probability)


def _scaled_metric_views(
    data: ExposureData,
    train_indices: np.ndarray,
    evaluation_index: int | np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, ScaleSpec]]:
    """Fit every exposure transformation on training participants only.

    This helper is intentionally fold-local.  The held-out participant never
    contributes to log1p centring/scaling or to the subsequent metric choice.
    The scaling rules are exactly those used by the existing independent
    MAP+Laplace validator: participation is centred only; continuous metrics
    use log1p and are divided by 2 training SDs.
    """
    train_scaled: dict[str, np.ndarray] = {}
    evaluation_scaled: dict[str, np.ndarray] = {}
    specs: dict[str, ScaleSpec] = {}
    for metric in EXPOSURE_MODELS:
        raw_matrix = data.matrices[metric]
        spec = fit_scale_spec(raw_matrix[train_indices], metric)
        _, scaled_train = apply_scale(raw_matrix[train_indices], spec)
        _, scaled_evaluation = apply_scale(raw_matrix[evaluation_index], spec)
        train_scaled[metric] = scaled_train
        evaluation_scaled[metric] = scaled_evaluation
        specs[metric] = spec
    return train_scaled, evaluation_scaled, specs


def _choose_metric_from_inner_elpd(
    scores: dict[str, float],
    *,
    anchor: str,
    tolerance: float,
) -> tuple[str, str, float]:
    """Choose one metric conservatively from inner-LOPO ELPD values.

    Numerical ties within ``tolerance`` are resolved in favour of the Contacts
    anchor, because the data then provide no predictive evidence for replacing
    the global anchor representation.  Remaining ties follow the declared
    EXPOSURE_MODELS order, making the rule deterministic and auditable.
    """
    if set(scores) != set(EXPOSURE_MODELS):
        raise ValueError("D-module score dictionary must contain all exposure metrics")
    if anchor not in scores:
        raise ValueError(f"Unknown D-module anchor metric: {anchor!r}")
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Non-finite inner ELPD in D-module metric selection")

    best_value = max(scores.values())
    tied = [
        metric
        for metric in EXPOSURE_MODELS
        if best_value - scores[metric] <= tolerance
    ]
    chosen = anchor if anchor in tied else tied[0]
    remaining = [metric for metric in EXPOSURE_MODELS if metric != chosen]
    runner_up = max(
        remaining,
        key=lambda metric: (
            scores[metric],
            1 if metric == anchor else 0,
            -EXPOSURE_MODELS.index(metric),
        ),
    )
    return chosen, runner_up, float(scores[chosen] - scores[runner_up])


def run_nested_setting_specific_selection(
    data: ExposureData,
    common_predictions: Sequence[dict[str, object]],
    *,
    bootstrap_draws: int,
    seed: int,
    anchor: str = D_ANCHOR_MODEL,
    selection_tolerance: float = D_SELECTION_TOLERANCE,
    inner_folds: int | None = 5,
) -> dict[str, Any]:
    """Run the post-review D-module with strict nested participant-level LOPO.

    Scientific target
    -----------------
    Test whether allowing the exposure representation to vary by setting
    improves prediction for a completely unseen participant.

    Outer loop
    ----------
    One participant is held out for final evaluation.

    Inner loop
    ----------
    Within the remaining participants, every setting is evaluated with four
    candidate representations.  Eleven columns stay on the Contacts anchor and
    only the target setting is substituted.  Candidate performance is measured
    by nested participant-level cross-validation.  By default the inner loop is
    5-fold stratified CV; setting ``inner_folds=None`` requests explicit fold-wise refitted inner LOPO.

    Final outer prediction
    ----------------------
    The 12 metrics selected using the outer training set are assembled into one
    12-column hybrid, refit on the entire outer training set, and used exactly
    once to predict the untouched outer participant.

    This is therefore valid for evaluating the *selection procedure*.  It is
    deliberately separate from the existing full-data RQ2 local comparisons,
    which describe local support but must not be assembled into a hybrid and
    scored on the same LOO contributions.
    """
    if bootstrap_draws < 100:
        raise ValueError("D-module bootstrap_draws must be at least 100")
    if anchor not in EXPOSURE_MODELS:
        raise ValueError(f"D-module anchor must be one of {EXPOSURE_MODELS}")
    if selection_tolerance < 0 or not np.isfinite(selection_tolerance):
        raise ValueError("D-module selection_tolerance must be finite and non-negative")
    if inner_folds is not None and inner_folds < 2:
        raise ValueError("D-module inner_folds must be >=2 or None for explicit fold-wise refitted inner LOPO")

    from sklearn.model_selection import StratifiedKFold

    n = len(data.participant_ids)
    n_settings = len(data.settings)
    if n < 6:
        raise ValueError("D-module nested LOPO requires at least 6 participants")
    if data.outcomes.shape != (n,):
        raise ValueError("D-module outcome vector is misaligned")
    for metric in EXPOSURE_MODELS:
        if data.matrices[metric].shape != (n, n_settings):
            raise ValueError(f"D-module matrix shape mismatch for {metric}")

    metric_index = {metric: index for index, metric in enumerate(EXPOSURE_MODELS)}
    anchor_index = metric_index[anchor]
    all_indices = np.arange(n, dtype=int)

    outer_predictions: list[dict[str, object]] = []
    outer_selection_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    inner_fit_count = 0
    expected_inner_fit_count = 0
    outer_fit_count = 0

    for outer_index in range(n):
        outer_train = all_indices[all_indices != outer_index]
        if len(np.unique(data.outcomes[outer_train])) < 2:
            raise RuntimeError(
                f"D-module outer fold {outer_index + 1} has only one outcome class"
            )

        # inner_score[setting, metric] accumulates nested inner-CV log scores.
        inner_score = np.zeros((n_settings, len(EXPOSURE_MODELS)), dtype=float)

        if inner_folds is None:
            inner_splits = [
                (outer_train[outer_train != inner_index], np.asarray([inner_index], dtype=int))
                for inner_index in outer_train
            ]
        else:
            outer_outcomes = data.outcomes[outer_train].astype(int)
            class_counts = np.bincount(outer_outcomes, minlength=2)
            n_splits = min(int(inner_folds), int(class_counts.min()))
            if n_splits < 2:
                raise RuntimeError(
                    f"D-module outer fold {outer_index + 1} cannot form stratified inner CV"
                )
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=int(seed + outer_index),
            )
            inner_splits = [
                (outer_train[train_pos], outer_train[validation_pos])
                for train_pos, validation_pos in splitter.split(outer_train, outer_outcomes)
            ]

        expected_inner_fit_count += len(inner_splits) * (
            1 + n_settings * (len(EXPOSURE_MODELS) - 1)
        )

        # One Contacts-anchor fit is shared by all settings for each inner split.
        # Every non-anchor candidate then changes exactly one setting column.
        for inner_train, inner_validation in inner_splits:
            train_outcomes = data.outcomes[inner_train]
            train_scaled, heldout_scaled, _ = _scaled_metric_views(
                data, inner_train, inner_validation
            )

            anchor_train = np.column_stack(
                [np.ones(len(inner_train), dtype=float), train_scaled[anchor]]
            )
            anchor_heldout = np.column_stack(
                [
                    np.ones(len(inner_validation), dtype=float),
                    heldout_scaled[anchor],
                ]
            )
            anchor_fit = fit_map_laplace(anchor_train, train_outcomes)
            anchor_lpd = 0.0
            for local_index, design_row in enumerate(anchor_heldout):
                anchor_probability, _, _ = laplace_predictive_probability(
                    design_row, anchor_fit
                )
                anchor_lpd += _bernoulli_log_score(
                    data.outcomes[inner_validation[local_index]], anchor_probability
                )
            inner_score[:, anchor_index] += anchor_lpd
            inner_fit_count += 1

            for setting_index in range(n_settings):
                design_column = 1 + setting_index
                for metric in EXPOSURE_MODELS:
                    if metric == anchor:
                        continue
                    candidate_train = anchor_train.copy()
                    candidate_heldout = anchor_heldout.copy()
                    candidate_train[:, design_column] = train_scaled[metric][
                        :, setting_index
                    ]
                    candidate_heldout[:, design_column] = heldout_scaled[metric][
                        :, setting_index
                    ]
                    candidate_fit = fit_map_laplace(
                        candidate_train, train_outcomes
                    )
                    candidate_lpd = 0.0
                    for local_index, design_row in enumerate(candidate_heldout):
                        candidate_probability, _, _ = laplace_predictive_probability(
                            design_row, candidate_fit
                        )
                        candidate_lpd += _bernoulli_log_score(
                            data.outcomes[inner_validation[local_index]],
                            candidate_probability,
                        )
                    inner_score[setting_index, metric_index[metric]] += candidate_lpd
                    inner_fit_count += 1

        selected_metrics: list[str] = []
        for setting_index, setting in enumerate(data.settings):
            scores = {
                metric: float(inner_score[setting_index, metric_index[metric]])
                for metric in EXPOSURE_MODELS
            }
            chosen, runner_up, delta = _choose_metric_from_inner_elpd(
                scores,
                anchor=anchor,
                tolerance=selection_tolerance,
            )
            selected_metrics.append(chosen)
            for metric in EXPOSURE_MODELS:
                candidate_rows.append(
                    {
                        "data_variant": data.variant,
                        "heldout_fold": outer_index + 1,
                        "setting": setting,
                        "candidate_metric": metric,
                        "inner_lopo_elpd": scores[metric],
                        "selected_for_outer_fold": int(metric == chosen),
                        "anchor_metric": anchor,
                    }
                )
            outer_selection_rows.append(
                {
                    "data_variant": data.variant,
                    "heldout_fold": outer_index + 1,
                    "setting": setting,
                    "selected_metric": chosen,
                    "runner_up_metric": runner_up,
                    "selected_inner_lopo_elpd": scores[chosen],
                    "runner_up_inner_lopo_elpd": scores[runner_up],
                    "delta_inner_elpd_best_minus_runner_up": delta,
                    "selection_tolerance": selection_tolerance,
                    "tie_rule": "prefer_contacts_anchor_then_declared_metric_order",
                }
            )

        # Refit the selected 12-column hybrid on the complete outer training set.
        outer_train_scaled, outer_heldout_scaled, _ = _scaled_metric_views(
            data, outer_train, outer_index
        )
        final_train = np.ones((len(outer_train), n_settings + 1), dtype=float)
        final_heldout = np.ones(n_settings + 1, dtype=float)
        for setting_index, metric in enumerate(selected_metrics):
            final_train[:, 1 + setting_index] = outer_train_scaled[metric][
                :, setting_index
            ]
            final_heldout[1 + setting_index] = outer_heldout_scaled[metric][
                setting_index
            ]

        final_fit = fit_map_laplace(final_train, data.outcomes[outer_train])
        final_probability, logit_mean, logit_variance = laplace_predictive_probability(
            final_heldout, final_fit
        )
        outcome = int(data.outcomes[outer_index])
        final_lpd = _bernoulli_log_score(outcome, final_probability)
        outer_fit_count += 1
        outer_predictions.append(
            {
                "data_variant": data.variant,
                "model": D_NESTED_MODEL,
                "heldout_fold": outer_index + 1,
                "outcome": outcome,
                "predicted_probability": final_probability,
                "log_predictive_density": final_lpd,
                "brier_contribution": (outcome - final_probability) ** 2,
                "heldout_logit_mean": logit_mean,
                "heldout_logit_variance": logit_variance,
                "n_train": len(outer_train),
                "train_positives": int(data.outcomes[outer_train].sum()),
                "train_negatives": int(
                    len(outer_train) - data.outcomes[outer_train].sum()
                ),
                "optimizer_method": final_fit.optimizer_method,
                "optimizer_reported_success": int(
                    final_fit.optimizer_reported_success
                ),
                "fit_accepted": int(final_fit.accepted),
                "gradient_inf_norm": final_fit.gradient_norm,
                "hessian_min_eigenvalue": final_fit.hessian_min_eigenvalue,
                "hessian_condition_number": final_fit.hessian_condition_number,
                "selected_metric_count_participation": selected_metrics.count(
                    "participation"
                ),
                "selected_metric_count_contacts": selected_metrics.count("contacts"),
                "selected_metric_count_duration": selected_metrics.count(
                    "duration_minutes"
                ),
                "selected_metric_count_pcm": selected_metrics.count(
                    "pcm"
                ),
            }
        )

    prediction_frame = pd.DataFrame(outer_predictions).sort_values("heldout_fold")
    selection_frame = pd.DataFrame(outer_selection_rows).sort_values(
        ["setting", "heldout_fold"]
    )
    candidate_frame = pd.DataFrame(candidate_rows).sort_values(
        ["setting", "heldout_fold", "candidate_metric"]
    )

    frequency_frame = (
        selection_frame.groupby(["data_variant", "setting", "selected_metric"])
        .size()
        .rename("selection_count")
        .reset_index()
    )
    complete_frequency = pd.MultiIndex.from_product(
        [[data.variant], data.settings, EXPOSURE_MODELS],
        names=["data_variant", "setting", "selected_metric"],
    ).to_frame(index=False)
    frequency_frame = complete_frequency.merge(
        frequency_frame,
        on=["data_variant", "setting", "selected_metric"],
        how="left",
    )
    frequency_frame["selection_count"] = (
        frequency_frame["selection_count"].fillna(0).astype(int)
    )
    frequency_frame["selection_proportion"] = (
        frequency_frame["selection_count"] / n
    )
    frequency_frame["most_selected_for_setting"] = frequency_frame.groupby(
        ["data_variant", "setting"]
    )["selection_count"].transform("max").eq(frequency_frame["selection_count"])

    outcomes = prediction_frame["outcome"].to_numpy(dtype=float)
    probabilities = prediction_frame["predicted_probability"].to_numpy(dtype=float)
    log_density = prediction_frame["log_predictive_density"].to_numpy(dtype=float)
    brier = prediction_frame["brier_contribution"].to_numpy(dtype=float)
    nested_metrics = {
        "data_variant": data.variant,
        "model": D_NESTED_MODEL,
        "n": n,
        "positives": int(outcomes.sum()),
        "negatives": int(n - outcomes.sum()),
        "elpd": float(log_density.sum()),
        "elpd_se": float(math.sqrt(n * np.var(log_density, ddof=1))),
        "mean_log_score": float(log_density.mean()),
        "brier_score": float(brier.mean()),
        "auc": binary_auc(outcomes, probabilities),
        "optimizer_reported_failures": int(
            (prediction_frame["optimizer_reported_success"] == 0).sum()
        ),
        "fit_acceptance_failures": int(
            (prediction_frame["fit_accepted"] == 0).sum()
        ),
        "max_gradient_inf_norm": float(
            prediction_frame["gradient_inf_norm"].max()
        ),
        "min_hessian_eigenvalue": float(
            prediction_frame["hessian_min_eigenvalue"].min()
        ),
        "max_hessian_condition_number": float(
            prediction_frame["hessian_condition_number"].max()
        ),
    }

    # Pair the nested predictions with the already-computed common-model outer
    # LOPO predictions.  This guarantees identical held-out participants and the
    # same independent-validator scaling convention for the comparator models.
    common_pointwise: dict[str, np.ndarray] = {}
    paired_rows: list[dict[str, object]] = []
    for metric in EXPOSURE_MODELS:
        rows = _group_predictions(common_predictions, data.variant, metric)
        if len(rows) != n:
            raise ValueError(
                f"D-module expected {n} common predictions for {metric}, got {len(rows)}"
            )
        common_outcomes = np.asarray([row["outcome"] for row in rows], dtype=float)
        if not np.array_equal(common_outcomes, outcomes):
            raise ValueError(f"D-module outcome alignment failed for {metric}")
        common_lpd = np.asarray(
            [row["log_predictive_density"] for row in rows], dtype=float
        )
        common_pointwise[metric] = common_lpd
        difference = log_density - common_lpd
        paired_rows.append(
            {
                "data_variant": data.variant,
                "candidate_model": D_NESTED_MODEL,
                "reference_model": metric,
                "delta_elpd_candidate_minus_reference": float(difference.sum()),
                "paired_dse": float(math.sqrt(n * np.var(difference, ddof=1))),
                "mean_delta_per_participant": float(difference.mean()),
            }
        )

    # Ranking uncertainty is computed only among the four common metrics plus
    # the nested setting-specific procedure; the intercept-only baseline is not
    # part of this specific project question.
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n), size=bootstrap_draws)
    ranking_models = (*EXPOSURE_MODELS, D_NESTED_MODEL)
    pointwise_matrix = np.column_stack(
        [
            common_pointwise[model]
            if model in common_pointwise
            else log_density
            for model in ranking_models
        ]
    )
    score_matrix = n * weights @ pointwise_matrix
    order = np.argsort(-score_matrix, axis=1, kind="stable")
    rank_matrix = np.empty_like(order)
    rank_matrix[np.arange(bootstrap_draws)[:, None], order] = np.arange(
        1, len(ranking_models) + 1
    )
    rank_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(ranking_models):
        rank_rows.append(
            {
                "data_variant": data.variant,
                "model": model,
                "probability_rank_1": float(
                    np.mean(rank_matrix[:, model_index] == 1)
                ),
                "mean_rank": float(rank_matrix[:, model_index].mean()),
                "n_bootstrap": bootstrap_draws,
                "seed": seed,
            }
        )

    if inner_folds is None:
        inner_validation_label = "exact participant-level LOPO inside each outer training set"
    else:
        requested = int(inner_folds)
        inner_validation_label = f"up to {requested}-fold stratified participant CV inside each outer training set"
    checks = {
        "outer_prediction_count": len(prediction_frame) == n,
        "selection_rows_count": len(selection_frame) == n * n_settings,
        "candidate_rows_count": (
            len(candidate_frame) == n * n_settings * len(EXPOSURE_MODELS)
        ),
        "inner_fit_count": inner_fit_count == expected_inner_fit_count,
        "outer_fit_count": outer_fit_count == n,
        "all_outer_probabilities_finite": bool(
            np.isfinite(prediction_frame["predicted_probability"]).all()
        ),
        "all_outer_probabilities_in_unit_interval": bool(
            prediction_frame["predicted_probability"].between(0.0, 1.0, inclusive="neither").all()
        ),
        "all_outer_fits_accepted": bool(
            (prediction_frame["fit_accepted"] == 1).all()
        ),
        "selection_frequency_sums_to_outer_folds": bool(
            frequency_frame.groupby("setting")["selection_count"].sum().eq(n).all()
        ),
    }
    metadata = {
        "status": "pass" if all(checks.values()) else "fail",
        "analysis_label": (
            "D module: post-review nested participant-level setting-specific metric "
            "selection using independent MAP+Laplace approximate robustness validation"
        ),
        "data_variant": data.variant,
        "anchor_metric": anchor,
        "candidate_metrics": list(EXPOSURE_MODELS),
        "selection_unit": "setting within outer training participants",
        "inner_validation": inner_validation_label,
        "inner_folds": inner_folds,
        "outer_validation": "participant-level LOPO",
        "selection_tolerance": selection_tolerance,
        "tie_rule": "prefer Contacts anchor on numerical ties; otherwise declared metric order",
        "final_model_columns": n_settings,
        "inner_fit_count": inner_fit_count,
        "outer_final_fit_count": outer_fit_count,
        "expected_inner_fit_count": expected_inner_fit_count,
        "checks": checks,
    }
    return {
        "predictions": prediction_frame,
        "selections": selection_frame,
        "candidate_inner_elpd": candidate_frame,
        "selection_frequency": frequency_frame,
        "metrics": pd.DataFrame([nested_metrics]),
        "paired_comparisons": pd.DataFrame(paired_rows),
        "rank_probabilities": pd.DataFrame(rank_rows),
        "manifest": metadata,
    }


# =============================================================================
# 8B. Main manuscript figures and appendix figures
# =============================================================================

BLUE = "#356E9F"
LIGHT_BLUE = "#9FC3DF"
ORANGE = "#D97A2B"
TEAL = "#3B8C87"
RED = "#C84C4C"
GRAY = "#737B83"
LIGHT_GRAY = "#E9EDF0"
INK = "#24313A"

MODEL_EN = {
    "participation": "Binary participation",
    "contacts": "Contact count",
    "duration": "Activity duration",
    "pcm": "Person-contact minutes (PCM)",
    "duration_minutes": "Activity duration",
    "intercept_only": "Intercept only",
}
MODEL_EN_HEATMAP = {
    "participation": "Binary\nparticipation",
    "contacts": "Contact\ncount",
    "duration": "Activity\nduration",
    "pcm": "PCM",
}
SCENARIO_EN = {
    "main": "Main analysis",
    "raw_scale": "No-log direct standardization",
    "stronger_prior": "Stronger prior",
    "weaker_prior": "Weaker prior",
    "drop_sparse": "Exclude separated settings",
}
SETTING_EN = {
    "Holiday": "Holiday",
    "Campus": "Campus",
    "Exercise": "Exercise",
    "Hospitality": "Hospitality",
    "Work": "Work",
    "Travel": "Travel",
    "Other": "Other",
    "Research": "Research",
    "Retail": "Retail",
    "Social": "Social",
    "Teaching": "Teaching",
    "Testing": "Testing",
}

def save_figure(fig: plt.Figure, name: str) -> None:
    # Historical source copies contained a few mojibake glyphs in otherwise
    # English figure labels.  Sanitize every rendered text artist before export.
    replacements = {
        "\u5364": "±",
        "\u8796": "Δ",
        "\u5c3e": "β",
        "\u7eac": "γ",
        "\u8133": "×",
    }
    for artist in fig.findobj(match=mpl.text.Text):
        cleaned = artist.get_text()
        for bad, good in replacements.items():
            cleaned = cleaned.replace(bad, good)
        cleaned = re.sub(r"5th.*?5th percentile", "5th–95th percentile", cleaned)
        # No Chinese text is intended inside the publication figures.  Any
        # remaining CJK glyph here is a damaged symbol, not a label.
        cleaned = re.sub(r"[\u4e00-\u9fff]", "", cleaned)
        artist.set_text(cleaned)
    # Keep the three exports explicit so static QA can verify vector and raster
    # deliverables without having to infer extensions from a dynamic loop.
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def metric_from_model(name: str) -> str:
    return name.split("__", 1)[1].replace("participation_household", "participation")


def figure_1_workflow() -> None:
    with open(AUDIT / "audit_summary.json", encoding="utf-8") as f:
        audit = json.load(f)
    n_source_rows = int(audit["dataset"]["source_rows"])
    n_activities = int(audit["dataset"]["reported_activities"])
    n_participants = int(audit["dataset"]["participants"])
    n_settings = len(audit["setting_participant_counts"])
    fig, ax = plt.subplots(figsize=(6.7, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.62, 0.19, 0.22, f"Updated wide data\n{n_source_rows} participant rows\n{n_activities} reported activities", BLUE),
        (0.24, 0.62, 0.23, 0.22, f"Validated {n_participants} × {n_settings} grid\nParticipation / contacts\nDuration / PCM", TEAL),
        (0.50, 0.62, 0.22, 0.22, "Equal-complexity\nBayesian logistic models\nShared shrinkage prior", ORANGE),
        (0.75, 0.62, 0.23, 0.22, "Participant-level\nPSIS-LOO\nExact reloo when needed", BLUE),
        (0.17, 0.14, 0.31, 0.22, "Robustness\nTransform / priors /\nseparated settings", GRAY),
        (0.53, 0.14, 0.30, 0.22, "Household + D module\nSeparate HH fields / nested\nsetting-specific selection", GRAY),
    ]
    for x, y, w, h, txt, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=color, edgecolor="none", alpha=0.95,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, color="white", ha="center", va="center", fontsize=7.0, linespacing=1.28)
    for x1, x2 in [(0.21, 0.24), (0.47, 0.50), (0.72, 0.75)]:
        ax.add_patch(FancyArrowPatch((x1, 0.73), (x2, 0.73), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK))
    ax.add_patch(FancyArrowPatch((0.60, 0.62), (0.34, 0.36), connectionstyle="arc3,rad=0.1", arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK))
    ax.add_patch(FancyArrowPatch((0.64, 0.62), (0.68, 0.36), connectionstyle="arc3,rad=-0.1", arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK))
    ax.text(
        0.5, 0.94,
        "Updated participant-level workflow: validation without re-cleaning,\nmodel comparison, robustness, household analysis and nested selection",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK, linespacing=1.25,
    )
    ax.text(0.5, 0.05, "No activity-row deduplication or Contacts-HHcontacts subtraction is performed", ha="center", fontsize=8.3, color=GRAY)
    save_figure(fig, "fig_1_research_workflow")


def figure_2_data_audit() -> None:
    with open(AUDIT / "audit_summary.json", encoding="utf-8") as f:
        audit = json.load(f)
    setting = pd.DataFrame(audit["setting_participant_counts"])
    outcome_counts = {int(row["outcome"]): int(row["participants"]) for row in audit["outcome_counts"]}
    setting["total_visitors"] = setting["visitors_outcome_0"] + setting["visitors_outcome_1"]
    setting = setting.sort_values("total_visitors")

    fig = plt.figure(figsize=(6.7, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.45], width_ratios=[0.9, 1.1], hspace=0.42, wspace=0.62)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    vals = [outcome_counts[1], outcome_counts[0]]
    ax1.bar([0, 1], vals, color=[RED, BLUE], width=0.58)
    ax1.set_xticks([0, 1], ["Positive", "Negative"])
    ax1.set_ylabel("Participants")
    ax1.set_ylim(0, max(vals) + 6)
    for x, v in enumerate(vals):
        ax1.text(x, v + 0.8, str(v), ha="center", fontweight="bold")
    ax1.set_title(f"Outcome distribution (n = {sum(vals)})")
    add_panel_label(ax1, "a")

    audit_vals = [
        int(audit["dataset"]["source_rows"]),
        int(audit["dataset"]["participant_setting_cells"]),
        int(audit["dataset"]["reported_activities"]),
        int(audit["dataset"]["all_zero_positive_count"]),
    ]
    labels = ["Participant rows", "Participant-setting cells", "Reported activities", "All-zero positive rows"]
    ax2.barh(range(4), audit_vals, color=[BLUE, TEAL, ORANGE, RED], height=0.58)
    ax2.set_yticks(range(4), labels)
    ax2.tick_params(axis="y", labelsize=7.2, pad=2)
    ax2.set_xlabel("Count")
    for y, v in enumerate(audit_vals):
        ax2.text(v + max(audit_vals) * 0.02, y, str(v), va="center", fontweight="bold")
    ax2.set_title("Validation-only data contract")
    add_panel_label(ax2, "b")

    y = np.arange(len(setting))
    neg = setting["visitors_outcome_0"].to_numpy()
    pos = setting["visitors_outcome_1"].to_numpy()
    ax3.barh(y, neg, color=BLUE, label="Negative participants")
    ax3.barh(y, pos, left=neg, color=RED, label="Positive participants")
    ax3.set_yticks(y, [SETTING_EN[x] for x in setting["setting_group"]])
    ax3.set_xlabel("Participants reporting each setting")
    ax3.set_title("Setting coverage and complete separation")
    ax3.legend(loc="lower right", ncol=2)
    for idx, row in setting.reset_index(drop=True).iterrows():
        if row["separation_flag"] == "complete":
            ax3.text(row["total_visitors"] + 0.6, idx, "Complete separation", va="center", color=RED, fontsize=7.5, fontweight="bold")
    ax3.set_xlim(0, max(setting["total_visitors"]) + 7)
    add_panel_label(ax3, "c")
    fig.suptitle("Updated sample structure and validation checks", fontsize=11.5, fontweight="bold", y=1.00, color=INK)
    save_figure(fig, "fig_2_data_audit")


def figure_3_main_comparison() -> None:
    comp = pd.read_csv(RESULTS / "tables" / "table_2_main_model_comparison.csv")
    ranks = pd.read_csv(RESULTS / "tables" / "main_rank_probabilities.csv")
    comp["metric_key"] = comp["model"].map(metric_from_model)
    ranks["metric_key"] = ranks["model"].map(metric_from_model)
    order = ["contacts", "duration", "participation", "pcm"]
    comp = comp.set_index("metric_key").loc[order].reset_index()
    ranks = ranks.set_index("metric_key").loc[order].reset_index()
    y = np.arange(len(order))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 3.05), gridspec_kw={"width_ratios": [1.35, 0.8], "wspace": 0.35})
    ax1.errorbar(comp["elpd"], y, xerr=comp["se"], fmt="o", color=BLUE, ecolor=LIGHT_BLUE, elinewidth=2, capsize=3, markersize=5)
    ax1.set_yticks(y, [MODEL_EN[x] for x in order])
    ax1.set_xlabel("PSIS-LOO ELPD (estimate ± 1 SE)")
    ax1.set_title("Participant-level predictive performance")
    ax1.axvline(comp["elpd"].max(), color=GRAY, ls=":", lw=0.9)
    for i, row in comp.iterrows():
        txt = "Reference" if i == 0 else f"Δ={row['elpd_diff']:.2f}"
        ax1.text(row["elpd"] + 0.12, y[i] + 0.15, txt, fontsize=7, color=INK)
    add_panel_label(ax1, "a")

    probs = ranks["probability_rank_1"].to_numpy()
    ax2.barh(y, probs, color=[ORANGE, TEAL, LIGHT_BLUE, GRAY], height=0.6)
    ax2.set_yticks(y, [MODEL_EN[x] for x in order])
    ax2.set_xlim(0, 0.6)
    ax2.set_xlabel("Bayesian bootstrap P(rank = 1)")
    ax2.set_title("Ranking uncertainty")
    for yi, p in zip(y, probs):
        ax2.text(p + 0.015, yi, f"{p:.3f}", va="center", fontsize=7.5)
    add_panel_label(ax2, "b")
    fig.suptitle("Common-metric predictive comparison under the updated dataset", fontsize=11.0, fontweight="bold", y=1.02, color=INK)
    save_figure(fig, "fig_3_main_model_comparison")


def figure_4_setting_forest() -> None:
    coef = pd.read_csv(RESULTS / "tables" / "table_3_setting_coefficients.csv")
    sub = coef[(coef["model"] == "main__contacts") & coef["parameter"].str.startswith("beta[")].copy()
    sub["setting"] = sub["parameter"].str.removeprefix("beta[").str.removesuffix("]")
    sub = sub.sort_values("mean")
    y = np.arange(len(sub))
    colors = [RED if (r.hdi95_lb > 0 or r.hdi95_ub < 0) else BLUE for r in sub.itertuples()]

    fig, ax = plt.subplots(figsize=(6.7, 4.7))
    for yi, (_, r), c in zip(y, sub.iterrows(), colors):
        ax.plot([r["hdi95_lb"], r["hdi95_ub"]], [yi, yi], color=LIGHT_BLUE, lw=2)
        ax.plot(r["mean"], yi, "o", color=c, ms=5)
        ax.text(r["hdi95_ub"] + 0.05, yi, f"P(β>0)={r['posterior_probability_positive']:.2f}", va="center", fontsize=6.8, color=GRAY)
    ax.axvline(0, color=INK, ls="--", lw=0.9)
    ax.set_yticks(y, [SETTING_EN[x] for x in sub["setting"]])
    ax.set_xlabel("Standardized log-odds coefficient (posterior mean and 95% HDI)")
    ax.set_title("Twelve setting associations in the contact-count model")
    ax.set_xlim(min(sub["hdi95_lb"]) - 0.15, max(sub["hdi95_ub"]) + 0.62)
    ax.text(0.01, -0.13, "All 95% HDIs cross 0; ordering shows posterior direction only, not causal effects or significance.", transform=ax.transAxes, color=GRAY, fontsize=7.5)
    save_figure(fig, "fig_4_setting_forest_contacts")


def figure_5_sensitivity() -> None:
    sens = pd.read_csv(RESULTS / "tables" / "table_5_sensitivity.csv")
    order_s = ["main", "raw_scale", "stronger_prior", "weaker_prior", "drop_sparse"]
    order_m = ["participation", "contacts", "duration", "pcm"]
    order_s = [scenario for scenario in order_s if scenario in set(sens["scenario"])]
    sens["metric_key"] = sens["metric"]
    matrix = sens.pivot(index="scenario", columns="metric_key", values="delta_elpd_to_scenario_best").loc[order_s, order_m]
    rankp = pd.read_csv(RESULTS / "tables" / "sensitivity_rank_probabilities.csv")
    rankp["metric_key"] = rankp["model"].map(metric_from_model)
    pmat = rankp.pivot(index="scenario", columns="metric_key", values="probability_rank_1").loc[order_s, order_m]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 4.0), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.28})
    vmax = 0
    vmin = min(-3.1, float(matrix.min().min()))
    im = ax1.imshow(matrix.to_numpy(), cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_xticks(range(4), [MODEL_EN_HEATMAP[x] for x in order_m])
    ax1.set_yticks(range(len(order_s)), [SCENARIO_EN[x] for x in order_s])
    ax1.set_title("Within-scenario ΔELPD\n(relative to point-estimate best)", fontsize=9.2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iloc[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < -1.2 else INK, fontsize=7.5, fontweight="bold" if abs(v) < 1e-8 else "normal")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("ΔELPD")
    add_panel_label(ax1, "a")

    im2 = ax2.imshow(pmat.to_numpy(), cmap="Oranges", vmin=0, vmax=0.65, aspect="auto")
    ax2.set_xticks(range(4), [MODEL_EN_HEATMAP[x] for x in order_m])
    ax2.set_yticks(range(len(order_s)), [])
    ax2.set_title("Bayesian bootstrap P(rank = 1)")
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            v = pmat.iloc[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.42 else INK, fontsize=7.5)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("P(rank = 1)")
    add_panel_label(ax2, "b")
    fig.suptitle("Analysis-choice perturbations alter point rankings: no stable winner across scenarios", fontsize=10.8, fontweight="bold", y=1.02, color=INK)
    save_figure(fig, "fig_5_sensitivity_matrix")


def figure_6_household() -> None:
    hh = pd.read_csv(RESULTS / "tables" / "table_6_household_extension.csv")
    coef = pd.read_csv(RESULTS / "tables" / "table_3_setting_coefficients.csv")
    gamma = coef[(coef["model"] == "main__participation_household") & (coef["parameter"] == "gamma_household")].iloc[0]
    base = hh[hh["model"] == "main__participation"].iloc[0]
    ext = hh[hh["model"] == "main__participation_household"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(6.7, 3.05), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    ax.errorbar([base["elpd"], ext["elpd"]], [1, 0], xerr=[base["se"], ext["se"]], fmt="o", color=BLUE, ecolor=LIGHT_BLUE, capsize=3)
    ax.set_yticks([1, 0], ["Participation model", "+ household term"])
    ax.set_xlabel("ELPD (±1 SE)")
    ax.set_title("Incremental prediction")
    ax.text(0.5, 0.52, f"ΔELPD={ext['elpd'] - base['elpd']:.2f}", transform=ax.transAxes, ha="center", fontsize=7.5, color=GRAY)
    add_panel_label(ax, "a")

    ax = axes[1]
    # Frozen 3.0.0 coefficient tables used legacy ``odds_ratio_*`` column names
    # for exp(E[coefficient]) and exponentiated HDI endpoints.  Reporting-only
    # regeneration accepts both schemas but labels the quantity correctly.
    exp_mean_col = (
        "exp_coefficient_posterior_mean"
        if "exp_coefficient_posterior_mean" in gamma.index
        else "odds_ratio_mean"
    )
    exp_lb_col = (
        "exp_coefficient_hdi95_lb"
        if "exp_coefficient_hdi95_lb" in gamma.index
        else "odds_ratio_hdi95_lb"
    )
    exp_ub_col = (
        "exp_coefficient_hdi95_ub"
        if "exp_coefficient_hdi95_ub" in gamma.index
        else "odds_ratio_hdi95_ub"
    )
    ax.plot([gamma[exp_lb_col], gamma[exp_ub_col]], [0, 0], color=LIGHT_BLUE, lw=3)
    ax.plot(gamma[exp_mean_col], 0, "o", color=ORANGE, ms=6)
    ax.axvline(1, color=INK, ls="--", lw=0.9)
    ax.set_yticks([])
    ax.set_xlabel("Exponentiated household coefficient\n(95% interval from coefficient HDI)")
    ax.set_title("Household-term posterior")
    ax.text(
        0.5,
        0.93,
        f"exp(E[γ])={gamma[exp_mean_col]:.2f}; P(γ>0)={gamma['posterior_probability_positive']:.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.0,
        color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
    )
    ax.set_xlim(0, max(2.1, gamma[exp_ub_col] + 0.15))
    add_panel_label(ax, "b")

    ax = axes[2]
    metrics = ["Brier", "AUC"]
    base_vals = [base["brier"], base["roc_auc"]]
    ext_vals = [ext["brier"], ext["roc_auc"]]
    x = np.arange(2)
    ax.bar(x - 0.17, base_vals, 0.34, label="Participation model", color=GRAY)
    ax.bar(x + 0.17, ext_vals, 0.34, label="+ household term", color=TEAL)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 0.72)
    ax.set_title("Auxiliary predictive metrics")
    ax.legend(fontsize=6.5, loc="upper left")
    add_panel_label(ax, "c")
    fig.subplots_adjust(top=0.76)
    fig.suptitle("Binary household-occurrence term shows no stable incremental predictive gain", fontsize=10.5, fontweight="bold", y=0.98, color=INK)
    save_figure(fig, "fig_6_household_extension")


def figure_7_predictive_checks() -> None:
    ppc = pd.read_csv(RESULTS / "tables" / "prior_posterior_predictive_counts.csv")
    ppc["metric_key"] = ppc["metric"]
    order = ["participation", "contacts", "duration", "pcm"]
    ppc = ppc.set_index("metric_key").loc[order].reset_index()
    observed_values = ppc["observed_positive"].astype(int).unique()
    if len(observed_values) != 1:
        raise ValueError("Predictive-check table has inconsistent observed counts")
    observed = int(observed_values[0])
    max_count = int(
        max(
            ppc["prior_count_q95"].max(),
            ppc["posterior_count_q95"].max(),
            observed,
        )
    )
    y = np.arange(4)[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 2.8), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, prefix, title, color in [(ax1, "prior", "Prior predictive", GRAY), (ax2, "posterior", "Posterior predictive", BLUE)]:
        lo = ppc[f"{prefix}_count_q05"].to_numpy()
        med = ppc[f"{prefix}_count_median"].to_numpy()
        hi = ppc[f"{prefix}_count_q95"].to_numpy()
        ax.errorbar(med, y, xerr=[med - lo, hi - med], fmt="o", color=color, ecolor=LIGHT_BLUE if prefix == "posterior" else "#B8B1AE", capsize=3, lw=2)
        ax.axvline(observed, color=RED, ls="--", lw=1.1, label=f"Observed = {observed}")
        ax.set_title(title)
        ax.set_xlabel("Predicted positive participants\n(5th–95th percentile)")
        ax.set_xlim(-1, max_count + 5)
        ax.legend(fontsize=7, loc="lower right")
    ax1.set_yticks(y, [MODEL_EN[x] for x in order])
    fig.suptitle("Priors have broad support; posterior predictions cover the observed positive count", fontsize=10.8, fontweight="bold", y=1.02, color=INK)
    save_figure(fig, "fig_7_prior_posterior_predictive")


def figure_a1_independent_validation() -> None:
    metrics = pd.read_csv(INDEPENDENT / "model_metrics.csv")
    metrics = metrics[metrics["model"] != "intercept_only"].copy()
    order = ["contacts", "duration_minutes", "pcm", "participation"]
    sub = metrics[metrics["data_variant"] == "updated_wide"].set_index("model").loc[order].reset_index()
    y = np.arange(4)[::-1]
    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    ax.errorbar(sub["elpd"], y, xerr=sub["elpd_se"], fmt="o", color=TEAL, ecolor="#9CCBC7", capsize=3)
    ax.set_yticks(y, [MODEL_EN[x] for x in order])
    ax.set_xlabel("Approximate LOPO ELPD (±1 SE)")
    ax.set_title("Separate MAP+Laplace robustness validation on the same participant-level data")
    save_figure(fig, "fig_a1_independent_validation")


def write_tables() -> None:
    """Write English thesis summary tables for the updated participant-level dataset."""
    with open(AUDIT / "audit_summary.json", encoding="utf-8") as handle:
        audit = json.load(handle)
    outcome_counts = {
        int(row["outcome"]): int(row["participants"])
        for row in audit["outcome_counts"]
    }
    sample = pd.DataFrame(
        [
            ["Source participant rows", audit["dataset"]["source_rows"]],
            ["Participant × setting cells", audit["dataset"]["participant_setting_cells"]],
            ["Participants", audit["dataset"]["participants"]],
            ["Positive participants", outcome_counts[1]],
            ["Negative participants", outcome_counts[0]],
            ["Setting categories", len(audit["setting_participant_counts"])],
            ["Reported activities summed from Activities fields", audit["dataset"]["reported_activities"]],
            ["All-zero positive records retained by external project confirmation", audit["dataset"]["all_zero_positive_count"]],
            ["Participation/Activities mismatch cells", audit["dataset"]["participation_activity_mismatch_count"]],
            ["Structural-zero violations where Participation=0", audit["dataset"]["structural_zero_violation_count"]],
            ["Cells with HHcontacts > Contacts", audit["dataset"]["hhcontacts_gt_contacts_cell_count"]],
            ["Data cleaning performed", "No"],
        ],
        columns=["Statistic", "Value"],
    )
    sample.to_csv(TAB / "table_sample_audit_en.csv", index=False, encoding="utf-8-sig")

    setting = pd.DataFrame(audit["setting_participant_counts"])
    setting["Setting"] = setting["setting_group"].map(SETTING_EN)
    setting["Negative participants"] = setting["visitors_outcome_0"].astype(int)
    setting["Positive participants"] = setting["visitors_outcome_1"].astype(int)
    setting["Complete separation"] = setting["separation_flag"].map(
        {"complete": "Yes", "near": "Near", "none": "No"}
    )
    setting[["Setting", "Negative participants", "Positive participants", "Complete separation"]].to_csv(
        TAB / "table_setting_audit_en.csv", index=False, encoding="utf-8-sig"
    )

    desc = pd.read_csv(AUDIT / "participant_descriptives_by_outcome.csv")
    labels = {
        "n_activities": "Reported activities",
        "n_setting_groups": "Number of settings represented",
        "total_contacts": "Total contact count",
        "total_duration_minutes": "Total duration (minutes)",
        "total_pcm": "Total person-contact minutes (PCM)",
        "total_household_contacts": "Total HHcontacts",
        "total_household_pcm": "Total household PCM",
    }
    descriptive_rows = []
    for variable, label in labels.items():
        values = desc[desc["level_0"] == variable].set_index("level_1")

        def median_iqr(column: str) -> str:
            return (
                f"{values.loc['50%', column]:.1f} "
                f"[{values.loc['25%', column]:.1f}, {values.loc['75%', column]:.1f}]"
            )

        descriptive_rows.append([label, median_iqr("0"), median_iqr("1")])
    pd.DataFrame(
        descriptive_rows,
        columns=["Participant-level metric", "Negative: median [IQR]", "Positive: median [IQR]"],
    ).to_csv(
        TAB / "table_participant_descriptives_en.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison = pd.read_csv(RESULTS / "tables" / "table_2_main_model_comparison.csv")
    comparison["Representation"] = comparison["metric"].map(MODEL_EN)
    main_table = comparison[
        ["Representation", "elpd", "se", "elpd_diff", "dse", "brier", "roc_auc", "high_k_count_initial", "exact_refit_count"]
    ].copy()
    main_table.columns = ["Representation", "ELPD", "SE", "ΔELPD", "Difference SE", "Brier", "AUC", "Initial high-k count", "Exact reloo count"]
    main_table.to_csv(TAB / "table_main_comparison_en.csv", index=False, encoding="utf-8-sig")

    diagnostics = pd.read_csv(RESULTS / "tables" / "model_diagnostics.csv")
    full_diagnostics = diagnostics[diagnostics["fit_role"].fillna("full_data") == "full_data"]
    diagnostics_table = pd.DataFrame(
        [
            ["Full-data model count", len(full_diagnostics)],
            ["Exact reloo refit count", int((diagnostics["fit_role"] == "exact_reloo").sum())],
            ["Total divergences", int(diagnostics["divergences"].sum())],
            ["Maximum R-hat", diagnostics["max_rhat"].max()],
            ["Minimum bulk ESS", diagnostics["min_ess_bulk"].min()],
            ["Minimum tail ESS", diagnostics["min_ess_tail"].min()],
            ["Minimum BFMI", diagnostics["min_bfmi"].min()],
            ["Total maximum-tree-depth hits", int(diagnostics["reached_max_treedepth_count"].sum())],
        ],
        columns=["Diagnostic", "Value"],
    )
    diagnostics_table.to_csv(TAB / "table_diagnostics_summary_en.csv", index=False, encoding="utf-8-sig")

    sensitivity = pd.read_csv(RESULTS / "tables" / "table_5_sensitivity.csv")
    sensitivity["Scenario"] = sensitivity["scenario"].map(SCENARIO_EN)
    sensitivity["Representation"] = sensitivity["metric"].map(MODEL_EN)
    sensitivity[
        ["Scenario", "Representation", "elpd", "se", "delta_elpd_to_scenario_best", "rank_within_scenario", "high_k_count_initial", "exact_refit_count", "brier", "roc_auc"]
    ].rename(
        columns={
            "elpd": "ELPD",
            "se": "SE",
            "delta_elpd_to_scenario_best": "Within-scenario ΔELPD",
            "rank_within_scenario": "Within-scenario rank",
            "high_k_count_initial": "Initial high-k count",
            "exact_refit_count": "Exact reloo count",
            "brier": "Brier",
            "roc_auc": "AUC",
        }
    ).to_csv(TAB / "table_sensitivity_en.csv", index=False, encoding="utf-8-sig")

    household = pd.read_csv(RESULTS / "tables" / "table_6_household_extension.csv")
    household["Model"] = household["model"].map(
        {
            "main__participation": "Participation model",
            "main__participation_household": "Participation model + HHcontacts occurrence term",
        }
    )
    household[["Model", "elpd", "se", "elpd_diff", "dse", "brier", "roc_auc"]].rename(
        columns={"elpd": "ELPD", "se": "SE", "elpd_diff": "ΔELPD", "dse": "Difference SE", "brier": "Brier", "roc_auc": "AUC"}
    ).to_csv(TAB / "table_household_en.csv", index=False, encoding="utf-8-sig")

    independent = pd.read_csv(INDEPENDENT / "model_metrics.csv")
    independent["Data version"] = independent["data_variant"].map(
        {"updated_wide": "Supervisor-updated participant-level wide data"}
    )
    independent["Model"] = independent["model"].map(MODEL_EN)
    independent[["Data version", "Model", "elpd", "elpd_se", "brier_score", "auc"]].rename(
        columns={"elpd": "Approximate LOPO ELPD", "elpd_se": "SE", "brier_score": "Brier", "auc": "AUC"}
    ).to_csv(TAB / "table_independent_validation_en.csv", index=False, encoding="utf-8-sig")

    model_labels = {
        "main__participation": "Binary participation model",
        "main__contacts": "Contact-count model",
        "main__duration": "Activity-duration model",
        "main__pcm": "Person-contact-minutes (PCM) model",
        "main__participation_household": "Binary participation model + HHcontacts occurrence term",
    }
    coefficient = pd.read_csv(RESULTS / "tables" / "table_3_setting_coefficients.csv")
    coefficient = coefficient[coefficient["model"].isin(model_labels)].copy()
    # Compatibility for frozen 3.0.0 reporting tables: the legacy
    # ``odds_ratio_*`` fields actually stored exp(E[coefficient]) and
    # exponentiated coefficient-HDI endpoints. Copy them into accurately named
    # reporting aliases without changing any numerical value.
    legacy_exp_aliases = {
        "odds_ratio_mean": "exp_coefficient_posterior_mean",
        "odds_ratio_hdi95_lb": "exp_coefficient_hdi95_lb",
        "odds_ratio_hdi95_ub": "exp_coefficient_hdi95_ub",
    }
    for legacy_name, reporting_name in legacy_exp_aliases.items():
        if reporting_name not in coefficient.columns and legacy_name in coefficient.columns:
            coefficient[reporting_name] = coefficient[legacy_name]

    def parameter_label(value: str) -> str:
        if value == "alpha":
            return "Intercept α"
        if value == "tau":
            return "Global shrinkage scale τ"
        if value == "gamma_household":
            return "HHcontacts occurrence coefficient γ"
        if value.startswith("beta[") and value.endswith("]"):
            setting_name = value.removeprefix("beta[").removesuffix("]")
            return f"Setting coefficient β[{SETTING_EN[setting_name]}]"
        raise ValueError(f"Missing English parameter label: {value}")

    coefficient["Model"] = coefficient["model"].map(model_labels)
    coefficient["Scenario"] = coefficient["variant"].map(SCENARIO_EN).fillna(coefficient["variant"])
    coefficient["Representation"] = coefficient["metric"].map(MODEL_EN).fillna(coefficient["metric"])
    coefficient["Parameter"] = coefficient["parameter"].map(parameter_label)
    selected_columns = [
        "Model", "Scenario", "Representation", "Parameter", "mean", "sd", "hdi95_lb", "hdi95_ub",
        "ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd",
        "posterior_probability_positive", "exp_coefficient_posterior_mean", "exp_coefficient_hdi95_lb", "exp_coefficient_hdi95_ub",
    ]
    coefficient[selected_columns].rename(
        columns={
            "mean": "Posterior mean",
            "sd": "Posterior SD",
            "hdi95_lb": "95% HDI lower bound",
            "hdi95_ub": "95% HDI upper bound",
            "ess_bulk": "Bulk ESS",
            "ess_tail": "Tail ESS",
            "r_hat": "R-hat",
            "mcse_mean": "Mean MCSE",
            "mcse_sd": "SD MCSE",
            "posterior_probability_positive": "P(parameter > 0)",
            "exp_coefficient_posterior_mean": "exp(posterior mean coefficient)",
            "exp_coefficient_hdi95_lb": "exp(95% HDI lower coefficient bound)",
            "exp_coefficient_hdi95_ub": "exp(95% HDI upper coefficient bound)",
        }
    ).to_csv(TAB / "table_all_coefficients.csv", index=False, encoding="utf-8-sig")

# =============================================================================
# 9. Post-review continuous household-contact extension, independent validation, and unified delivery orchestration
# =============================================================================

CONTINUOUS_HOUSEHOLD_DEFINITION = (
    "z(log1p(participant total Household contacts summed across all activity rows)); "
    "one coefficient added to the binary-participation baseline"
)

PUBLIC_DIAGNOSTIC_COLUMNS = (
    "model",
    "variant",
    "metric",
    "include_household",
    "household_mode",
    "draws",
    "tune",
    "chains",
    "divergences",
    "max_tree_depth",
    "reached_max_treedepth_count",
    "max_rhat",
    "min_ess_bulk",
    "min_ess_tail",
    "min_bfmi",
    "fit_role",
    "parent_model",
    "heldout_fold",
)


def _stage(number: int, total: int, label: str) -> None:
    print(f"\n[{number}/{total}] {label}", flush=True)


def _diagnostic_gate(frame: pd.DataFrame, *, full: bool) -> dict[str, Any]:
    if not full:
        return {"status": "not_evaluated", "rule": "Quick chains are not inferential."}
    checks = {
        "zero_divergences": bool((frame["divergences"] == 0).all()),
        "zero_max_treedepth_hits": bool(
            (frame["reached_max_treedepth_count"] == 0).all()
        ),
        "max_rhat_le_1_01": bool(
            frame["max_rhat"].notna().all() and (frame["max_rhat"] <= 1.01).all()
        ),
        "min_ess_bulk_ge_400": bool(
            frame["min_ess_bulk"].notna().all()
            and (frame["min_ess_bulk"] >= 400).all()
        ),
        "min_ess_tail_ge_400": bool(
            frame["min_ess_tail"].notna().all()
            and (frame["min_ess_tail"] >= 400).all()
        ),
        "min_bfmi_gt_0_3": bool(
            frame["min_bfmi"].notna().all() and (frame["min_bfmi"] > 0.3).all()
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "rule": "All full-data and exact-reloo fits must pass the stated thresholds.",
    }


def deidentify_public_frames(
    prediction_frame: pd.DataFrame, diagnostic_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove direct identifiers and reject identifiers leaked into strings."""
    if "participant_id" not in prediction_frame:
        raise ValueError("Expected participant_id before public-output de-identification")
    source_ids = {str(value) for value in prediction_frame["participant_id"].unique()}
    public_predictions = prediction_frame.drop(columns=["participant_id"]).copy()
    available = [column for column in PUBLIC_DIAGNOSTIC_COLUMNS if column in diagnostic_frame]
    public_diagnostics = diagnostic_frame.loc[:, available].copy()
    for frame_name, frame in (
        ("predictions", public_predictions),
        ("diagnostics", public_diagnostics),
    ):
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            for value in frame[column].dropna().astype(str):
                if value in source_ids or any(value.endswith(f"_{item}") for item in source_ids):
                    raise ValueError(
                        f"Source participant identifier leaked into public {frame_name}.{column}"
                    )
    return public_predictions, public_diagnostics


def run_continuous_household_stage(
    data_path: Path,
    output_dir: Path,
    config: RunConfig,
    *,
    strict_known_data: bool,
) -> dict[str, Any]:
    """Fit the post-review continuous household-contact auxiliary comparison."""
    paths = ensure_output_tree(output_dir)
    started = datetime.now(timezone.utc)
    clean = load_and_validate(data_path, strict_known_data=strict_known_data)
    designs, _ = build_designs(clean.frame, variant="main")
    design = designs["participation"]

    h = design.household_contact_intensity_z
    augmented = np.column_stack([np.ones(len(h)), design.X, h])
    household_totals = (
        clean.frame.groupby("participant_id")["household_contacts"]
        .sum()
        .reindex(design.participant_ids)
        .to_numpy(dtype=float)
    )
    correlations = np.corrcoef(np.column_stack([design.X, h]), rowvar=False)
    identifiability = {
        "definition": CONTINUOUS_HOUSEHOLD_DEFINITION,
        "standard_deviation_definition": "population SD, ddof=0",
        "participants": int(len(h)),
        "nonzero_participants": int(np.sum(household_totals > 0)),
        "zero_participants": int(np.sum(household_totals == 0)),
        "unique_raw_totals": int(len(np.unique(household_totals))),
        "raw_total_min": float(household_totals.min()),
        "raw_total_median": float(np.median(household_totals)),
        "raw_total_max": float(household_totals.max()),
        "log1p_mean": design.household_contact_log1p_mean,
        "log1p_sd_ddof0": design.household_contact_log1p_scale,
        "augmented_matrix_columns": int(augmented.shape[1]),
        "augmented_matrix_rank": int(np.linalg.matrix_rank(augmented)),
        "condition_number": float(np.linalg.cond(augmented)),
        "max_abs_correlation_with_participation_columns": float(
            np.max(np.abs(correlations[-1, :-1]))
        ),
        "identifiable": bool(np.linalg.matrix_rank(augmented) == augmented.shape[1]),
    }
    if not identifiability["identifiable"]:
        raise ValueError("Continuous household predictor is not identifiable")

    settings = _settings(config)
    exact_reloo = config.exact_reloo if config.mode == "full" else False
    model_specs = [
        ("main__participation", False, "binary"),
        ("aux__participation_household_continuous", True, "continuous"),
    ]
    evaluations: dict[str, Evaluation] = {}
    diagnostics: list[dict[str, Any]] = []
    coefficients: list[pd.DataFrame] = []
    for name, include_household, household_mode in model_specs:
        seed = config.derived_seed(name)
        fit = fit_model(
            design,
            name=name,
            prior_scale=0.5,
            settings=settings,
            seed=seed,
            include_household=include_household,
            household_mode=household_mode,
            sample_predictive=True,
        )
        evaluation = evaluate_fit(
            fit,
            exact_reloo=exact_reloo,
            reloo_seed=seed + 100_000,
        )
        row = diagnostic_row(fit)
        row.update({"fit_role": "full_data", "parent_model": None, "heldout_fold": None})
        diagnostics.append(row)
        diagnostics.extend(evaluation.reloo_diagnostics)
        coefficients.append(coefficient_table(fit))
        evaluations[name] = evaluation
        if config.save_idata:
            _save_idata(fit, paths["models"] / f"{name}.nc")
        del fit
        gc.collect()

    baseline_name, continuous_name = model_specs[0][0], model_specs[1][0]
    pointwise_delta = (
        evaluations[continuous_name].loo.elpd_i.to_numpy().astype(float)
        - evaluations[baseline_name].loo.elpd_i.to_numpy().astype(float)
    )
    delta = float(pointwise_delta.sum())
    dse = float(np.sqrt(len(pointwise_delta) * np.var(pointwise_delta, ddof=0)))
    comparison_rows: list[dict[str, Any]] = []
    for name, role in (
        (baseline_name, "baseline"),
        (continuous_name, "continuous_household"),
    ):
        metric = evaluations[name].metrics.copy()
        metric["role"] = role
        metric["delta_elpd_vs_baseline"] = 0.0 if name == baseline_name else delta
        metric["dse_vs_baseline"] = 0.0 if name == baseline_name else dse
        comparison_rows.append(metric)
    comparison = pd.DataFrame(comparison_rows)

    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    effect = coefficient_frame.loc[
        (coefficient_frame["model"] == continuous_name)
        & (coefficient_frame["parameter"] == "gamma_household")
    ].copy()
    if len(effect) != 1:
        raise ValueError("Expected exactly one continuous household coefficient")
    effect.insert(3, "definition", CONTINUOUS_HOUSEHOLD_DEFINITION)
    effect["interpretation"] = (
        "exp(coefficient posterior mean) per 1 SD increase; interval is obtained "
        "by exponentiating coefficient 95% HDI endpoints"
    )

    prediction_frame = pd.concat(
        [evaluation.predictions for evaluation in evaluations.values()], ignore_index=True
    )
    diagnostic_frame = pd.DataFrame(diagnostics)
    pareto_summary = (
        prediction_frame.groupby("model", as_index=False)
        .agg(
            max_pareto_k_initial=("pareto_k", "max"),
            pareto_good_k=("pareto_good_k", "first"),
            high_k_count_initial=(
                "pareto_k",
                lambda value: int(
                    np.sum(value > prediction_frame.loc[value.index, "pareto_good_k"])
                ),
            ),
            exact_refit_count=("exact_refit", "sum"),
        )
    )
    public_predictions, public_diagnostics = deidentify_public_frames(
        prediction_frame, diagnostic_frame
    )
    write_csv(comparison, paths["tables"] / "continuous_household_model_comparison.csv")
    write_csv(effect, paths["tables"] / "continuous_household_effect.csv")
    write_csv(coefficient_frame, paths["tables"] / "continuous_household_all_coefficients.csv")
    write_csv(public_predictions, paths["tables"] / "continuous_household_loo_predictions.csv")
    write_csv(public_diagnostics, paths["tables"] / "continuous_household_diagnostics.csv")
    write_csv(pareto_summary, paths["tables"] / "continuous_household_pareto_summary.csv")
    write_json(
        paths["root"] / "validation.json",
        {"data_contract": clean.report, "continuous_household": identifiability},
    )

    loo_checks = {
        name: bool(
            evaluation.metrics["high_k_count_initial"]
            == evaluation.metrics["exact_refit_count"]
        )
        for name, evaluation in evaluations.items()
    }
    manifest = {
        "status": "complete",
        "preliminary": config.mode != "full",
        "analysis": "post-review exploratory continuous household-contact intensity",
        "definition": CONTINUOUS_HOUSEHOLD_DEFINITION,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": clean.sha256,
        "script_version": SCRIPT_VERSION,
        "config": config.to_dict(),
        "identifiability": identifiability,
        "diagnostic_gate": _diagnostic_gate(
            public_diagnostics, full=config.mode == "full"
        ),
        "loo_reliability_gate": {
            "status": (
                "pass"
                if config.mode == "full" and all(loo_checks.values())
                else "not_evaluated" if config.mode != "full" else "fail"
            ),
            "checks": loo_checks,
        },
        "completed_models": list(evaluations),
    }
    artifacts = sorted(
        path
        for folder in (paths["tables"], paths["figures"], paths["models"])
        for path in folder.rglob("*")
        if path.is_file()
    )
    manifest["artifact_sha256"] = {
        path.relative_to(paths["root"]).as_posix(): _file_sha256(path)
        for path in artifacts
    }
    write_json(paths["root"] / "manifest.json", manifest)
    return manifest


def build_independent_exposure_data(
    frame: pd.DataFrame, *, variant: str = "updated_wide"
) -> ExposureData:
    """Build the independent validator matrices from the canonical 49×12 grid."""
    required = {
        "participant_id", "outcome", "setting_group", "participation",
        "contacts", "duration", "pcm",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Independent validation grid missing columns: {missing}")
    participant = (
        frame[["participant_id", "outcome"]].drop_duplicates()
        .sort_values("participant_id")
    )
    participants = participant["participant_id"].tolist()
    settings = list(WIDE_SETTINGS)
    if len(frame) != len(participants) * len(settings):
        raise ValueError("Independent validator requires a balanced participant-setting grid")
    model_column = {
        "participation": "participation",
        "contacts": "contacts",
        "duration_minutes": "duration",
        "pcm": "pcm",
    }
    matrices = {}
    for model, column in model_column.items():
        pivot = frame.pivot(index="participant_id", columns="setting_group", values=column)
        pivot = pivot.reindex(index=participants, columns=settings)
        if pivot.isna().any().any():
            raise ValueError(f"Missing participant-setting cells for independent model {model}")
        matrices[model] = pivot.to_numpy(dtype=float)
    outcomes = (
        participant.set_index("participant_id").reindex(participants)["outcome"]
        .to_numpy(dtype=float)
    )
    return ExposureData(
        variant=variant,
        source=Path("in_memory_updated_participant_wide"),
        participant_ids=tuple(str(value) for value in participants),
        settings=tuple(settings),
        outcomes=outcomes,
        matrices=matrices,
    )


def _independent_summary_markdown(
    metrics: Sequence[dict[str, object]], variants: Sequence[str]
) -> str:
    lines = [
        "# Independent MAP + Laplace approximate robustness validation",
        "",
        "This is an approximate direction check, not the main PyMC/MCMC result.",
        "",
    ]
    for variant in variants:
        lines.extend(
            [
                f"## {variant}",
                "",
                "| Model | ELPD | SE | Brier | AUC | Exposure rank |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        rows = [row for row in metrics if row["data_variant"] == variant]
        for row in sorted(rows, key=lambda item: float(item["elpd"]), reverse=True):
            lines.append(
                "| {model} | {elpd:.3f} | {se:.3f} | {brier:.4f} | {auc:.4f} | {rank} |".format(
                    model=row["model"],
                    elpd=float(row["elpd"]),
                    se=float(row["elpd_se"]),
                    brier=float(row["brier_score"]),
                    auc=float(row["auc"]),
                    rank=row["nominal_rank_exposure"] or "NA",
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_independent_validation_stage(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_draws: int,
    seed: int,
    input_sha256: str,
    nested_inner_folds: int | None = None,
) -> dict[str, Any]:
    """Run one-version approximate LOPO plus the nested setting-specific D module."""
    if bootstrap_draws < 100:
        raise ValueError("independent bootstrap_draws must be at least 100")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = build_independent_exposure_data(frame, variant="updated_wide")
    predictions, scaling_rows = run_loo_for_dataset(data)
    variants = ("updated_wide",)
    metrics = calculate_metrics(predictions, variants)
    ranks, pairwise = bayesian_bootstrap_outputs(
        predictions, variants, bootstrap_draws, seed
    )

    d_nested = run_nested_setting_specific_selection(
        data,
        predictions,
        bootstrap_draws=bootstrap_draws,
        seed=seed + 50_000,
        anchor=D_ANCHOR_MODEL,
        inner_folds=nested_inner_folds,
    )
    d_nested["predictions"].to_csv(
        output_dir / "d_nested_setting_specific_predictions.csv",
        index=False, encoding="utf-8-sig",
    )
    d_nested["selections"].to_csv(
        output_dir / "d_nested_setting_metric_selections_by_fold.csv",
        index=False, encoding="utf-8-sig",
    )
    d_nested["candidate_inner_elpd"].to_csv(
        output_dir / "d_nested_setting_candidate_inner_elpd.csv",
        index=False, encoding="utf-8-sig",
    )
    d_nested["selection_frequency"].to_csv(
        output_dir / "d_nested_setting_selection_frequency.csv",
        index=False, encoding="utf-8-sig",
    )
    d_nested["metrics"].to_csv(
        output_dir / "d_nested_setting_specific_model_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    d_nested["paired_comparisons"].to_csv(
        output_dir / "d_nested_setting_specific_vs_common.csv",
        index=False, encoding="utf-8-sig",
    )
    d_nested["rank_probabilities"].to_csv(
        output_dir / "d_nested_setting_specific_rank_probabilities.csv",
        index=False, encoding="utf-8-sig",
    )
    write_json(output_dir / "d_nested_setting_specific_manifest.json", d_nested["manifest"])

    participant_ids = list(data.participant_ids)
    fold_map = {value: index + 1 for index, value in enumerate(participant_ids)}
    public_predictions: list[dict[str, object]] = []
    for row in predictions:
        public = dict(row)
        public["heldout_fold"] = fold_map[str(public.pop("participant_id"))]
        public_predictions.append(public)
    public_scaling: list[dict[str, object]] = []
    for row in scaling_rows:
        public = dict(row)
        public["heldout_fold"] = fold_map[str(public.pop("heldout_participant_id"))]
        public_scaling.append(public)

    pd.DataFrame(public_predictions).to_csv(
        output_dir / "loo_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(public_scaling).to_csv(
        output_dir / "fold_scaling.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(metrics).to_csv(
        output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pairwise).to_csv(
        output_dir / "pairwise_delta_elpd.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(ranks).to_csv(
        output_dir / "bootstrap_rank_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.md").write_text(
        _independent_summary_markdown(metrics, variants), encoding="utf-8"
    )

    expected_fits = len(data.participant_ids) * len(ALL_MODELS)
    checks = {
        "fit_count_245": len(predictions) == expected_fits == 245,
        "all_probabilities_finite": bool(
            np.isfinite([float(row["predicted_probability"]) for row in predictions]).all()
        ),
        "all_probabilities_in_unit_interval": all(
            0.0 < float(row["predicted_probability"]) < 1.0 for row in predictions
        ),
        "all_fits_accepted": all(int(row["fit_accepted"]) == 1 for row in predictions),
        "d_nested_setting_specific_pass": d_nested["manifest"]["status"] == "pass",
    }
    metadata = {
        "status": "pass" if all(checks.values()) else "fail",
        "analysis_label": "independent approximate robustness validation; not a numerical reimplementation of the main hierarchical PyMC/MCMC posterior",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SciPy MAP with analytic derivatives and Hessian Laplace approximation; independent N(0,0.5^2) slope priors, fold-local scaling",
        "predictive_integration": f"{N_GAUSS_HERMITE}-node Gauss-Hermite quadrature",
        "input_sha256": input_sha256,
        "source_path_redacted": True,
        "source_variant": "updated_wide",
        "loo_unit": "participant",
        "n_participants": len(data.participant_ids),
        "n_positive": int(data.outcomes.sum()),
        "n_negative": int(len(data.outcomes) - data.outcomes.sum()),
        "n_settings": len(data.settings),
        "models": list(ALL_MODELS),
        "bootstrap_draws": bootstrap_draws,
        "seed": seed,
        "checks": checks,
        "fit_counts": {
            "total": len(predictions),
            "per_model": len(data.participant_ids),
            "d_nested_inner": d_nested["manifest"]["inner_fit_count"],
            "d_nested_outer_final": d_nested["manifest"]["outer_final_fit_count"],
        },
        "d_nested_setting_specific": d_nested["manifest"],
        "d_nested_outputs": {
            "predictions": "d_nested_setting_specific_predictions.csv",
            "selections_by_fold": "d_nested_setting_metric_selections_by_fold.csv",
            "candidate_inner_elpd": "d_nested_setting_candidate_inner_elpd.csv",
            "selection_frequency": "d_nested_setting_selection_frequency.csv",
            "model_metrics": "d_nested_setting_specific_model_metrics.csv",
            "paired_vs_common": "d_nested_setting_specific_vs_common.csv",
            "rank_probabilities": "d_nested_setting_specific_rank_probabilities.csv",
        },
    }
    write_json(output_dir / "manifest.json", metadata)
    return metadata


THESIS_FIGURE_STEMS = (
    "fig_1_research_workflow",
    "fig_2_data_audit",
    "fig_3_main_model_comparison",
    "fig_4_setting_forest_contacts",
    "fig_5_sensitivity_matrix",
    "fig_6_household_extension",
    "fig_7_prior_posterior_predictive",
    "fig_a1_independent_validation",
)
THESIS_FIGURE_FORMATS = ("png", "tiff", "svg", "pdf")
THESIS_TABLE_FILES = (
    "table_sample_audit_en.csv",
    "table_setting_audit_en.csv",
    "table_participant_descriptives_en.csv",
    "table_main_comparison_en.csv",
    "table_diagnostics_summary_en.csv",
    "table_sensitivity_en.csv",
    "table_household_en.csv",
    "table_independent_validation_en.csv",
    "table_all_coefficients.csv",
)


def _configure_thesis_matplotlib() -> str:
    """Configure the exclusive Python/matplotlib publication backend."""
    global mpl, plt, font_manager, FancyBboxPatch, FancyArrowPatch, FONT_NAME

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    candidates = [
        windir / "Fonts" / "arial.ttf",
        windir / "Fonts" / "arialbd.ttf",
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    regular = next((path for path in candidates if path.is_file()), None)
    if regular is not None:
        font_manager.fontManager.addfont(str(regular))
        for candidate in candidates:
            if candidate.is_file() and candidate != regular:
                font_manager.fontManager.addfont(str(candidate))
        FONT_NAME = font_manager.FontProperties(fname=str(regular)).get_name()
    else:
        FONT_NAME = "DejaVu Sans"

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                FONT_NAME,
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return FONT_NAME


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_number(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def build_thesis_audit_inputs(frame: pd.DataFrame, audit_dir: Path) -> dict[str, Any]:
    """Create aggregate, identifier-free audit inputs from the validated 49×12 grid."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "participant_id", "outcome", "setting_group", "participation", "activity_count",
        "contacts", "duration", "pcm", "household_contacts", "household_pcm",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Thesis audit grid missing columns: {missing}")
    used = frame.copy()
    participant = (
        used.groupby("participant_id", as_index=False)
        .agg(
            outcome=("outcome", "first"),
            n_activities=("activity_count", "sum"),
            n_setting_groups=("participation", "sum"),
            total_contacts=("contacts", "sum"),
            total_duration_minutes=("duration", "sum"),
            total_pcm=("pcm", "sum"),
            total_household_contacts=("household_contacts", "sum"),
            total_household_pcm=("household_pcm", "sum"),
        )
        .sort_values("participant_id")
    )

    visit = (
        used.groupby(["setting_group", "outcome"], as_index=False)
        .agg(
            participants_visiting=("participation", "sum"),
            activities=("activity_count", "sum"),
            contacts=("contacts", "sum"),
            duration_minutes=("duration", "sum"),
            pcm=("pcm", "sum"),
            household_contacts=("household_contacts", "sum"),
            household_pcm=("household_pcm", "sum"),
        )
    )
    visit_wide = (
        visit.pivot(index="setting_group", columns="outcome", values="participants_visiting")
        .fillna(0)
        .reindex(index=list(WIDE_SETTINGS), fill_value=0)
        .reset_index()
    )
    visit_wide.columns = [
        "setting_group" if column == "setting_group" else f"visitors_outcome_{int(column)}"
        for column in visit_wide.columns
    ]
    for column in ("visitors_outcome_0", "visitors_outcome_1"):
        if column not in visit_wide:
            visit_wide[column] = 0
    visit_wide["separation_flag"] = np.where(
        (visit_wide["visitors_outcome_0"] == 0) | (visit_wide["visitors_outcome_1"] == 0),
        "complete",
        np.where(
            visit_wide[["visitors_outcome_0", "visitors_outcome_1"]].min(axis=1) <= 1,
            "near",
            "none",
        ),
    )

    participant_metrics = [
        "n_activities", "n_setting_groups", "total_contacts", "total_duration_minutes",
        "total_pcm", "total_household_contacts", "total_household_pcm",
    ]
    descriptives = (
        participant.groupby("outcome")[participant_metrics]
        .describe()
        .transpose()
        .reset_index()
    )

    all_zero_mask = participant[
        ["n_activities", "n_setting_groups", "total_contacts", "total_duration_minutes",
         "total_pcm", "total_household_contacts", "total_household_pcm"]
    ].eq(0).all(axis=1)
    all_zero_positive_count = int((all_zero_mask & participant["outcome"].eq(1)).sum())
    hh_gt_contacts = int((used["household_contacts"] > used["contacts"]).sum())
    participation_activity_mismatch = int(
        (used["participation"].astype(int) != (used["activity_count"] > 0).astype(int)).sum()
    )
    structural_zero_violation = int(
        (
            used.loc[used["participation"].eq(0), [
                "activity_count", "contacts", "duration", "pcm",
                "household_contacts", "household_pcm",
            ]].ne(0).any(axis=1)
        ).sum()
    )
    report = {
        "dataset": {
            "source_rows": int(participant["participant_id"].nunique()),
            "participant_setting_cells": int(len(used)),
            "participants": int(participant["participant_id"].nunique()),
            "reported_activities": int(used["activity_count"].sum()),
            "all_zero_positive_count": all_zero_positive_count,
            "participation_activity_mismatch_count": participation_activity_mismatch,
            "structural_zero_violation_count": structural_zero_violation,
            "hhcontacts_gt_contacts_cell_count": hh_gt_contacts,
            "cleaning_performed": False,
            "validation_only": True,
        },
        "outcome_counts": _safe_records(
            participant.groupby("outcome", as_index=False).agg(
                participants=("participant_id", "size"),
                activities=("n_activities", "sum"),
            )
        ),
        "setting_participant_counts": _safe_records(visit_wide),
        "source_path_redacted": True,
        "contacts_household_relationship": CONTACTS_HOUSEHOLD_RELATIONSHIP,
    }
    write_json(audit_dir / "audit_summary.json", report)
    write_csv(descriptives, audit_dir / "participant_descriptives_by_outcome.csv")
    return report


def _write_thesis_asset_manifest(output_dir: Path, font_name: str) -> dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "asset_manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    missing_figures = [
        f"figures/{stem}.{extension}"
        for stem in THESIS_FIGURE_STEMS
        for extension in THESIS_FIGURE_FORMATS
        if not (output_dir / "figures" / f"{stem}.{extension}").is_file()
        or (output_dir / "figures" / f"{stem}.{extension}").stat().st_size == 0
    ]
    missing_tables = [
        f"tables/{name}"
        for name in THESIS_TABLE_FILES
        if not (output_dir / "tables" / name).is_file()
        or (output_dir / "tables" / name).stat().st_size == 0
    ]
    missing = missing_figures + missing_tables
    table_files = sorted((output_dir / "tables").glob("*.csv"))
    manifest = {
        "status": "pass" if not missing else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "python/matplotlib only",
        "font": font_name,
        "source_paths_redacted": True,
        "figure_count": len(THESIS_FIGURE_STEMS),
        "formats_per_figure": list(THESIS_FIGURE_FORMATS),
        "figure_file_count": len(THESIS_FIGURE_STEMS)
        * len(THESIS_FIGURE_FORMATS),
        "expected_table_count": len(THESIS_TABLE_FILES),
        "table_count": len(table_files),
        "missing_expected_files": missing,
        "files": files,
    }
    write_json(output_dir / "asset_manifest.json", manifest)
    if missing:
        raise RuntimeError(f"Thesis figure export incomplete: {missing}")
    return manifest


def run_thesis_assets_stage(
    frame: pd.DataFrame,
    *,
    audit_dir: Path,
    results_dir: Path,
    independent_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate all seven main-text figures and Appendix Figure A1."""
    global AUDIT, RESULTS, INDEPENDENT, OUT, FIG, TAB

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty thesis asset directory: {output_dir}")
    AUDIT = audit_dir.resolve()
    RESULTS = results_dir.resolve()
    INDEPENDENT = independent_dir.resolve()
    OUT = output_dir.resolve()
    FIG = OUT / "figures"
    TAB = OUT / "tables"
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    build_thesis_audit_inputs(frame, AUDIT)
    required = [
        AUDIT / "audit_summary.json",
        AUDIT / "participant_descriptives_by_outcome.csv",
        RESULTS / "manifest.json",
        RESULTS / "tables" / "table_2_main_model_comparison.csv",
        RESULTS / "tables" / "main_rank_probabilities.csv",
        RESULTS / "tables" / "table_3_setting_coefficients.csv",
        RESULTS / "tables" / "table_5_sensitivity.csv",
        RESULTS / "tables" / "sensitivity_rank_probabilities.csv",
        RESULTS / "tables" / "table_6_household_extension.csv",
        RESULTS / "tables" / "prior_posterior_predictive_counts.csv",
        RESULTS / "tables" / "model_diagnostics.csv",
        INDEPENDENT / "model_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing thesis-figure inputs: {missing}")

    font_name = _configure_thesis_matplotlib()
    figure_1_workflow()
    figure_2_data_audit()
    figure_3_main_comparison()
    figure_4_setting_forest()
    figure_5_sensitivity()
    figure_6_household()
    figure_7_predictive_checks()
    figure_a1_independent_validation()
    write_tables()
    return _write_thesis_asset_manifest(OUT, font_name)


SAFE_BAYESIAN_FILES = (
    "validation.json",
    "feature_metadata.json",
    "contacts_household_semantic_contract.json",
    "household_structured_feature_audit.json",
    "tables/table_1_sample_audit.csv",
    "tables/table_2_main_model_comparison.csv",
    "tables/main_rank_probabilities.csv",
    "tables/prior_posterior_predictive_counts.csv",
    "tables/table_3_setting_coefficients.csv",
    "tables/table_5_sensitivity.csv",
    "tables/sensitivity_model_comparisons.csv",
    "tables/sensitivity_rank_probabilities.csv",
    "tables/table_6_household_extension.csv",
    "tables/table_7_setting_metric_candidate_configurations.csv",
    "tables/table_8_setting_metric_predictive_support.csv",
    "tables/table_9_setting_metric_preference_summary.csv",
    "tables/setting_metric_arviz_comparisons.csv",
    "tables/setting_metric_rank_probabilities.csv",
    "tables/table_10_planned_hybrid_mapping.csv",
    "tables/table_11_common_vs_planned_hybrid_comparison.csv",
    "tables/table_12_common_vs_hybrid_paired_differences.csv",
    "tables/common_vs_hybrid_rank_probabilities.csv",
    "tables/setting_exposure_by_outcome_descriptive.csv",
    "tables/setting_contact_count_by_outcome_plot_data.csv",
    "tables/table_14_household_additive_modification_comparison.csv",
    "tables/table_15_household_paired_differences.csv",
    "tables/table_16_household_structured_effects.csv",
    "tables/household_structured_rank_probabilities.csv",
    "tables/household_modification_prior_sensitivity.csv",
    "tables/household_modification_prior_coefficients.csv",
    "tables/household_participant_setting_deidentified.csv",
    "tables/model_diagnostics.csv",
)


def publish_safe_bayesian_outputs(restricted_root: Path, public_root: Path) -> None:
    """Copy aggregate outputs only; pointwise participant tables remain restricted."""
    for relative in SAFE_BAYESIAN_FILES:
        source = restricted_root / relative
        if not source.is_file():
            continue
        destination = public_root / "bayesian" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    figures = restricted_root / "figures"
    if figures.is_dir():
        destination = public_root / "bayesian" / "figures"
        destination.mkdir(parents=True, exist_ok=True)
        for extension in ("*.png", "*.svg", "*.pdf", "*.tiff"):
            for source in figures.glob(extension):
                shutil.copy2(source, destination / source.name)


def _scan_public_tree(public_root: Path, source_ids: Sequence[Any]) -> dict[str, Any]:
    """Reject identifier-bearing fields, leaked model names, and user paths.

    Numeric participant IDs cannot safely be searched as arbitrary substrings:
    values such as ``1`` or ``46`` legitimately occur in estimates and counts.
    The hard privacy contract is therefore structural: public CSV/JSON files may
    not contain identifier fields, reloo labels must be anonymous fold labels,
    and no absolute user path may appear.  Distinctive non-numeric IDs are also
    searched as complete delimited values.
    """
    identifiers = {
        str(value)
        for value in source_ids
        if not str(value).isdigit() and len(str(value)) >= 4
    }

    def json_has_forbidden_field(node: Any, parent_key: str = "") -> bool:
        if isinstance(node, dict):
            for key, value in node.items():
                normalized = str(key).lower()
                if normalized in {"participant_id", "individual", "heldout_participant_id"}:
                    # validation.json records the missing-value count for every
                    # source column; the key here is schema metadata, not an ID.
                    if parent_key == "missing_counts" and isinstance(value, (int, float)):
                        continue
                    return True
                if json_has_forbidden_field(value, normalized):
                    return True
        elif isinstance(node, list):
            return any(json_has_forbidden_field(value, parent_key) for value in node)
        return False

    forbidden_field_hits: list[str] = []
    distinctive_value_hits: list[str] = []
    absolute_path_hits: list[str] = []
    for path in public_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        relative = path.relative_to(public_root).as_posix()
        forbidden_field = False
        if path.suffix.lower() == ".csv":
            header = next(iter(text.splitlines()), "")
            columns = {item.strip().strip('"').lower() for item in header.split(",")}
            forbidden_field = bool(
                columns.intersection({"participant_id", "individual", "heldout_participant_id"})
            )
        elif path.suffix.lower() == ".json":
            try:
                forbidden_field = json_has_forbidden_field(json.loads(text))
            except json.JSONDecodeError:
                forbidden_field = True
        if forbidden_field:
            forbidden_field_hits.append(relative)
        for identifier in identifiers:
            pattern = rf'(?<![A-Za-z0-9_-]){re.escape(identifier)}(?![A-Za-z0-9_-])'
            if re.search(pattern, text):
                distinctive_value_hits.append(relative)
                break
        if re.search(r"[A-Za-z]:[\\/]Users[\\/]", text, flags=re.IGNORECASE):
            absolute_path_hits.append(relative)
    result = {
        "forbidden_identifier_field_hits": sorted(set(forbidden_field_hits)),
        "distinctive_identifier_value_hits": sorted(set(distinctive_value_hits)),
        "absolute_user_path_hits": sorted(set(absolute_path_hits)),
    }
    result["status"] = (
        "pass"
        if not result["forbidden_identifier_field_hits"]
        and not result["distinctive_identifier_value_hits"]
        and not result["absolute_user_path_hits"]
        else "fail"
    )
    return result


def synthetic_self_test(
    seed: int = 20_260_804, *, exact_reloo: bool = False
) -> dict[str, object]:
    """Small API-path test; this is never inferential evidence."""
    rng = np.random.default_rng(seed)
    n, p = 16, 3
    X = rng.normal(size=(n, p))
    probability = expit(-0.3 + X @ np.array([0.5, -0.2, 0.1]))
    y = rng.binomial(1, probability)
    if len(np.unique(y)) < 2:
        y[:2] = [0, 1]
    design = Design(
        metric="synthetic",
        X=X,
        y=y,
        participant_ids=np.arange(n),
        setting_names=[f"s{index + 1}" for index in range(p)],
        household_any=rng.binomial(1, 0.5, size=n).astype(float),
        household_center=0.5,
        household_contact_intensity_z=rng.normal(size=n),
        household_contact_log1p_mean=1.0,
        household_contact_log1p_scale=0.8,
        transform="synthetic",
        means=np.zeros(p),
        scales=np.ones(p),
        variant="self_test",
    )
    settings = FitSettings(100, 100, 1, 1, 0.9, False)
    fit = fit_model(
        design,
        name="synthetic",
        prior_scale=0.5,
        settings=settings,
        seed=seed,
        sample_predictive=False,
    )
    evaluation = evaluate_fit(
        fit, exact_reloo=exact_reloo, reloo_seed=seed + 10_000
    )
    if len(evaluation.predictions) != n:
        raise AssertionError("Unexpected number of LOO predictions")
    if not evaluation.predictions["loo_probability"].between(0, 1).all():
        raise AssertionError("Invalid LOO probabilities")
    return {
        "status": "pass",
        "participants": n,
        "draws": settings.draws,
        "elpd": evaluation.metrics["elpd"],
        "brier": evaluation.metrics["brier"],
        "high_k_count": evaluation.metrics["high_k_count_initial"],
        "exact_refit_count": evaluation.metrics["exact_refit_count"],
        "note": "Single-chain smoke test verifies API paths only; not inferential evidence.",
    }


def run_complete_analysis(
    data_path: Path,
    output_dir: Path,
    config: RunConfig,
    *,
    strict_known_data: bool,
    steps: str,
    run_continuous: bool,
    run_independent: bool,
    run_thesis_figures: bool,
) -> dict[str, Any]:
    """Execute all requested stages in scientific workflow order."""
    if run_thesis_figures and not run_independent:
        raise ValueError(
            "Complete thesis figures require the independent-validation stage for Figure A1"
        )
    source = data_path.resolve()
    root = output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    configure_runtime(root / "restricted" / ".cache")

    clean = load_and_validate(source, strict_known_data=strict_known_data)
    total = (
        1
        if steps == "audit"
        else 2
        + int(run_continuous)
        + int(run_independent)
        + int(run_thesis_figures)
    )
    master: dict[str, Any] = {
        "status": "running",
        "preliminary": config.mode != "full",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "source_file": source.name,
        "source_path_redacted": True,
        "input_sha256": clean.sha256,
        "config": config.to_dict(),
        "requested_steps": steps,
        "privacy_layout": {
            "public": "aggregate or de-identified outputs",
            "restricted": "participant-level outputs; do not publish",
            "thesis_figures": "public/thesis_assets/figures; 8 figures x 4 formats",
        },
        "stages": {},
    }
    write_json(root / "manifest.json", master)
    try:
        _stage(1, total, "Data-contract checks, feature construction, and main Bayesian analysis")
        restricted_bayesian = root / "restricted" / "bayesian"
        bayesian_manifest = run_pipeline(
            source,
            restricted_bayesian,
            config,
            strict_known_data=strict_known_data,
            steps=("audit",) if steps == "audit" else ("all",),
        )
        master["stages"]["bayesian"] = bayesian_manifest
        if steps == "audit":
            master["status"] = "complete"
            master["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(root / "manifest.json", master)
            return master

        public_root = root / "public"
        publish_safe_bayesian_outputs(restricted_bayesian, public_root)

        current = 2
        if run_continuous:
            _stage(current, total, "Post-review continuous household-contact intensity auxiliary analysis")
            continuous_manifest = run_continuous_household_stage(
                source,
                public_root / "continuous_household",
                config,
                strict_known_data=strict_known_data,
            )
            master["stages"]["continuous_household"] = continuous_manifest
            current += 1

        if run_independent:
            _stage(current, total, "Independent MAP+Laplace approximate robustness validation (leave-one-participant-out)")
            independent_manifest = run_independent_validation_stage(
                clean.frame,
                public_root / "independent_validation",
                bootstrap_draws=config.bootstrap_draws,
                seed=config.derived_seed("independent_validation"),
                input_sha256=clean.sha256,
                # Formal full mode uses explicit fold-wise refitted inner LOPO; quick mode uses
                # 5-fold stratified inner CV only as a pipeline smoke check.
                nested_inner_folds=None if config.mode == "full" else 5,
            )
            master["stages"]["independent_validation"] = independent_manifest
            current += 1

        if run_thesis_figures:
            _stage(current, total, "Generate main manuscript figures, appendix figures, and summary tables")
            thesis_figure_manifest = run_thesis_assets_stage(
                clean.frame,
                audit_dir=root / "restricted" / "thesis_figure_inputs" / "audit",
                results_dir=restricted_bayesian,
                independent_dir=public_root / "independent_validation",
                output_dir=public_root / "thesis_assets",
            )
            master["stages"]["thesis_figures"] = thesis_figure_manifest
            current += 1

        _stage(total, total, "Privacy scan, artifact hashes, and final manifest")
        privacy = _scan_public_tree(public_root, clean.frame["participant_id"].unique())
        if privacy["status"] != "pass":
            raise RuntimeError(f"Public-output privacy scan failed: {privacy}")
        master["public_output_privacy_gate"] = privacy

        artifacts = sorted(
            path for path in root.rglob("*")
            if path.is_file()
            and path.name != "manifest.json"
            and ".cache" not in path.parts
        )
        master["artifact_sha256"] = {
            path.relative_to(root).as_posix(): _file_sha256(path)
            for path in artifacts
        }
        full_stage_gates: list[str] = []
        if config.mode == "full":
            full_stage_gates.extend(
                [
                    bayesian_manifest.get("diagnostic_gate", {}).get("status", "missing"),
                    bayesian_manifest.get("loo_reliability_gate", {}).get("status", "missing"),
                ]
            )
            if run_continuous:
                full_stage_gates.extend(
                    [
                        continuous_manifest["diagnostic_gate"]["status"],
                        continuous_manifest["loo_reliability_gate"]["status"],
                    ]
                )
            if run_independent:
                full_stage_gates.append(independent_manifest["status"])
            if run_thesis_figures:
                full_stage_gates.append(thesis_figure_manifest["status"])
        master["all_full_gates_pass"] = (
            all(value == "pass" for value in full_stage_gates)
            if config.mode == "full" else None
        )
        master["status"] = "complete"
        master["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "manifest.json", master)
        return master
    except Exception as error:
        master["status"] = "failed"
        master["finished_utc"] = datetime.now(timezone.utc).isoformat()
        master["error_type"] = type(error).__name__
        master["error"] = str(error).replace(str(source), source.name)
        write_json(root / "manifest.json", master)
        raise


# =============================================================================
# 10. Single command-line entry point
# =============================================================================

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file complete Bayesian analysis, independent validation, and manuscript figures/tables"
    )
    parser.add_argument("--data", type=Path, help="Path to the source CSV")
    parser.add_argument("--output", type=Path, help="Output directory; it must be empty")
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--config", type=Path, default=None, help="Optional TOML configuration; not required by default")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=None)
    parser.add_argument("--sensitivity-draws", type=int, default=None)
    parser.add_argument("--sensitivity-tune", type=int, default=None)
    parser.add_argument("--sensitivity-chains", type=int, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--steps", choices=("audit", "all"), default="all")
    parser.add_argument("--strict-known-data", action="store_true")
    parser.add_argument("--save-idata", action="store_true")
    parser.add_argument("--progressbar", action="store_true")
    parser.add_argument("--skip-continuous-household", action="store_true")
    parser.add_argument("--skip-independent-validation", action="store_true")
    parser.add_argument("--skip-thesis-figures", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-exact", action="store_true")
    parser.add_argument("--print-embedded-config", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_embedded_config:
        print(json.dumps({key: value.to_dict() for key, value in DEFAULTS.items()}, indent=2))
        return 0

    if args.self_test or args.self_test_exact:
        cache_root = (args.output or Path.cwd() / ".single_file_runtime") / ".cache"
        configure_runtime(cache_root)
        result = synthetic_self_test(
            seed=args.seed or 20_260_804,
            exact_reloo=args.self_test_exact,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.data is None or args.output is None:
        raise SystemExit("--data and --output are required except in self-test mode")
    if args.skip_independent_validation and not args.skip_thesis_figures:
        raise SystemExit(
            "Figure A1 depends on independent validation; when using --skip-independent-validation, you must also use "
            "--skip-thesis-figures"
        )

    config = load_config(
        args.mode,
        args.config,
        seed=args.seed,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        sensitivity_draws=args.sensitivity_draws,
        sensitivity_tune=args.sensitivity_tune,
        sensitivity_chains=args.sensitivity_chains,
        bootstrap_draws=args.bootstrap_draws,
        save_idata=True if args.save_idata else None,
        progressbar=True if args.progressbar else None,
    )
    manifest = run_complete_analysis(
        args.data,
        args.output,
        config,
        strict_known_data=args.strict_known_data,
        steps=args.steps,
        run_continuous=not args.skip_continuous_household,
        run_independent=not args.skip_independent_validation,
        run_thesis_figures=not args.skip_thesis_figures,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
