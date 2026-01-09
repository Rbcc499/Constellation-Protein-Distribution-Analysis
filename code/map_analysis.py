"""
map_analysis.py

Core analysis logic for Constellation:
- Load Imaris SpotsDistances CSVs
- Apply expansion factor correction
- Build df_all with all pairwise protein-pair columns (3 proteins => 3 pairs, 4 proteins => 6 pairs)
- Optional alignment filter (only when toggled on)
- Optional angle filter (only when toggled on)

Notes:
- Protein names are cleaned to the first token (split by space), with underscores turned into spaces.
- Assumes CSV has: Object1, Object2, SpotIndex1, ClosestSpotIndex2, MinDistance (plus optional MeanDistance/MaxDistance).
"""

from __future__ import annotations

import os
import re
import math
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers: cleaning & parsing
# -----------------------------

def load_spots_distances(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path, encoding="latin1")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported data file type: {ext}")

    # Clean object names
    if "Object1" not in df.columns or "Object2" not in df.columns:
        raise ValueError("Data file must contain Object1 and Object2 columns.")

    df["Object1"] = df["Object1"].astype(str).map(clean_protein_name)
    df["Object2"] = df["Object2"].astype(str).map(clean_protein_name)

    # Drop optional columns
    for col in ["MeanDistance", "MaxDistance"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

def clean_protein_name(raw: str) -> str:
    """Take first 'word' of an Imaris object name, after replacing underscores with spaces."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):  # type: ignore[arg-type]
        return ""
    s = str(raw).replace("_", " ").strip()
    if not s:
        return ""
    return s.split()[0]


def parse_mouse_slice_round(filename: str) -> Tuple[str, str, int]:
    """
    Expect file starts like: Mouse_Slice_Round_...
    Example: K276B_s2h2_1_DG_...csv  -> ("K276B","s2h2",1)
    """
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Could not parse Mouse/Slice/Round from filename: {filename}")
    mouse = parts[0]
    slice_id = parts[1]
    try:
        round_id = int(re.sub(r"\D", "", parts[2]))  # keep digits only
    except Exception as e:
        raise ValueError(f"Could not parse Round from filename: {filename}") from e
    return mouse, slice_id, round_id


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_save_excel(df: pd.DataFrame, path: str) -> str:
    """
    Save df to Excel. If the file is locked (PermissionError), save a timestamped copy.
    Returns the actual path used.
    """
    folder = os.path.dirname(path)
    if folder:
        safe_mkdir(folder)

    try:
        df.to_excel(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"{root}_{ts}{ext}"
        df.to_excel(alt, index=False)
        return alt


# -----------------------------
# Public API for the app
# -----------------------------

def get_protein_pairs_with_distances(data_folder: str):
    if not data_folder or not os.path.isdir(data_folder):
        return [], []

    files = [f for f in os.listdir(data_folder) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    if not files:
        return [], []

    # Pick the smallest file (often faster/less likely to be huge)
    candidates = [os.path.join(data_folder, f) for f in files]
    first_path = min(candidates, key=lambda p: os.path.getsize(p))

    ext = os.path.splitext(first_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(
            first_path,
            usecols=["Object1", "Object2"],
            dtype=str,
            engine="c",
            low_memory=False
        )
    else:
        # Excel
        df = pd.read_excel(
            first_path,
            usecols=["Object1", "Object2"],
            dtype=str
        )

    if "Object1" not in df.columns or "Object2" not in df.columns:
        return [], []

    obj1 = df["Object1"].astype(str).map(clean_protein_name)
    obj2 = df["Object2"].astype(str).map(clean_protein_name)

    proteins = sorted(set(obj1.tolist() + obj2.tolist()) - {""})
    pairs = sorted({f"{a}-{b}" for a, b in set(zip(obj1, obj2)) if a and b})

    return pairs, proteins


# -----------------------------
# Data loading & expansion factor
# -----------------------------

def load_spots_distances_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin1")

    # Clean object names
    df["Object1"] = df["Object1"].astype(str).map(clean_protein_name)
    df["Object2"] = df["Object2"].astype(str).map(clean_protein_name)

    # Drop optional columns
    for col in ["MeanDistance", "MaxDistance"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def load_expansion_factors(expansion_factor_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(expansion_factor_xlsx)
    # normalize types
    for c in ["Mouse Name", "Slice"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "Round" in df.columns:
        df["Round"] = df["Round"].astype(int)
    if "Expansion Factor" in df.columns:
        df["Expansion Factor"] = df["Expansion Factor"].astype(float)
    return df


def get_expansion_factor(df_exp: pd.DataFrame, mouse: str, slice_id: str, round_id: int) -> float:
    required = {"Mouse Name", "Slice", "Round", "Expansion Factor"}
    if not required.issubset(df_exp.columns):
        raise ValueError(f"Expansion factor file must contain columns: {sorted(required)}")

    match = df_exp[
        (df_exp["Mouse Name"].astype(str) == str(mouse)) &
        (df_exp["Slice"].astype(str) == str(slice_id)) &
        (df_exp["Round"].astype(int) == int(round_id))
    ]
    if match.empty:
        raise ValueError(f"No expansion factor found for Mouse={mouse}, Slice={slice_id}, Round={round_id}")

    return float(match.iloc[0]["Expansion Factor"])


def apply_expansion_factor(df: pd.DataFrame, expansion_factor: float) -> pd.DataFrame:
    if "MinDistance" not in df.columns:
        raise ValueError("CSV missing 'MinDistance' column.")

    df = df.copy()
    df["Actual_Min_Distance"] = df["MinDistance"].astype(float) / float(expansion_factor)
    return df


# -----------------------------
# Enrichment summary (optional)
# -----------------------------

def compute_pair_enrichment(df_actual: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    """
    Summarize 'enrichment' per protein pair using the *raw* pairwise table (df_actual).

    For each (undirected) protein pair, counts:
      - Total Count: total pair observations in the CSV
      - Count < cutoff: how many are within the cutoff (in the same units as Actual_Min_Distance, typically µm)
      - Fraction Close: Count < cutoff / Total Count

    This matches the intent: "within 300 nm vs overall amount" (300 nm = 0.3 µm).
    """
    needed = {"Object1", "Object2", "Actual_Min_Distance"}
    if not needed.issubset(df_actual.columns):
        raise ValueError(
            f"df_actual missing required columns for enrichment: {sorted(needed - set(df_actual.columns))}"
        )

    df_tmp = df_actual[["Object1", "Object2", "Actual_Min_Distance"]].copy()
    df_tmp["Object1"] = df_tmp["Object1"].astype(str).map(clean_protein_name)
    df_tmp["Object2"] = df_tmp["Object2"].astype(str).map(clean_protein_name)
    df_tmp = df_tmp[(df_tmp["Object1"] != "") & (df_tmp["Object2"] != "")].copy()

    # Make pair undirected so "A-B" and "B-A" collapse together
    o1 = df_tmp["Object1"].astype(str)
    o2 = df_tmp["Object2"].astype(str)
    pair = np.where(o1 <= o2, o1 + "-" + o2, o2 + "-" + o1)
    df_tmp["Protein Pair"] = pair

    df_tmp["Distance"] = df_tmp["Actual_Min_Distance"].astype(float)
    df_tmp["IsClose"] = df_tmp["Distance"] < float(cutoff)

    grp = df_tmp.groupby("Protein Pair", sort=True, dropna=False)
    out = grp.agg(
        **{
            "Total Count": ("Distance", "size"),
            "Count < cutoff": ("IsClose", "sum"),
        }
    ).reset_index()

    out["Fraction Close"] = out["Count < cutoff"] / out["Total Count"]
    out["Cutoff (µm)"] = float(cutoff)
    return out

# -----------------------------
# Build df_all (clusters)
# -----------------------------

def _pair_key(p1: str, p2: str) -> str:
    return f"{p1}-{p2}"


def _find_anchor(proteins: List[str], directed_pairs: List[Tuple[str, str]]) -> str:
    """Pick a protein that has outgoing edges to all others if possible; otherwise best out-degree."""
    out = {p: 0 for p in proteins}
    out_edges = {p: set() for p in proteins}
    for a, b in directed_pairs:
        if a in out:
            out[a] += 1
            out_edges[a].add(b)

    # Prefer full coverage
    for p in proteins:
        if out_edges[p] >= (set(proteins) - {p}):
            return p

    # Otherwise, max out-degree
    return max(proteins, key=lambda p: out[p])


def build_df_all(
    df_actual: pd.DataFrame,
    number_of_proteins: int,
    distance_cutoff: float = 0.3,
    reference_pair: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build df_all where each row represents a multi-protein cluster and each pair has its own df# columns.

    distance_cutoff: keeps only pair-rows with Actual_Min_Distance <= cutoff before forming clusters.
    reference_pair: if provided and present, becomes df1 in the output column ordering.
    """
    if number_of_proteins not in (3, 4):
        raise ValueError("number_of_proteins must be 3 or 4")

    needed_cols = {"Object1", "Object2", "SpotIndex1", "ClosestSpotIndex2", "Actual_Min_Distance"}
    if not needed_cols.issubset(df_actual.columns):
        raise ValueError(f"df_actual missing required columns: {sorted(needed_cols - set(df_actual.columns))}")

    # proteins in this file
    proteins = sorted(set(df_actual["Object1"].tolist() + df_actual["Object2"].tolist()) - {""})

    if len(proteins) < number_of_proteins:
        raise ValueError(f"Found only {len(proteins)} proteins in CSV, expected {number_of_proteins}: {proteins}")

    # If more than expected proteins exist, keep the most frequent ones
    if len(proteins) > number_of_proteins:
        counts = (
            pd.concat([df_actual["Object1"], df_actual["Object2"]], ignore_index=True)
            .value_counts()
        )
        proteins = counts.index.tolist()[:number_of_proteins]
        proteins = sorted(proteins)

    # create per-pair dfs (directed, as in file)
    pair_rows = df_actual[["Object1", "Object2"]].drop_duplicates()
    directed_pairs = [(r["Object1"], r["Object2"]) for _, r in pair_rows.iterrows() if r["Object1"] in proteins and r["Object2"] in proteins]

    # Filter to the complete pair set among selected proteins
    pair_dfs: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (p1, p2) in directed_pairs:
        dfp = df_actual[(df_actual["Object1"] == p1) & (df_actual["Object2"] == p2)].copy()
        dfp = dfp[dfp["Actual_Min_Distance"] <= float(distance_cutoff)].copy()
        pair_dfs[(p1, p2)] = dfp[["SpotIndex1", "ClosestSpotIndex2", "Actual_Min_Distance"]].rename(
            columns={
                "SpotIndex1": f"{p1}_idx",
                "ClosestSpotIndex2": f"{p2}_idx",
                "Actual_Min_Distance": f"dist_{p1}_{p2}",
            }
        )

    if not pair_dfs:
        return pd.DataFrame()

    # Choose anchor that can link to all others
    anchor = _find_anchor(proteins, directed_pairs)
    others = [p for p in proteins if p != anchor]

    # Anchor must have outgoing edges to all others to build an initial tuple
    missing = [p for p in others if (anchor, p) not in pair_dfs]
    if missing:
        # Try another anchor that has full outgoing edges
        for cand in proteins:
            if cand == anchor:
                continue
            cand_missing = [p for p in proteins if p != cand and (cand, p) not in pair_dfs]
            if not cand_missing:
                anchor = cand
                others = [p for p in proteins if p != anchor]
                missing = []
                break

    if missing:
        # If we can't build tuples, return empty (better than wrong merges)
        return pd.DataFrame()

    # Build initial clusters from anchor->other mappings
    df_clusters = pair_dfs[(anchor, others[0])].copy()
    for p in others[1:]:
        df_clusters = df_clusters.merge(pair_dfs[(anchor, p)], on=f"{anchor}_idx", how="inner")

    # Enforce consistency across remaining pairs among others
    # (use whichever directed version exists)
    for p1, p2 in itertools.combinations(others, 2):
        if (p1, p2) in pair_dfs:
            df_clusters = df_clusters.merge(pair_dfs[(p1, p2)], on=[f"{p1}_idx", f"{p2}_idx"], how="inner")
        elif (p2, p1) in pair_dfs:
            df_clusters = df_clusters.merge(pair_dfs[(p2, p1)], on=[f"{p2}_idx", f"{p1}_idx"], how="inner")
        else:
            # if the pair is missing in file, we can't enforce it
            pass

    if df_clusters.empty:
        return pd.DataFrame()

    # Determine output pair ordering (only pairs that exist in file among selected proteins)
    pair_list = sorted(pair_dfs.keys(), key=lambda t: _pair_key(t[0], t[1]))
    if reference_pair:
        reference_pair = reference_pair.strip()
        # accept either direction
        ref_tuple = None
        for t in pair_list:
            if _pair_key(*t) == reference_pair:
                ref_tuple = t
                break
        if ref_tuple:
            pair_list = [ref_tuple] + [t for t in pair_list if t != ref_tuple]

    # Build df_all with df1..dfN columns
    out = pd.DataFrame()
    for i, (p1, p2) in enumerate(pair_list, start=1):
        out[f"Object1_df{i}"] = [p1] * len(df_clusters)
        out[f"Object2_df{i}"] = [p2] * len(df_clusters)
        out[f"SpotIndex_df{i}"] = df_clusters[f"{p1}_idx"].astype(int)
        out[f"ClosestSpotIndex_df{i}"] = df_clusters[f"{p2}_idx"].astype(int)

        dist_col = f"dist_{p1}_{p2}"
        if dist_col in df_clusters.columns:
            out[f"Actual_Min_Distance_df{i}"] = df_clusters[dist_col].astype(float)
        else:
            # if distance came from reversed directed pair merge
            # (e.g. we enforced (p2,p1)), try that:
            alt = f"dist_{p2}_{p1}"
            out[f"Actual_Min_Distance_df{i}"] = df_clusters.get(alt, np.nan)

    return out.reset_index(drop=True)


# -----------------------------
# Alignment filter
# -----------------------------

def _distance_col_for_pair(df_all: pd.DataFrame, p1: str, p2: str) -> Optional[str]:
    """Return the distance column name (Actual_Min_Distance_dfX) for a pair in either direction."""
    for col in df_all.columns:
        if not col.startswith("Object1_df"):
            continue
        i = col.split("_df")[-1]
        o1 = df_all[f"Object1_df{i}"].iloc[0]
        o2 = df_all[f"Object2_df{i}"].iloc[0]
        if (o1 == p1 and o2 == p2) or (o1 == p2 and o2 == p1):
            dist_col = f"Actual_Min_Distance_df{i}"
            if dist_col in df_all.columns:
                return dist_col
    return None




def normalize_reference_pairs(reference_pair: Optional[str]) -> List[str]:
    """
    Accept a single reference pair like "ProtA-ProtB" OR a composite like:
      - "ProtA-ProtB OR ProtC-ProtD"
      - "ProtA-ProtB|ProtC-ProtD"
    Returns a de-duplicated list preserving order.
    """
    if not reference_pair:
        return []
    s = str(reference_pair).strip()
    if not s:
        return []
    # Split on ' OR ' (case-insensitive) and also allow '|' as a delimiter.
    parts = [p.strip() for p in re.split(r"\s+OR\s+", s, flags=re.IGNORECASE) if p.strip()]
    out: List[str] = []
    for p in parts:
        for q in [x.strip() for x in p.split("|") if x.strip()]:
            if q not in out:
                out.append(q)
    return out

def apply_alignment_filter(
    df_all: pd.DataFrame,
    reference_pair: str,
    protein_locations: Dict[str, str],
    tolerance_frac: float,
) -> pd.DataFrame:
    """
    Keep rows where intermediate proteins (excluding endpoint locations) satisfy:
      dist(A,B) + dist(B,C) ≈ dist(A,C) within tolerance.

    If multiple proteins exist in the same compartment (e.g., two postsynaptic membrane proteins),
    ALL proteins in that compartment must satisfy the condition (AND rule).
    """
    if df_all.empty:
        return df_all

    if not reference_pair or "-" not in reference_pair:
        raise ValueError("reference_pair must look like 'ProtA-ProtB'")

    A, C = [s.strip() for s in reference_pair.split("-", 1)]
    if A not in protein_locations or C not in protein_locations:
        raise ValueError("Protein locations missing for reference pair endpoints.")

    distAC = _distance_col_for_pair(df_all, A, C)
    if distAC is None:
        raise ValueError(f"Reference pair {reference_pair} not found in df_all")

    locA = protein_locations.get(A, "")
    locC = protein_locations.get(C, "")

    # Discover proteins present
    obj_cols = [c for c in df_all.columns if c.startswith("Object1_df")]
    proteins_in_df = sorted(set(pd.unique(df_all[obj_cols].values.ravel())))
    proteins_in_df = [p for p in proteins_in_df if isinstance(p, str) and p]

    # Candidates: exclude endpoints AND exclude same-location as endpoints
    candidates = []
    for p in proteins_in_df:
        if p in (A, C):
            continue
        locp = protein_locations.get(p, "")
        if locp and locp not in (locA, locC):
            candidates.append(p)

    if not candidates:
        return df_all

    # Group candidates by location (so membrane proteins are checked together)
    candidates_by_loc: Dict[str, List[str]] = {}
    for p in candidates:
        loc = protein_locations.get(p, "")
        if loc:
            candidates_by_loc.setdefault(loc, []).append(p)

    dAC = df_all[distAC].astype(float)
    lo = dAC * (1.0 - float(tolerance_frac))
    hi = dAC * (1.0 + float(tolerance_frac))

    overall_mask = pd.Series(True, index=df_all.index)
    intermediates_used: List[str] = []

    # For each location group: require ALL proteins in that group to align
    for loc, prots in candidates_by_loc.items():
        group_mask = pd.Series(True, index=df_all.index)

        for B in prots:
            distAB = _distance_col_for_pair(df_all, A, B)
            distBC = _distance_col_for_pair(df_all, B, C)
            if distAB is None or distBC is None:
                continue  # can't test this protein, so don't constrain on it

            dsum = df_all[distAB].astype(float) + df_all[distBC].astype(float)
            group_mask &= (dsum >= lo) & (dsum <= hi)
            intermediates_used.append(B)

        overall_mask &= group_mask  # AND across locations too

    result = df_all.loc[overall_mask].copy()
    if result.empty:
        return pd.DataFrame()

    result["Alignment_Intermediates"] = ", ".join(sorted(set(intermediates_used)))

    # Deduplicate by spot identity
    spot_cols = [c for c in result.columns if c.startswith("SpotIndex_df") or c.startswith("ClosestSpotIndex_df")]
    if spot_cols:
        result = result.drop_duplicates(subset=spot_cols)

    return result.reset_index(drop=True)


# -----------------------------
# Angle filter (optional)
# -----------------------------

def apply_angle_filter(
    df_all: pd.DataFrame,
    protein_A: str,
    protein_B: str,
    protein_C: str,
    angle_a: float,
    angle_b: float,
    angle_c: float,
) -> pd.DataFrame:
    """Compute triangle angles from distances and filter by max angles."""
    if df_all.empty:
        return df_all

    distAB = _distance_col_for_pair(df_all, protein_A, protein_B)
    distBC = _distance_col_for_pair(df_all, protein_B, protein_C)
    distAC = _distance_col_for_pair(df_all, protein_A, protein_C)

    if not distAB or not distBC or not distAC:
        raise ValueError("Could not find distance columns for chosen proteins in df_all.")

    AB = df_all[distAB].astype(float)
    BC = df_all[distBC].astype(float)
    AC = df_all[distAC].astype(float)

    # Cosine law
    Aang = np.degrees(np.arccos(np.clip((AB**2 + AC**2 - BC**2) / (2 * AB * AC), -1, 1)))
    Bang = np.degrees(np.arccos(np.clip((AB**2 + BC**2 - AC**2) / (2 * AB * BC), -1, 1)))
    Cang = np.degrees(np.arccos(np.clip((AC**2 + BC**2 - AB**2) / (2 * AC * BC), -1, 1)))

    out = df_all.copy()
    out["Angle_at_A"] = Aang
    out["Angle_at_B"] = Bang
    out["Angle_at_C"] = Cang

    out = out[
        (out["Angle_at_A"] <= float(angle_a)) &
        (out["Angle_at_B"] <= float(angle_b)) &
        (out["Angle_at_C"] <= float(angle_c))
    ].reset_index(drop=True)

    return out


# -----------------------------
# Main run entry point
# -----------------------------

def run_analysis(
    data_folder: str,
    expansion_factor_file: str,
    number_of_proteins: int,
    *,
    use_alignment_analysis: bool = False,
    reference_pair: Optional[str] = None,
    protein_locations: Optional[Dict[str, str]] = None,
    alignment_tolerance: float = 0.1,
    use_angle_analysis: bool = False,
    protein_A: Optional[str] = None,
    protein_B: Optional[str] = None,
    protein_C: Optional[str] = None,
    angle_a: float = 180.0,
    angle_b: float = 180.0,
    angle_c: float = 180.0,
    use_enrichment_analysis: bool = False,
    enrichment_cutoff: Optional[float] = None,
    distance_cutoff: float = 0.3,
    results_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Run analysis for all CSVs in a folder.

    Returns dict with:
      - results_dir
      - summary_path
      - (optional) enrichment_summary_path
    """
    if not data_folder or not os.path.isdir(data_folder):
        raise ValueError("Please select a valid data folder.")
    if not expansion_factor_file or not os.path.isfile(expansion_factor_file):
        raise ValueError("Please select a valid expansion factor .xlsx file.")
    if number_of_proteins not in (3, 4):
        raise ValueError("Number of proteins must be 3 or 4.")

    # Decide results directory
    if results_dir is None:
        results_dir = os.path.join(data_folder, "results")
    safe_mkdir(results_dir)

    df_exp = load_expansion_factors(expansion_factor_file)

    data_files = [f for f in os.listdir(data_folder) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    data_files = sorted(data_files)
    if not data_files:
        raise ValueError("No CSV/XLSX files found in the selected data folder.")

    summary_rows = []
    all_enrichment_dfs = []
    all_filtered = []
    all_clusters = []

    for csv_name in data_files:
        csv_path = os.path.join(data_folder, csv_name)

        mouse, slice_id, round_id = parse_mouse_slice_round(csv_name)
        exp_factor = get_expansion_factor(df_exp, mouse, slice_id, round_id)
        stem = os.path.splitext(os.path.basename(csv_name))[0]  # filename without extension
        m = re.search(r"_3D_(\d+)", stem, flags=re.IGNORECASE)
        rep = f"3D_{m.group(1)}" if m else "3D"

        base_id = f"{mouse}_{slice_id}_{round_id}_{rep}"


        df_raw = load_spots_distances(csv_path)
        df_actual = apply_expansion_factor(df_raw, exp_factor)
        # enrichment summary (optional; based on RAW pairs, not clusters)
        enrichment_path = ""
        if use_enrichment_analysis:
            cutoff = float(enrichment_cutoff) if enrichment_cutoff is not None else float(distance_cutoff)
            df_enrichment = compute_pair_enrichment(df_actual, cutoff=cutoff)

            # add file context columns
            df_enrichment.insert(0, "ExpansionFactor", exp_factor)
            df_enrichment.insert(0, "Round", round_id)
            df_enrichment.insert(0, "Slice", slice_id)
            df_enrichment.insert(0, "Mouse", mouse)
            df_enrichment.insert(0, "File", csv_name)

            enrichment_path = safe_save_excel(df_enrichment, os.path.join(results_dir, f"{base_id}_enrichment.xlsx"))
            all_enrichment_dfs.append(df_enrichment)


        # build clusters (df_all)
        df_all = build_df_all(
            df_actual=df_actual,
            number_of_proteins=number_of_proteins,
            distance_cutoff=distance_cutoff,
            reference_pair=(normalize_reference_pairs(reference_pair)[0] if (use_alignment_analysis and normalize_reference_pairs(reference_pair)) else None),
        )
        df_all_out = df_all.copy()
        df_all_out.insert(0, "ExpansionFactor", exp_factor)
        df_all_out.insert(0, "Round", round_id)
        df_all_out.insert(0, "Slice", slice_id)
        df_all_out.insert(0, "Mouse", mouse)
        df_all_out.insert(0, "File", csv_name)
        all_clusters.append(df_all_out)

        # Save clusters (pre-filters)
        # base_id already defined above
        clusters_path = safe_save_excel(df_all, os.path.join(results_dir, f"{base_id}_clusters.xlsx"))

        df_filtered = df_all


        # alignment (ONLY if toggled)
        if use_alignment_analysis:
            if not reference_pair:
                raise ValueError("Alignment analysis is ON but no reference pair was selected.")
            if not protein_locations:
                raise ValueError("Alignment analysis is ON but no protein locations were provided.")

            ref_pairs = normalize_reference_pairs(reference_pair)
            if not ref_pairs:
                raise ValueError("Alignment analysis is ON but no valid reference pair(s) were provided.")

            kept_chunks = []
            for rp in ref_pairs:
                tmp = apply_alignment_filter(
                    df_filtered,
                    reference_pair=rp,
                    protein_locations=protein_locations,
                    tolerance_frac=float(alignment_tolerance),
                )
                if not tmp.empty:
                    tmp["Reference_Pair"] = rp
                    kept_chunks.append(tmp)

            if not kept_chunks:
                df_filtered = pd.DataFrame()
            else:
                df_filtered = pd.concat(kept_chunks, ignore_index=True)

                # de-dup by all SpotIndex columns (cluster identity)
                spot_cols = [c for c in df_filtered.columns if c.startswith("SpotIndex_df") or c.startswith("ClosestSpotIndex_df")]
                if spot_cols:
                    df_filtered = df_filtered.drop_duplicates(subset=spot_cols)

                df_filtered = df_filtered.reset_index(drop=True)

        # angle filter (ONLY if toggled)
        if use_angle_analysis:
            if not (protein_A and protein_B and protein_C):
                raise ValueError("Angle analysis is ON but Protein A/B/C were not provided.")
            df_filtered = apply_angle_filter(
                df_filtered,
                protein_A=protein_A,
                protein_B=protein_B,
                protein_C=protein_C,
                angle_a=angle_a,
                angle_b=angle_b,
                angle_c=angle_c,
            )
        if df_filtered is not None and not df_filtered.empty:
            df_filtered_out = df_filtered.copy()
            df_filtered_out.insert(0, "ExpansionFactor", exp_factor)
            df_filtered_out.insert(0, "Round", round_id)
            df_filtered_out.insert(0, "Slice", slice_id)
            df_filtered_out.insert(0, "Mouse", mouse)
            df_filtered_out.insert(0, "File", csv_name)
            all_filtered.append(df_filtered_out)

        filtered_path = safe_save_excel(df_filtered, os.path.join(results_dir, f"{base_id}_filtered.xlsx"))

        summary_rows.append({
            "File": csv_name,
            "Mouse": mouse,
            "Slice": slice_id,
            "Round": round_id,
            "ExpansionFactor": exp_factor,
            "ClustersFound": int(len(df_all)),
            "ClustersAfterFilters": int(len(df_filtered)),
            "ClustersExcel": clusters_path,
            "FilteredExcel": filtered_path,
            "EnrichmentExcel": enrichment_path,
        })

    # --- Combined outputs (all files) ---
    if all_clusters:
        df_all_clusters = pd.concat(all_clusters, ignore_index=True)
        safe_save_excel(df_all_clusters, os.path.join(results_dir, "ALL_clusters.xlsx"))

    if all_filtered:
        df_all_filtered = pd.concat(all_filtered, ignore_index=True)
        safe_save_excel(df_all_filtered, os.path.join(results_dir, "ALL_filtered.xlsx"))
    else:
        safe_save_excel(pd.DataFrame(), os.path.join(results_dir, "ALL_filtered.xlsx"))

    df_summary = pd.DataFrame(summary_rows)
    summary_path = safe_save_excel(df_summary, os.path.join(results_dir, "summary.xlsx"))

    enrichment_summary_path = ""
    if use_enrichment_analysis and all_enrichment_dfs:
        df_all_enrichment = pd.concat(all_enrichment_dfs, ignore_index=True)
        enrichment_summary_path = safe_save_excel(df_all_enrichment, os.path.join(results_dir, "enrichment_summary.xlsx"))

    out = {"results_dir": results_dir, "summary_path": summary_path}
    if enrichment_summary_path:
        out["enrichment_summary_path"] = enrichment_summary_path
    return out
