"""Panel construction: two ways of picking ~100 personas to stand in for the
whole PLP, plus the honest measurement of what that substitution costs.

F1 (stratified + raked) and F2 (Gower/k-medoids) are both built only from
``schemas.SEG_VARS``. ``frame_error`` then checks the weighted panel against
the population on ``schemas.HELDOUT_VARS`` -- variables neither frame ever
saw -- which is what makes the resulting number an honest upper bound on any
persona method's achievable accuracy, rather than a number the construction
was allowed to optimise against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from kmedoids import KMedoids
from scipy.spatial.distance import cdist

from plp_sim import schemas
from plp_sim.config import Settings

#: Below this population count, a stratum cannot support a meaningful
#: proportional draw -- allocation would round to 0 or 1 almost every time,
#, and a "margin" made of one member is not a margin. Merge such strata into
#: a coarser bucket instead of erroring or drawing from a group of one.
MIN_STRATUM_SIZE = 8

#: Quantile bins used to turn the three continuous SEG_VARS into categories
#: for raking. Kept small (terciles) so the total contingency across all six
#: SEG_VARS stays something a ~100-row panel can plausibly cover -- more bins
#: would make an empty cell (and therefore an unsatisfiable margin) the
#: normal outcome rather than the edge case it should be.
RAKE_QUANTILE_BINS = 3

#: Population categories smaller than this are collapsed before raking. IPF
#: cannot scale zero mass up, so a category the sample is likely to miss makes
#: the margins unsatisfiable. Mirrors MIN_STRATUM_SIZE, which already guards
#: the stratum cells.
MIN_RAKE_CELL = 8
RARE_CATEGORY_LABEL = "__rare__"


def _continuous_seg_vars() -> tuple[str, ...]:
    """The non-categorical SEG_VARS, derived from the schema rather than typed
    out here.

    This function exists because the literal it replaces
    (``("majority_pct", "rebellion_rate", "nomination_day")``) silently
    included ``nomination_day`` -- a POST_CUTOFF_VAR, i.e. the outcome. F1 was
    therefore raking the panel to match the true nomination split, reproducing
    it to ~5e-8 by construction, and every distribution-accuracy number for F1
    on that event was meaningless. schemas.py fences those columns off and
    test_schemas asserts the tuples are disjoint -- but nothing checked what
    this module actually consumed at runtime, so the fence was decorative.
    Deriving from the schema is the only version of this that cannot drift.
    """
    return tuple(c for c in schemas.SEG_VARS if c not in schemas.CATEGORICAL_SEG_VARS)

#: Bin count for the population-quantile binning frame_error's TVD is computed
#: on. Deciles are the standard granularity for this kind of goodness-of-fit
#: check: fine enough to catch a shifted or reshaped distribution, coarse
#: enough that a ~100-member weighted panel still puts real weight in every
#: bin instead of being dominated by empty-bin noise.
HELDOUT_QUANTILE_BINS = 10


# --------------------------------------------------------------------------
# Gower distance
# --------------------------------------------------------------------------


def gower_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Gower distance over ``columns`` of ``df``.

    Numeric columns contribute range-normalised absolute difference; object
    and bool (categorical) columns contribute a 0/1 mismatch. Distances are
    averaged across ``columns`` per pair -- but a column is only counted for a
    pair if *both* rows have a non-null value for it, so a null in
    ``nomination_day`` (~15% of rows) excludes just that one feature from that
    pair's average rather than forcing the row out of the panel entirely,
    which would bias the panel toward MPs with a sourced declaration date.

    Returns a symmetric ``(n, n)`` array with a zero diagonal and every entry
    in ``[0, 1]``.
    """
    n = len(df)
    dist_sum = np.zeros((n, n), dtype=float)
    weight_sum = np.zeros((n, n), dtype=float)

    for col in columns:
        s = df[col]
        valid = s.notna().to_numpy()
        pair_valid = valid[:, None] & valid[None, :]

        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values = s.to_numpy(dtype=float)
            span = (np.nanmax(values) - np.nanmin(values)) if valid.any() else 0.0
            if span > 0:
                diff = np.abs(values[:, None] - values[None, :]) / span
            else:
                diff = np.zeros((n, n))
        else:
            values = s.to_numpy()
            diff = (values[:, None] != values[None, :]).astype(float)

        dist_sum += np.where(pair_valid, diff, 0.0)
        weight_sum += pair_valid

    gower = np.divide(
        dist_sum, weight_sum, out=np.zeros_like(dist_sum), where=weight_sum > 0
    )
    gower = (gower + gower.T) / 2.0
    np.fill_diagonal(gower, 0.0)
    return gower


# --------------------------------------------------------------------------
# F1: stratified sample + raking
# --------------------------------------------------------------------------


def _majority_tercile(s: pd.Series) -> pd.Series:
    cats = pd.qcut(s, 3, duplicates="drop")
    labels = [f"majority_q{i + 1}" for i in range(len(cats.cat.categories))]
    return cats.cat.rename_categories(labels).astype(str)


def _stratum_labels(df: pd.DataFrame, min_size: int = MIN_STRATUM_SIZE) -> pd.Series:
    """Cross majority tercile x runner_up_party x is_payroll x is_2024_intake,
    collapsing to a coarser key wherever a cell has fewer than ``min_size``
    members.

    Collapse order: drop ``runner_up_party`` first (rare parties are the usual
    cause of a thin cell), then drop the majority tercile too, then fall back
    to one global bucket -- which by construction always has >= min_size
    members, so this always terminates without erroring. Only the *label*
    used to allocate and draw the sample changes; each row's own attributes
    (used later for raking) are untouched.
    """
    tercile = _majority_tercile(df["majority_pct"])
    payroll = pd.Series(np.where(df["is_payroll"], "payroll", "backbench"), index=df.index)
    intake = pd.Series(np.where(df["is_2024_intake"], "2024intake", "pre2024"), index=df.index)
    runner = df["runner_up_party"].astype(str)

    coarser_levels = [
        tercile + "|" + payroll + "|" + intake,
        tercile.copy(),
        pd.Series("all_members", index=df.index),
    ]
    stratum = tercile + "|" + runner + "|" + payroll + "|" + intake
    for coarser in coarser_levels:
        counts = stratum.value_counts()
        too_small = stratum.map(counts) < min_size
        if not too_small.any():
            break
        stratum = stratum.where(~too_small, coarser)
    return stratum


def _proportional_allocation(sizes: pd.Series, total: int) -> pd.Series:
    """Largest-remainder (Hamilton) apportionment of ``total`` across strata
    proportional to ``sizes``, capped so no stratum is ever asked for more
    members than it actually has.
    """
    population = int(sizes.sum())
    exact = sizes * total / population
    alloc = np.floor(exact).astype(int)
    frac_order = (exact - alloc).sort_values(ascending=False).index

    remainder = total - int(alloc.sum())
    for label in frac_order[:remainder]:
        alloc[label] += 1

    capped = pd.Series(np.minimum(alloc.to_numpy(), sizes.to_numpy()), index=sizes.index)
    shortfall = total - int(capped.sum())
    if shortfall > 0:
        capacity = sizes - capped
        for label in frac_order:
            if shortfall <= 0:
                break
            take = min(shortfall, int(capacity[label]))
            capped[label] += take
            shortfall -= take
    return capped


def _rake_bin_edges(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Population-derived quantile bin edges for the continuous SEG_VARS,
    open-ended at both tails so a sample value outside the population's
    observed range still lands in the extreme bin instead of falling out of
    every bin.
    """
    edges: dict[str, np.ndarray] = {}
    for col in _continuous_seg_vars():
        values = df[col].dropna()
        _, bins = pd.qcut(values, RAKE_QUANTILE_BINS, retbins=True, duplicates="drop")
        bins = bins.astype(float).copy()
        bins[0] = -np.inf
        bins[-1] = np.inf
        edges[col] = bins
    return edges


def _seg_var_categories(df: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """SEG_VARS turned into raking categories. Continuous vars are cut on
    ``edges`` (computed once, from the population, so the sample is binned on
    the exact same boundaries). ``nomination_day``'s ~15% missingness becomes
    its own category rather than being dropped, so the panel is raked to
    match how much of the population has *no* declaration date too.
    """
    out = pd.DataFrame(index=df.index)
    for col in _continuous_seg_vars():
        binned = pd.cut(df[col], bins=edges[col], include_lowest=True).astype(str)
        if df[col].isna().any():
            binned = binned.where(df[col].notna(), "missing")
        out[col] = binned
    for col in schemas.CATEGORICAL_SEG_VARS:
        out[col] = df[col].astype(str)
    return out


def collapse_rare_categories(
    pop_categories: pd.DataFrame, min_size: int = MIN_RAKE_CELL
) -> pd.DataFrame:
    """Fold population categories below ``min_size`` into a shared bucket.

    IPF cannot scale zero mass up. A category with 3 members out of 405 is
    expected to draw 0.7 members into a 100-person sample, so it is usually
    absent, and then its target of 3 is permanently unreachable, the loop
    plateaus at a discrepancy of exactly 3, and (correctly) raises. On the real
    table that is not an edge case: ``runner_up_party`` has Other=3, Plaid
    Cymru=4, Liberal Democrat=6, and ``build_f1`` fails on 27 of 30 seeds.

    ``_stratum_labels`` already guards its own cells this way via
    ``MIN_STRATUM_SIZE``; the raking margins were simply never given the same
    protection. Collapsing loses the Plaid/LD/Other distinction in the margins,
    which is an acceptable price: those are 13 MPs between them: where
    silently dropping the frame entirely is not.
    """
    out = pop_categories.copy()
    for col in out.columns:
        counts = out[col].value_counts()
        rare = set(counts[counts < min_size].index)
        if not rare:
            continue
        pooled = int(counts[list(rare)].sum())
        if pooled >= min_size:
            # Enough rare categories to form a viable bucket of their own.
            out[col] = out[col].where(~out[col].isin(rare), RARE_CATEGORY_LABEL)
        else:
            # Not enough. Relabelling a lone 1-member category as "__rare__"
            # leaves a 1-member category with a different name and the margin
            # is still unsatisfiable -- which is precisely what a null
            # rebellion_rate does: exactly one MP (the by-election winner, who
            # has no pre-cutoff votes) forms a "missing" cell of size 1 and
            # every seed fails with a discrepancy of exactly 1. Absorb into the
            # largest category instead, so no undersized cell survives at all.
            largest = counts.idxmax()
            out[col] = out[col].where(~out[col].isin(rare), largest)
    return out


def rake(
    categories: pd.DataFrame,
    targets: dict[str, pd.Series],
    weights: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Iterative proportional fitting.

    ``categories[col]`` gives each row's category for raking variable ``col``;
    ``targets[col]`` maps each category to its population total on the weight
    scale. Each sweep rescales every category of every variable in turn to hit its
    target.

    Raises if a sample category has no target. Such a category is never rescaled,
    keeps its initial weight forever, and the loop would report convergence with
    that weight arbitrarily wrong. Likewise a population category the sample never
    sampled cannot be rescaled into existence, so its discrepancy never clears.

    Convergence is checked after a full sweep, not per variable: rescaling one
    variable perturbs the margins of the last, so only the all-margins-at-once
    discrepancy is a real signal. Raises with the achieved discrepancy if
    ``max_iter`` sweeps are not enough.
    """
    for col, target_counts in targets.items():
        untargeted = sorted(set(categories[col].unique()) - set(target_counts.index))
        if untargeted:
            raise ValueError(
                f"raking variable {col!r} has sample categories with no target: "
                f"{untargeted}. Such rows are never rescaled, keep their initial "
                f"weight, and the loop would then report convergence with those "
                f"weights arbitrarily wrong."
            )

    weights = weights.astype(float).copy()
    achieved = float("inf")
    for _iteration in range(1, max_iter + 1):
        for col, target_counts in targets.items():
            cats = categories[col].to_numpy()
            for category, target in target_counts.items():
                mask = cats == category
                current = weights[mask].sum()
                if current > 0:
                    weights[mask] *= target / current

        discrepancies = []
        for col, target_counts in targets.items():
            cats = categories[col].to_numpy()
            for category, target in target_counts.items():
                discrepancies.append(abs(weights[cats == category].sum() - target))
        achieved = max(discrepancies) if discrepancies else 0.0
        if achieved <= tol:
            return weights

    raise ValueError(
        f"raking did not converge in {max_iter} iterations: "
        f"max margin discrepancy {achieved:.6g} is still above tolerance {tol:.2g}"
    )


def build_f1(attributes: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Stratified sample across SEG_VAR-derived strata (proportional
    allocation, seeded), then raked so the weighted panel reproduces the
    population's SEG_VAR margins to within ``settings.rake_tolerance``.
    """
    df = attributes.reset_index(drop=True)
    n_pop = len(df)

    stratum = _stratum_labels(df)
    sizes = stratum.value_counts()
    alloc = _proportional_allocation(sizes, settings.n_personas)

    rng = np.random.default_rng(settings.random_seed)
    design_weight = np.zeros(n_pop, dtype=float)
    sampled_positions: list[int] = []
    for label in sorted(alloc.index):
        k = int(alloc[label])
        if k <= 0:
            continue
        group_positions = np.flatnonzero((stratum == label).to_numpy())
        chosen = rng.choice(group_positions, size=k, replace=False)
        design_weight[chosen] = sizes[label] / k
        sampled_positions.extend(chosen.tolist())
    sampled_positions_arr = np.array(sorted(sampled_positions))

    edges = _rake_bin_edges(df)
    # Collapse before deriving targets, so the target set and the sample's
    # categories are built from the same collapsed labels. Without this, real
    # data fails on 27 of 30 seeds: a 3-member category is usually absent from
    # a 100-person sample and its margin is then permanently unsatisfiable.
    pop_categories = collapse_rare_categories(_seg_var_categories(df, edges))
    targets = {col: pop_categories[col].value_counts() for col in pop_categories.columns}

    sample_categories = pop_categories.iloc[sampled_positions_arr].reset_index(drop=True)
    initial_weights = design_weight[sampled_positions_arr]

    final_weights = rake(
        sample_categories,
        targets,
        initial_weights,
        max_iter=settings.rake_max_iter,
        tol=settings.rake_tolerance,
    )

    result = pd.DataFrame(
        {
            "frame": "F1",
            "member_id": df.loc[sampled_positions_arr, "member_id"].to_numpy(),
            "weight": final_weights,
            "stratum": stratum.iloc[sampled_positions_arr].to_numpy(),
        }
    )
    return schemas.validate(result, schemas.FRAME)


# --------------------------------------------------------------------------
# F2: Gower + k-medoids
# --------------------------------------------------------------------------


def build_f2(attributes: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """k-medoids (k = ``settings.n_personas``) over the Gower distance on
    SEG_VARS. Each medoid is a real member, weighted by the size of the
    cluster it represents.
    """
    df = attributes.reset_index(drop=True)
    distance = gower_matrix(df, list(schemas.SEG_VARS))

    km = KMedoids(
        settings.n_personas,
        metric="precomputed",
        method="fasterpam",
        random_state=settings.random_seed,
    ).fit(distance)

    medoid_positions = km.medoid_indices_
    cluster_sizes = np.bincount(km.labels_, minlength=settings.n_personas).astype(float)

    result = pd.DataFrame(
        {
            "frame": "F2",
            "member_id": df.loc[medoid_positions, "member_id"].to_numpy(),
            "weight": cluster_sizes,
            "stratum": [None] * settings.n_personas,
        }
    )
    return schemas.validate(result, schemas.FRAME)


# --------------------------------------------------------------------------
# frame_error: the headline measurement
# --------------------------------------------------------------------------


def _tvd(population: pd.Series, frame_values: pd.Series, frame_weights: pd.Series) -> float:
    """Total variation distance between the population's distribution of one
    HELDOUT_VAR and the weighted panel's, on a shared binning of population
    quantiles (see ``HELDOUT_QUANTILE_BINS``). Rows missing a value for this
    specific variable are dropped from whichever side has the gap.
    """
    pop = population.dropna()
    present = frame_values.notna()
    fv, fw = frame_values[present], frame_weights[present]

    _, edges = pd.qcut(pop, HELDOUT_QUANTILE_BINS, retbins=True, duplicates="drop")
    edges = edges.astype(float).copy()
    edges[0], edges[-1] = -np.inf, np.inf

    pop_binned = pd.cut(pop, bins=edges, include_lowest=True)
    frame_binned = pd.cut(fv, bins=edges, include_lowest=True)

    pop_share = pop_binned.value_counts(normalize=True, sort=False)
    frame_weighted = fw.groupby(frame_binned, observed=False).sum()
    frame_share = (frame_weighted / frame_weighted.sum()).reindex(pop_share.index, fill_value=0.0)

    return float(0.5 * (pop_share - frame_share).abs().sum())


def _energy_distance(x: np.ndarray, wx: np.ndarray, y: np.ndarray, wy: np.ndarray) -> float:
    """Weighted two-sample energy distance (Szekely & Rizzo), generalised
    from equal-mass points to weighted ones by weighting each pairwise
    distance term instead of assuming one row = one unit of mass.

    Chosen over a Frobenius gap between correlation matrices because it is a
    proper distance between the *distributions* themselves: sensitive to
    shifts in location and spread (a correlation matrix is blind to both),
    not only to how the variables co-move. It also has a natural weighted
    form, which a correlation-matrix comparison does not get for free -- the
    panel is a weighted set of real members, not an unweighted resample.
    """
    dxy, dxx, dyy = cdist(x, y), cdist(x, x), cdist(y, y)
    term_xy = float((wx[:, None] * wy[None, :] * dxy).sum())
    term_xx = float((wx[:, None] * wx[None, :] * dxx).sum())
    term_yy = float((wy[:, None] * wy[None, :] * dyy).sum())
    return max(0.0, 2 * term_xy - term_xx - term_yy)


def frame_error(attributes: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """The headline measurement: how much does substituting ``frame`` for the
    full population distort variables the frame construction never saw
    (``schemas.HELDOUT_VARS``)?

    Returns a tidy, plottable frame with one row per HELDOUT_VAR (metric
    ``"tvd"``) plus one ``"multivariate"`` row (metric ``"energy_distance"``).
    Both metrics are zero at a perfect reproduction and increase from there --
    lower is better throughout, unambiguously, in both columns.
    """
    heldout = list(schemas.HELDOUT_VARS)
    merged = frame.merge(
        attributes[["member_id", *heldout]], on="member_id", how="left", validate="many_to_one"
    )

    rows = [
        {"variable": col, "metric": "tvd", "error": _tvd(attributes[col], merged[col], merged["weight"])}
        for col in heldout
    ]

    pop_mv = attributes[heldout].astype("float64").dropna()
    frame_mv = merged.dropna(subset=heldout)
    mean, std = pop_mv.mean(), pop_mv.std(ddof=0).replace(0, 1.0)
    x = ((pop_mv - mean) / std).to_numpy(dtype="float64")
    y = ((frame_mv[heldout].astype("float64") - mean) / std).to_numpy(dtype="float64")
    wx = np.full(len(x), 1.0 / len(x))
    wy = (frame_mv["weight"] / frame_mv["weight"].sum()).to_numpy()
    rows.append(
        {"variable": "multivariate", "metric": "energy_distance", "error": _energy_distance(x, wx, y, wy)}
    )

    return pd.DataFrame(rows)
