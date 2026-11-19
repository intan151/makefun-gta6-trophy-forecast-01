"""gta6-trophy-rarity-forecast-01: forecast Exophase earn rates for GTA VI (PS5).

Model (per trophy, on the logit scale)
---------------------------------------
An Exophase earn rate is a ratio, rate = earners / players_tracked, and both
halves move: the tracked base jumps whenever Exophase refreshes its counts
(on launch night it sat at 638 players for ~4.5 hours, then jumped to 3,868
and every rate but the 100% one fell), while earners grow as tracked players
play through the game. Forecasting the ratio directly therefore has to
respect both motions, so this program fits the ratio's motion with two
complementary curves and blends them:

1.  Distinct snapshots.  Consecutive reads that are byte-identical carry no
    new information (Exophase refreshes hours apart, not every read), so the
    history is collapsed to its distinct snapshots, keeping each value's
    first-seen time. Fitting on raw reads would overweight long flat
    stretches and underweight the refreshes that actually move the rates.

2.  Trend branch (>= 3 distinct snapshots, the final case's shape).
    For each trophy, a recency-weighted least-squares line of
    logit(rate) against x = ln(1 + hours since release) (weights halve every
    24 h), read at the target. The slope is shrunk toward its tier mean
    (platinum / gold / silver / bronze), because per-trophy slopes for rare
    trophies are estimated from counts rounded to 2 decimals and are noisy.
    The forecast is anchored at the last observed rate -- only the fitted
    *move* from the last read is used, never the fitted intercept -- so a
    level error in the fit cannot shift the forecast.
    A second, saturating forecast is blended in (weight 0.75): each rate is
    also fitted as r(t) = A - B*exp(-t/T), a curve that rises toward a
    ceiling, with the time constant T chosen once by pooled residual over
    all trophies. The two estimates agree while rates are low and disagree
    exactly where the ceiling matters -- a logit line in ln(1+t) has no
    ceiling and, fitted over days 0-2 when rates rise steeply, can
    extrapolate an early trophy from 40% to 98% by day 7, while the
    saturating curve cannot. Both moves are anchored at the last observed
    rate, so a level error in either fit cannot shift the forecast. The
    final move is the weighted mean of the two (0.75 saturating, 0.25
    log-time): the saturating move is the structurally correct one for a
    ratio of two saturating quantities, and the log-time move is kept as a
    hedge against a world that does not saturate on this horizon.
    The median logit move from the last rate is capped at +/-4.5.

3.  Flat branch (< 3 distinct snapshots, the practice cases' shape).
    No trend is identifiable, so no trend is invented. The honest prior is
    that Exophase refreshes every few hours and a refresh drops every rate
    (newly tracked players hold far fewer trophies), so the median is
    shifted down by 0.5 logit, growing slowly with horizon
    (-0.3 * ln(1 + days)), with a wide spread. This expects a refresh
    without pretending to know when it lands.

4.  Spread.  sigma = sqrt(residual_var + (0.5 * sqrt(days))^2), floored at
    0.6 and capped at 2.0 in the trend branch: the fit's own residual
    variance plus the baseline's random-walk term. In the flat branch
    sigma = sqrt(1.0 + (0.5 * sqrt(days))^2). A 90% interval from this
    spread is meant to cover about 90% of the time.

5.  The 100% trophy.  A trophy every tracked player holds (the opening
    story trophy) is forecast at a median of 99% with sigma 1.0: it can
    only fall as fresh players sync in, and by an unknown amount.

Quantiles are inv_logit(median_logit + z_tau * sigma) with
z = (-1.645, -0.674, 0, 0.674, 1.645), rounded to 4 decimals half-up.

Pure Python 3.13 standard library, deterministic, no network, no random
numbers: the same input bytes always produce the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's value, kept as the diffusion floor).
RW_SIGMA = 0.5
# Flat-history prior: median logit shift for a refresh that may or may not
# land before the target, plus its slow growth with horizon.
FLAT_SHIFT = -0.5
FLAT_SHIFT_PER_LOG_DAY = -0.3
FLAT_SIGMA0 = 1.0
# Trend branch: blend weight of the saturating-curve move (the rest is the
# logit-log-time move). The saturating move is the structurally correct
# one for a ratio of two saturating quantities and was far more accurate
# than the log-time move on every synthetic final tested (mean |move
# error| 0.06-0.21 logit vs 0.43-0.84), so it carries most of the weight;
# the log-time move is kept as a hedge against a world that does not
# saturate on this horizon.
EXP_WEIGHT = 0.75
# Shrinkage strength of a per-trophy slope toward its tier mean, in
# units of (effective) observations.
SHRINK_K = 6.0
# Caps / floors.
MAX_LOGIT_MOVE = 4.5
SIGMA_FLOOR = 0.6
SIGMA_CAP = 2.0
# Fallback medians (percent) for a trophy with no usable history at all.
TIER_PRIOR = {"platinum": 0.1, "gold": 0.3, "silver": 0.5, "bronze": 0.8}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def wls(points):
    """Weighted least squares of y on x; returns (intercept, slope).

    points: list of (x, y, w). Returns (None, None) with < 2 points or no
    x-variance.
    """
    if len(points) < 2:
        return None, None
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None, None
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return None, None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    return my - slope * mx, slope


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0

    # Time origin: the release day's midnight UTC, from the case's game.
    try:
        rel = case["game"]["release_date"]  # "2026-11-19"
        origin = datetime.fromisoformat(rel).replace(tzinfo=timezone.utc)
    except Exception:
        origin = parse_time(history[0]["slot"])
    target_t = parse_time(case["target"])
    x_target = math.log1p(max(0.0, (target_t - origin).total_seconds() / 3600.0))

    # --- collapse to distinct snapshots (first-seen time of each value) ---
    snaps = []  # (time, players_tracked, rates tuple)
    seen = set()
    for h in history:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key in seen:
            continue
        seen.add(key)
        snaps.append((parse_time(h["at"]), h["players_tracked"], h["rates"]))
    snaps.sort(key=lambda s: s[0])
    last_time, last_n, last_rates = snaps[-1]
    x_last = math.log1p(max(0.0, (last_time - origin).total_seconds() / 3600.0))

    # Last non-null rate per trophy (the last read normally has them all).
    last_rate = []
    for i in range(n_tr):
        r = None
        for _, _, rates in reversed(snaps):
            if rates[i] is not None:
                r = rates[i]
                break
        last_rate.append(r)

    rows = []
    if len(snaps) >= 3:
        rows = _trend_branch(
            trophies, snaps, last_rate, origin, x_target, x_last, horizon_days)
    else:
        rows = _flat_branch(
            trophies, last_rate, horizon_days)
    return {"trophies": rows}


def _quantiles(median_logit: float, sigma: float):
    qs = []
    for z in Z:
        q = inv_logit(median_logit + z * sigma)
        q = min(99.9999, max(0.0001, q))
        qs.append(round4(q))
    # Enforce non-decreasing after rounding (rounding can only tie, but be safe).
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def _flat_branch(trophies, last_rate, horizon_days):
    shift = FLAT_SHIFT + FLAT_SHIFT_PER_LOG_DAY * math.log1p(horizon_days)
    sigma = math.sqrt(FLAT_SIGMA0 ** 2 + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
    rows = []
    for i, tr in enumerate(trophies):
        r = last_rate[i]
        if r is None:
            med = logit(TIER_PRIOR.get(tr["tier"], 0.8))
            rows.append({"id": tr["id"], "q": _quantiles(med, 1.5)})
        elif r >= 99.99:
            rows.append({"id": tr["id"], "q": _quantiles(logit(99.0), 1.0)})
        else:
            rows.append({"id": tr["id"], "q": _quantiles(logit(r) + shift, sigma)})
    return rows


def _trend_branch(trophies, snaps, last_rate, origin,
                  x_target, x_last, horizon_days):
    n_tr = len(trophies)
    t_last = snaps[-1][0]

    # Recency weights: halve every 24 h back from the last snapshot.
    weights = []
    for t, _, _ in snaps:
        age_h = max(0.0, (t_last - t).total_seconds() / 3600.0)
        weights.append(0.5 ** (age_h / 24.0))

    # Per-trophy logit-rate fits and residual collection.
    slopes = [None] * n_tr
    resid_sq = []  # (weighted sq residual, weight) pooled across trophies

    for i in range(n_tr):
        pts = []
        for j, (t, n, rates) in enumerate(snaps):
            r = rates[i]
            if r is None:
                continue
            x = math.log1p(max(0.0, (t - origin).total_seconds() / 3600.0))
            pts.append((x, logit(r), weights[j]))
        if len(pts) >= 2:
            a, b = wls(pts)
            if b is not None:
                slopes[i] = b
                for x, y, w in pts:
                    resid_sq.append((w * (y - (a + b * x)) ** 2, w))
        # trophies with < 2 points keep slope None -> tier mean below

    # Pooled residual variance (weighted), with a floor from rounding noise.
    if resid_sq:
        num = sum(v for v, _ in resid_sq)
        den = sum(w for _, w in resid_sq)
        resid_var = num / den if den > 0 else 0.09
    else:
        resid_var = 0.09
    # Rates are printed to 2 decimals, so a logit residual below ~0.05 is
    # mostly rounding; do not let the pool collapse to zero.
    resid_var = max(resid_var, 0.0025)

    # Tier-mean slopes, then shrinkage of each slope toward its tier mean.
    tiers = {}
    for i, tr in enumerate(trophies):
        if slopes[i] is not None:
            tiers.setdefault(tr["tier"], []).append(slopes[i])
    tier_mean = {k: sum(v) / len(v) for k, v in tiers.items()}
    all_slopes = [s for s in slopes if s is not None]
    global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    # Effective observation counts per trophy (distinct non-null points).
    counts_pts = []
    for i in range(n_tr):
        counts_pts.append(sum(1 for _, _, rates in snaps if rates[i] is not None))
    shrunk = []
    for i, tr in enumerate(trophies):
        base = tier_mean.get(tr["tier"], global_mean)
        if slopes[i] is None:
            shrunk.append(base)
        else:
            m = counts_pts[i]
            shrunk.append((m * slopes[i] + SHRINK_K * base) / (m + SHRINK_K))

    sigma = math.sqrt(resid_var + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))

    # --- saturating (exponential-approach) fit, pooled time constant ---
    # A logit line in ln(1+t) has no ceiling: fitted over days 0-2, when
    # rates rise steeply, it can extrapolate an early trophy from 40% to
    # 98% by day 7. But a rate is earners/tracked, and both halves
    # saturate -- the tracked cohort matures and fresh joiners dilute it --
    # so the net rate approaches a ceiling. Fit r(t) = A - B*exp(-t/T)
    # per trophy (linear in exp(-t/T) for fixed T), with T chosen once by
    # pooled residual over all trophies, and use its *move* from the last
    # read, anchored the same way. The final move is the mean of the
    # log-time move and the saturating move: the two disagree exactly
    # where the ceiling matters, which is where the average earns its keep.
    exp_moves = _exp_saturation_moves(
        trophies, snaps, weights, origin, x_target, t_last)

    rows = []
    for i, tr in enumerate(trophies):
        r = last_rate[i]
        if r is None:
            rows.append({"id": tr["id"],
                         "q": _quantiles(logit(TIER_PRIOR.get(tr["tier"], 0.8)), 1.5)})
            continue
        if r >= 99.99:
            rows.append({"id": tr["id"], "q": _quantiles(logit(99.0), 1.0)})
            continue
        med_logit = logit(r)
        # (a) logit-trend move from the last read.
        move = shrunk[i] * (x_target - x_last)
        # (b) saturating-curve move, where it is defined.
        if exp_moves[i] is not None:
            move = (1.0 - EXP_WEIGHT) * move + EXP_WEIGHT * exp_moves[i]
        move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
        rows.append({"id": tr["id"],
                     "q": _quantiles(med_logit + move, sigma)})
    return rows


def _exp_saturation_moves(trophies, snaps, weights, origin,
                          x_target, last_time):
    """Per-trophy logit move implied by r(t) = A - B*exp(-t/T).

    Returns a list of moves (or None where the fit is not usable). T is
    chosen once, by pooled weighted residual over trophies with >= 3
    points. Times are hours since the release origin.
    """
    n_tr = len(trophies)
    t_hours = [max(0.0, (t - origin).total_seconds() / 3600.0)
               for t, _, _ in snaps]
    t_last_h = max(0.0, (last_time - origin).total_seconds() / 3600.0)
    # Target time in hours: recover from x_target = ln(1 + t_target).
    t_tgt_h = math.expm1(x_target)

    series = []  # per trophy: list of (t_h, rate_pct, w)
    for i in range(n_tr):
        pts = []
        for j, (_, _, rates) in enumerate(snaps):
            if rates[i] is not None:
                pts.append((t_hours[j], rates[i], weights[j]))
        series.append(pts)

    grid = [4.0, 8.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0]
    best_T, best_sse = None, None
    for T in grid:
        sse = 0.0
        usable = 0
        for pts in series:
            if len(pts) < 3:
                continue
            a, b = wls([(math.exp(-t / T), r, w) for t, r, w in pts])
            if a is None:
                continue
            usable += 1
            for t, r, w in pts:
                sse += w * (r - (a + b * math.exp(-t / T))) ** 2
        if usable and (best_sse is None or sse < best_sse):
            best_sse, best_T = sse, T
    if best_T is None:
        return [None] * n_tr

    moves = []
    for pts in series:
        if len(pts) < 3:
            moves.append(None)
            continue
        a, b = wls([(math.exp(-t / best_T), r, w) for t, r, w in pts])
        # A usable ceiling fit: A inside [0, 100] (the ceiling itself) and
        # predictions inside range at both ends. The slope on
        # exp(-t/T) is negative for a rate rising toward its ceiling and
        # positive for one falling toward it; either is usable.
        if a is None or b is None or not (0.0 <= a <= 100.0):
            moves.append(None)
            continue
        r_last_fit = a + b * math.exp(-t_last_h / best_T)
        r_tgt_fit = a + b * math.exp(-t_tgt_h / best_T)
        if not (0.005 <= r_last_fit <= 99.995 and 0.005 <= r_tgt_fit <= 99.995):
            moves.append(None)
            continue
        moves.append(logit(r_tgt_fit) - logit(r_last_fit))
    return moves


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
