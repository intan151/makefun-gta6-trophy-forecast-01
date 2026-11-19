"""GTA VI trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Reads every <id>.in case in the input directory and writes <id>.out forecasts
to the output directory. Python 3.13, standard library only, no network.

Model (per trophy, on the logit scale):
  * History is collapsed to distinct snapshots (first-seen values), so long
    flat stretches where Exophase has not refreshed do not overweight the fit.
  * With >= 3 distinct snapshots, the log-odds of the earn rate is fitted by
    recency-weighted least squares against ln(1 + hours since release)
    (24 h half-life weights). Slopes are shrunk toward their tier mean, and
    the line is anchored at the last read: only the fitted *move* from the
    last read to the target is used. That line is blended 25/75 with a
    saturating curve r(t) = A - B*exp(-t/T), whose timescale T is chosen once
    for all trophies by pooled residual; a logit line has no ceiling, the
    saturating curve cannot run away, and the blend keeps both honest.
  * With < 3 distinct snapshots (e.g. launch night, flat history) no trend
    is invented: the median shifts down by 0.5 logit (a refresh brings in
    new players who hold fewer trophies) with spread sigma = 1.0.
  * Spread: sigma = sqrt(residual variance + (0.5*sqrt(days))^2), floored
    at 0.6 and capped at 2.0. The trophy sitting at 100% is forecast at a
    99% median with sigma 1.0.

Quantiles are inv_logit(median_logit + z_tau * sigma) for the standard
normal z at 0.05/0.25/0.5/0.75/0.95, rounded to 4 decimals. Deterministic.
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
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release-day midnight UTC; hours are measured from here.
RELEASE_EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

HALF_LIFE_HOURS = 24.0
LINE_WEIGHT = 0.25          # blend: 25% logit line, 75% saturating curve
FLAT_SHIFT = -0.5           # logit shift when history is flat / too short
FLAT_SIGMA = 1.0
SIGMA_FLOOR = 0.6
SIGMA_CAP = 2.0
RW_SIGMA = 0.5              # random-walk spread, logit per sqrt(day)
SAT_T_GRID = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return clamp_pct(100.0 / (1.0 + math.exp(-lg)))
    e = math.exp(lg)
    return clamp_pct(100.0 * e / (1.0 + e))


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime; None on failure."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def hours_since_release(dt, release):
    return max(0.0, (dt - release).total_seconds() / 3600.0)


def weighted_ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def plain_ols(xs, ys):
    """Unweighted least squares y ~ a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - b * mx, b)


def distinct_snapshots(history):
    """Collapse history to first-seen distinct rate vectors, oldest first."""
    seen = set()
    snaps = []
    for read in history or []:
        rates = read.get("rates") if isinstance(read, dict) else None
        if not isinstance(rates, list):
            continue
        key = tuple(rates)
        if key in seen:
            continue
        seen.add(key)
        snaps.append(read)
    return snaps


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon = case.get("horizon_minutes") or 0
    try:
        horizon = float(horizon)
    except (TypeError, ValueError):
        horizon = 0.0
    days = max(0.0, horizon / 1440.0)

    game = case.get("game") or {}
    release = RELEASE_EPOCH
    rd = game.get("release_date")
    if isinstance(rd, str) and len(rd) >= 10:
        parsed = parse_time(rd[:10] + "T00:00:00Z")
        if parsed is not None:
            release = parsed

    at = parse_time(case.get("at"))
    snaps = distinct_snapshots(history)
    snap_times = [parse_time(s.get("at")) for s in snaps]
    last_time = snap_times[-1] if snap_times and snap_times[-1] else at
    if last_time is None:
        last_time = release
    target_time = parse_time(case.get("target"))
    if target_time is None:
        target_time = last_time
    t_last = hours_since_release(last_time, release)
    t_target = hours_since_release(target_time, release)
    x_last = math.log1p(t_last)
    x_target = math.log1p(t_target)

    n_tr = len(trophies)
    # Per-trophy series over distinct snapshots: (t_hours, rate) pairs.
    series = []
    for i in range(n_tr):
        pts = []
        for snap, st in zip(snaps, snap_times):
            if st is None:
                continue
            rates = snap.get("rates") or []
            if i >= len(rates):
                continue
            r = rates[i]
            if r is None:
                continue
            try:
                r = float(r)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(r):
                continue
            pts.append((hours_since_release(st, release), r))
        series.append(pts)

    def last_rate(i):
        pts = series[i]
        if pts:
            return pts[-1][1]
        return None

    # Fallback last rates: last non-null rate of the same tier, else 1.0.
    tier_last = {}
    for i, tr in enumerate(trophies):
        r = last_rate(i)
        if r is not None:
            tier_last.setdefault(tr.get("tier"), []).append(r)
    tier_fallback = {}
    for tier, vals in tier_last.items():
        vals = sorted(vals)
        tier_fallback[tier] = vals[len(vals) // 2]
    all_last = sorted(v for vals in tier_last.values() for v in vals)
    global_fallback = all_last[len(all_last) // 2] if all_last else 1.0

    rich = sum(1 for pts in series if len(pts) >= 3) >= 1 and len(snaps) >= 3

    # --- Saturating-curve timescale, chosen once by pooled residual. ---
    sat_T = None
    sat_params = []  # per trophy (A, C) with r = A + C*exp(-t/T), or None
    if rich:
        best_T, best_sse = None, None
        for T in SAT_T_GRID:
            sse = 0.0
            used = 0
            for pts in series:
                if len(pts) < 3:
                    continue
                us = [math.exp(-t / T) for t, _ in pts]
                fit = plain_ols(us, [r for _, r in pts])
                if fit is None:
                    continue
                a, c = fit
                sse += sum((r - (a + c * u)) ** 2 for u, (_, r) in zip(us, pts))
                used += 1
            if used and (best_sse is None or sse < best_sse - 1e-12):
                best_T, best_sse = T, sse
        sat_T = best_T
        if sat_T is not None:
            for pts in series:
                if len(pts) < 2:
                    sat_params.append(None)
                    continue
                us = [math.exp(-t / sat_T) for t, _ in pts]
                sat_params.append(plain_ols(us, [r for _, r in pts]))
        else:
            sat_params = [None] * n_tr
    else:
        sat_params = [None] * n_tr

    # --- Recency-weighted logit-line fits, slopes shrunk to tier means. ---
    raw_slopes = [None] * n_tr
    line_intercepts = [None] * n_tr
    resid_sq_sum = 0.0
    resid_n = 0
    if rich:
        for i, pts in enumerate(series):
            if len(pts) < 2:
                continue
            xs = [math.log1p(t) for t, _ in pts]
            ys = [logit(r) for _, r in pts]
            ws = [0.5 ** ((t_last - t) / HALF_LIFE_HOURS) for t, _ in pts]
            fit = weighted_ols(xs, ys, ws)
            if fit is None:
                continue
            line_intercepts[i], raw_slopes[i] = fit
            for x, y in zip(xs, ys):
                resid_sq_sum += (y - (fit[0] + fit[1] * x)) ** 2
                resid_n += 1
        tier_slopes = {}
        for i, tr in enumerate(trophies):
            if raw_slopes[i] is not None:
                tier_slopes.setdefault(tr.get("tier"), []).append(raw_slopes[i])
        tier_mean = {}
        for tier, vals in tier_slopes.items():
            tier_mean[tier] = sum(vals) / len(vals)
        all_slopes = [s for s in raw_slopes if s is not None]
        global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    else:
        tier_mean = {}
        global_mean = 0.0

    if resid_n > 0:
        dof = max(1, resid_n - 2 * sum(1 for s in raw_slopes if s is not None))
        resid_var = resid_sq_sum / dof
    else:
        resid_var = 0.0
    rich_sigma = min(SIGMA_CAP, max(SIGMA_FLOOR,
                                    math.sqrt(resid_var + (RW_SIGMA * math.sqrt(days)) ** 2)))

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rate(i)
        if r_last is None:
            r_last = tier_fallback.get(tr.get("tier"), global_fallback)
        lg_last = logit(r_last)

        if r_last >= 99.99:
            # The 100% trophy: it can only stay (nearly) complete.
            median_lg, sigma = logit(99.0), FLAT_SIGMA
        elif not rich:
            median_lg, sigma = lg_last + FLAT_SHIFT, FLAT_SIGMA
        else:
            slope = raw_slopes[i]
            if slope is not None:
                tm = tier_mean.get(tr.get("tier"), global_mean)
                n_pts = len(series[i])
                w = n_pts / (n_pts + 2.0)  # shrink toward the tier mean
                # Never extrapolate backwards if the target precedes the last read.
                slope = w * slope + (1.0 - w) * tm
                line_lg = lg_last + slope * max(0.0, x_target - x_last)
            else:
                line_lg = lg_last
            sat_lg = None
            if sat_T is not None and sat_params[i] is not None:
                a, c = sat_params[i]
                r_sat = a + c * math.exp(-max(t_target, t_last) / sat_T)
                sat_lg = logit(r_sat)
            if sat_lg is not None:
                median_lg = LINE_WEIGHT * line_lg + (1.0 - LINE_WEIGHT) * sat_lg
            else:
                median_lg = line_lg
            sigma = rich_sigma

        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(median_lg + z * sigma))
            q = float(q)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
