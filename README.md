# gta6-trophy-rarity-forecast-01

Entry for the make.fun bounty "Forecast how many GTA VI players earn each
trophy in week one": a program (`main.py`, Python 3.13 standard library
only) that reads each `<id>.in` case and writes `<id>.out` with five
quantiles (5/25/50/75/95%) of every GTA VI (PS5) trophy's Exophase earn
rate at the case's target.

## Model (see the docstring in `main.py` for the full description)

An Exophase earn rate is a ratio — earners / players tracked — and both
halves move: the tracked base jumps whenever Exophase refreshes its counts
(on launch night it sat at 638 players for ~4.5 hours, then jumped to
3,868 and every rate but the 100% one fell), while earners grow as tracked
players play through the game.

- **Distinct snapshots.** The history is collapsed to its distinct
  snapshots (first-seen time of each value). Fitting raw reads would
  overweight long flat stretches and underweight the refreshes that
  actually move the rates.
- **Trend branch (≥ 3 distinct snapshots — the final case's shape).** Per
  trophy, a recency-weighted least-squares line of logit(rate) against
  ln(1 + hours since release), slope shrunk toward its tier mean, anchored
  at the last observed rate (only the fitted *move* is used, never the
  intercept). Blended 25/75 with a saturating fit, r(t) = A − B·exp(−t/T)
  with T chosen once by pooled residual: a logit line has no ceiling and,
  fitted over days 0–2, can extrapolate an early trophy from 40% to 98%
  by day 7, while the saturating curve cannot. On synthetic finals the
  saturating move's mean error was 0.06–0.21 logit vs 0.43–0.84 for the
  log-time move.
- **Flat branch (< 3 distinct snapshots — the practice cases' shape).** No
  trend is invented. The prior is that Exophase refreshes every few hours
  and a refresh drops every rate, so the median shifts down 0.5 logit with
  a wide spread (σ ≈ 1.0).
- **Spread.** σ = √(residual variance + (0.5·√days)²), floored at 0.6,
  capped at 2.0 — the fit's own residual plus the house baseline's
  random-walk term.
- **The 100% trophy** is forecast at a median of 99% with σ 1.0: it can
  only fall as fresh players sync in.

Deterministic, no network, no randomness; ~0.1 s per case.

## Local validation

The kit ships no reference answers for the practice cases, so the practice
answers were reconstructed from the facts the rules state (04:30 read:
3,868 players tracked, 4,276 trophies earned, every rate but the 100% one
lower). Two reconstructions were scored — A: new earns proportional to
old counts; B: new earns concentrated in early trophies. Reconstruction B
reproduces the house baseline's stated practice figures almost exactly
(loss 1.5096 vs stated ~1.51, typical miss 0.677 pp vs 0.68, coverage 0),
so B is the primary reference and A the sensitivity check.

| case | this entry | house baseline |
| --- | --- | --- |
| p00 vs recon B | **0.673** | 1.510 |
| p00 vs recon A | **0.593** | 1.404 |
| p01 | same as p00 (same answer read) | same |

Coverage of the 90% interval on the reconstructions: 0.982 (slightly
wide — the price of surviving a refresh shock the history never showed).

On four synthetic final-like cases (48 h of history, 4.5-day horizon,
different maturation/denominator regimes), this entry beat the baseline
in all four: fast-saturating 0.317 vs 0.332, slow 0.301 vs 0.547,
flat-tracked-base 0.304 vs 0.451, logit-linear 0.333 vs 0.728.

Output verified: 55 rows, one per trophy, non-decreasing quantiles,
≤ 65,536 bytes, byte-identical across repeated runs; edge cases (single
read, 2 snapshots, null rates, zero and 30-day horizons) all produce
valid output.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
