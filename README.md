# roblox-forecast

make.fun `roblox-up-and-coming-forecast-01` entry: a forecast of Roblox's
Up-and-Coming chart top 10, place by place.

- `main.py` -- the entry. Python 3.13, standard library only, no network, no
  randomness, byte-identical output on every run. Reads `$MAKEFUN_INPUT_DIR`
  (else `/input`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else
  `/output/forecast.json`). Runs in well under a second on the 34-read
  practice history.

## Method

The chart is sorted by a growth score Roblox does not publish, so the model
never rebuilds that score. It works in place space instead, projecting each
game's own record of the places it has held forward through the chart's rank
movement. One model runs whatever window the task names -- the window's own
hours are read from `task.json`, never assumed.

1. **Occupancy + transition** -- each game's recency-weighted distribution over
   the top-10 places (how often it held place 1..10), smoothed so an unseen
   (game, place) is not a hard zero, projected forward by convolving it with
   the pooled rank transition at every scored slot's horizon. The transition
   pools one observation per game per ordered pair of reads into (start rank,
   horizon) cells and is queried with a Gaussian in rank and a log-normal in
   the horizon, so a rank tells you less the further out you look; the first
   hours blend toward "it stays where it is". Games with no occupancy of their
   own project from their current rank.
2. **Age hazard** -- the chart only lists games published in the last 28 days,
   so a game is retired smoothly as it approaches the cut.

Each place's named mass is then **capped at 0.70**, and capping is where most of
the gain over v1 lives. A place's leftover is shared equally by every game on
that read's chart that the forecast did not name there -- it is the share held
by games that are not on the chart yet -- and a place whose named mass reaches
1.0 leaves nothing for them, so each of those misses is punished at the score
floor, ln(0.0001). Holding part of every place back for them is what buys the
improvement.

**Momentum tilt.** Before the cap is applied, each game's projected mass is
scaled by `exp(k * z)`, where `z` is the z-score of its log-playing slope over
the last 4 reads, taken against the pool's own mean and spread. A game whose
player count is climbing faster than the games it is competing with is scaled
up across every place; one that is falling is scaled down.

This is a *level* tilt, and the distinction from the earlier rejected idea
matters. Shifting the rank a game is queried at -- moving a climbing game a few
places up -- fought the transition model on its own ground and hurt the score.
Scaling the mass leaves the transition alone and adds a second opinion about
how much of the window a game should be expected to hold. The slope is fitted
by least squares over the reads a game actually appears in, so a missed read is
not penalised, and a game with fewer than two points to fit sits at z = 0
rather than being guessed at. Scoring against the pool's own spread rather than
an absolute rate is what makes it portable: player counts drift by orders of
magnitude between games, and only the comparison within the pool carries
signal.

The cap came down from 0.75 to 0.70 when the tilt went in, and the two are
linked: the tilt concentrates mass on the games that are climbing, which makes
the forecast sharper, and a sharper forecast needs more of each place held back
for the games that join the chart during the window.

The top 10 is chosen by a Hungarian assignment (pure Python, deterministic)
that maximises the expected top-10 agreement.

## Measured

Train on the 34 history reads, score the 10 archive reads in the 72 hours after
the close. Rolling backtest over cuts 6, 9, 12, 15, 18 and 21, and again over
every cut from 8 to 32 (25 folds, 230 scored reads).

| reading                              | practice | rolling | 25-fold |
| ------------------------------------ | -------: | ------: | ------: |
| v1 (the submitted entry)             |   -36.87 |  -40.00 |     -- |
| house persistence baseline           |   -43.51 |     -- |     -- |
| occupancy + transition, uncapped     |   -33.59 |  -39.85 |     -- |
| occupancy + transition, cap 0.75     |   -33.64 |  -37.97 |  -39.16 |
| rank marginal, capped                |   -33.75 |  -39.60 |     -- |
| the 0.6/0.4 blend, capped            |   -33.68 |  -38.54 |     -- |
| **v2: cap 0.75**                     | **-33.64** | **-37.97** | **-39.16** |
| **v3: v2 + momentum, cap 0.70**      | **-33.36** | **-37.70** | **-38.82** |

v3 is ahead of v2 on all three measures, ahead in 20 of the 25 folds and ahead
in both halves of them, and it lifts the practice top-10 hits from 6.40 to 7.00
and the agreement from 53.50 to 60.30.

A second reading was built and measured alongside: the **rank marginal**, which
hands a game the pooled record of the rank it holds now and carries no
per-game information at all. Mixed 0.6/0.4 with the occupancy reading it scored
-33.68 practice / -38.54 rolling; the occupancy reading alone, capped the same
way, scored -33.64 / -37.97. The rmarg weight is therefore zero, and
`_rmarg_probs` is kept in the file only as the record of that comparison.

Two further ideas were measured and rejected, both because they improved one
metric while damaging the other. Rank momentum as a *rank shift* hurt the
practice score badly. Projecting player counts forward did worse still: the
chart is computed rather than counted, `playing` is the noisiest field in a
read, and a ranking by extrapolated player count reached 3.5 top-10 hits a read
against 5.6 for the rank it started from. `DRIFT_HORIZON=0.6` looked like a
winner on the six-fold backtest (+0.04 rolling) but *lost* on the 25-fold one,
which is why the denser backtest is quoted alongside the six-fold one.

One further improvement was measured but not taken: raising the shrinkage
toward a flat spread from 0.08 to 0.15 (with the tilt in) scores -33.43 practice
and -37.57 six-fold / -38.75 twenty-five-fold -- better than v3 on both
backtests. It was left out because it buys backtest ground by giving up
practice ground, and the tilt plus the lower cap already delivers a win on
every measure without a third moving part.
