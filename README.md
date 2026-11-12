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

Each place's named mass is then **capped at 0.75**, and this is where most of
the gain over v1 lives. A place's leftover is shared equally by every game on
that read's chart that the forecast did not name there -- it is the share held
by games that are not on the chart yet -- and a place whose named mass reaches
1.0 leaves nothing for them, so each of those misses is punished at the score
floor, ln(0.0001). Holding a quarter of every place back for them is what buys
the improvement.

The top 10 is chosen by a Hungarian assignment (pure Python, deterministic)
that maximises the expected top-10 agreement.

## Measured

Train on the 34 history reads, score the 10 archive reads in the 72 hours after
the close; rolling backtest over cuts 6, 9, 12, 15, 18 and 21.

| reading                              | practice | rolling |
| ------------------------------------ | -------: | ------: |
| v1 (the submitted entry)             |   -36.87 |  -40.00 |
| house persistence baseline           |   -43.51 |     -- |
| occupancy + transition, uncapped     |   -33.59 |  -39.85 |
| occupancy + transition, capped       |   -33.64 |  -37.97 |
| rank marginal, capped                |   -33.75 |  -39.60 |
| the 0.6/0.4 blend, capped            |   -33.68 |  -38.54 |
| **this model (capped occupancy)**    | **-33.64** | **-37.97** |

A second reading was built and measured alongside: the **rank marginal**, which
hands a game the pooled record of the rank it holds now and carries no
per-game information at all. Mixed 0.6/0.4 with the occupancy reading it scored
-33.68 practice / -38.54 rolling; the occupancy reading alone, capped the same
way, scored -33.64 / -37.97. The rmarg weight is therefore zero, and
`_rmarg_probs` is kept in the file only as the record of that comparison.

Two further ideas were measured and dropped. Rank momentum -- a game's recent
climb used to shift the rank it is queried at -- hurt the practice score badly.
Projecting player counts forward did worse still: the chart is computed rather
than counted, `playing` is the noisiest field in a read, and a ranking by
extrapolated player count reached 3.5 top-10 hits a read against 5.6 for the
rank it started from.
