# roblox-forecast

make.fun `roblox-up-and-coming-forecast-01` entry: a forecast of Roblox's
Up-and-Coming chart top 10, place by place.

- `main.py` -- the entry. Python 3.13, standard library only, no network, no
  randomness, byte-identical output on every run. Reads `$MAKEFUN_INPUT_DIR`
  (else `/input`), writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else
  `/output/forecast.json`). Runs in well under a second on the 34-read
  practice history.
- `score.py` -- the local scorer, implementing the log score exactly as
  `RULES-ref.txt` describes it, plus the backtest that trains on the reads
  before a cut and scores every read in the 72 hours after it:

      python3 score.py backtest --cut 2026-11-06T00:00:00Z

- `tune.py` -- the same backtest over seven cuts, for a sturdier number.

      python3 tune.py
