#!/usr/bin/env python3
"""Forecast of Roblox's Up-and-Coming chart top 10.

The program reads the chart history from $MAKEFUN_INPUT_DIR (else /input),
writes /$MAKEFUN_OUTPUT_DIR/forecast.json (else /output) and uses nothing but
the standard library.  It is fully deterministic: no randomness, no clocks,
no network.

Model
-----
The chart is sorted by a growth score Roblox does not publish, so the model
never tries to rebuild that score.  It works in rank space instead, with one
idea: *how a game's rank moves is a function of the rank it starts from and of
how much time passes*, and that function can be read straight off the history.

1.  Every ordered pair of reads (i, j) in the history gives one observation per
    game on the chart at read i: its rank there, the hours between the reads,
    and its rank at read j (or "gone" when it had left the chart).
2.  Those observations are pooled into (rank, horizon) cells.  A query "a game
    at rank r, t hours later" is answered by a kernel-weighted average of the
    cells around (r, t): a Gaussian in rank whose width grows with t (a rank
    tells you less the further out you look) and a log-normal in the horizon.
    The answer is a full distribution over the chart: a mass for "left the
    chart" and a mass for each future rank.
3.  The window's 72 slots are each asked that question at their own horizon,
    and the answers are averaged.  A game's probability of holding place k is
    its chance of still being on the chart times its chance of sitting at rank
    k once there.  What is left over at a place is the share held by games that
    are not on the chart yet, which the scorer spreads over them for us.
4.  Age is the one thing a rank cannot see.  When the history carries the
    games' details (the house recorder sends them) the creation date is known
    and a game is retired smoothly as it approaches the 28-day cut; without
    details the age is estimated from how long the game has already been
    charting, and the retirement is much softer.
5.  Two guards keep the answer honest where the kernel has nothing to say.  A
    few hours out the chart has barely moved, so "it stays where it is" is
    blended in over the first hours -- which is also the limit as the horizon
    goes to nought.  And the model's own probabilities are mixed with a flat
    spread over the games now high on the chart, because a forecast that is a
    little too sure of itself is punished far harder than one that is a little
    too vague.

Because every place is scored on its own, each place gets its own marginal
distribution, and those marginals are spread over every game that could
plausibly hold the place rather than piled onto one guessed order.

What the backtest rejected
--------------------------
Three ideas measured well as correlations and badly as forecasts, all for the
same reason: 34 reads are about 34 independent trajectories, and the chart
churns so hard that a rank alone carries most of the usable signal, so any
per-game refinement costs more in variance than it buys in bias.

*   A game's rank trend: games that rose tend to keep rising (correlation
    0.25), but querying the kernel at a trend-adjusted rank made the forecast
    worse and the top-10 hits collapse from 5.6 to 3.0.
*   Player-count growth: the chart sorts on growth and `playing` is the only
    growth in the history, and the top growth quintile reaches the top 10
    three to four times as often as the bottom one.  Used as a rank shift it
    broke the per-place totals; used as a third kernel dimension it was worse
    at every width tried.
*   Anchoring each place's total to the share of it the chart's own games
    really hold: the model's own totals are already closer to optimal than the
    empirical share, so forcing them there lost ground.

The shape constants were settled by the rolling backtest in score.py, which
trains on the reads before a cut and scores every read in the 72 hours after
it, over seven cuts.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------- constants

PLACES = 10
MAX_ROWS_PER_PLACE = 100
RANK_CAP = 130

# horizons (hours) at which the pooled cells are anchored: log-spaced edges
GAP_EDGES = [0.5, 1.5, 3.0, 5.0, 8.0, 12.0, 17.0, 24.0, 34.0, 48.0, 68.0,
             96.0, 136.0, 192.0, 272.0]
GAP_REP = [math.sqrt(GAP_EDGES[i] * GAP_EDGES[i + 1]) for i in range(len(GAP_EDGES) - 1)]
N_GAP = len(GAP_REP)

# kernel shapes, settled by the rolling backtest in score.py
RANK_SIGMA_BASE = 1.0
RANK_SIGMA_PER_HOUR = 0.17
GAP_SIGMA_LOG = 0.26

# hours over which "the chart has not moved yet" is blended into the answer
IDENTITY_HOURS = 3.0

# shrinkage: the model's own probabilities are mixed with a flat spread over
# the games now high on the chart, which the log score rewards whenever the
# model is a little too sure of itself
SHRINK_WEIGHT = 0.10
SHRINK_POOL = 25

# age model
CHART_LIFETIME_DAYS = 28.0
AGE_RAMP_DAYS = 1.0          # softness of the 28-day cut when creation is known
AGE_RAMP_ESTIMATED = 3.0     # ... and when the age is only estimated
AGE_AT_ENTRY_DAYS = 5.0      # prior age of a game the first time it is charted

TIME_GRID = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0]


# ---------------------------------------------------------------- loading

def parse_ts(text):
    """ISO 8601 to an aware UTC datetime; a stamp with no offset is UTC."""
    if isinstance(text, datetime):
        stamp = text
    else:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def load_history(indir):
    """Reads as a list of (datetime, {universe_id: (rank, playing, name)}, created_map)."""
    path = os.path.join(indir, "history.ndjson")
    reads = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            if not raw.get("ok"):
                continue
            rows = raw.get("up_and_coming")
            if not rows:
                continue
            stamp = raw.get("at")
            if not stamp:
                continue
            games = {}
            for row in rows:
                uid = row.get("universe_id")
                rank = row.get("rank")
                if uid is None or rank is None:
                    continue
                games[int(uid)] = (int(rank), row.get("playing"), row.get("name"))
            if games:
                created = {}
                details = raw.get("details")
                if isinstance(details, dict):
                    for key, value in details.items():
                        if not isinstance(value, dict):
                            continue
                        born = value.get("created")
                        if not born:
                            continue
                        try:
                            made = parse_ts(born)
                        except (ValueError, AttributeError):
                            continue
                        try:
                            created[int(key)] = made
                        except (TypeError, ValueError):
                            continue
                reads.append((parse_ts(stamp), games, created))
    reads.sort(key=lambda item: item[0])
    return reads


def load_task(indir):
    """task.json, or a sane stand-in if it is missing or unreadable."""
    path = os.path.join(indir, "task.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            task = json.load(handle)
        if isinstance(task, dict):
            return task
    except (OSError, ValueError):
        pass
    return {"places": PLACES, "window": {"slot_minutes": 60, "slots": 72}}


# ---------------------------------------------------------------- pooling

def bucket_of(gap):
    """Index of the horizon bucket a gap in hours falls in."""
    lo, hi = 0, N_GAP - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if gap < GAP_EDGES[mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def pool_observations(reads):
    """(rank, horizon bucket) -> counts.

    Each cell holds [gone, at place 1..PLACES, elsewhere, total].  Only the
    top of the chart is ever scored, so a future rank past `PLACES` is
    collapsed into one "elsewhere" counter.
    """
    cells = {}
    count = len(reads)
    for i in range(count):
        stamp_i, rank_i, _created_i = reads[i]
        for j in range(i + 1, count):
            gap = (reads[j][0] - stamp_i).total_seconds() / 3600.0
            if gap < 0.5:
                continue
            slot = bucket_of(gap)
            rank_j = reads[j][1]
            for uid, entry in rank_i.items():
                key = (entry[0], slot)
                cell = cells.get(key)
                if cell is None:
                    cell = [0] * (PLACES + 3)
                    cells[key] = cell
                later = rank_j.get(uid)
                if later is None:
                    cell[0] += 1
                elif later[0] <= PLACES:
                    cell[later[0]] += 1
                else:
                    cell[PLACES + 1] += 1
                cell[PLACES + 2] += 1
    return cells


def rank_cells(cells):
    by_rank = {}
    for (rank, slot), cell in cells.items():
        by_rank.setdefault(rank, {})[slot] = cell
    return by_rank


def kernel_weights(rank, hours):
    """(source rank, weight) and (horizon bucket, weight) around a query."""
    sigma_rank = RANK_SIGMA_BASE + RANK_SIGMA_PER_HOUR * hours
    span = int(sigma_rank * 3.5) + 2
    ranks = []
    for delta in range(-span, span + 1):
        source = rank + delta
        if source < 1 or source > RANK_CAP:
            continue
        weight = math.exp(-0.5 * (delta / sigma_rank) ** 2)
        if weight > 1e-4:
            ranks.append((source, weight))
    buckets = []
    for slot in range(N_GAP):
        rep = GAP_REP[slot]
        diff = math.log(rep / max(hours, 0.25)) / GAP_SIGMA_LOG
        weight = math.exp(-0.5 * diff * diff)
        if weight > 1e-4:
            buckets.append((slot, weight))
    return ranks, buckets


class Horizon(object):
    """The pooled rank-transition table, queried at any horizon."""

    def __init__(self, reads):
        self.by_rank = rank_cells(pool_observations(reads))
        self._cache = {}

    def distribution(self, rank, hours):
        """(weighted mass, [gone, place1..placePLACES, elsewhere, total])."""
        key = (rank, round(hours, 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ranks, buckets = kernel_weights(rank, hours)
        total = [0.0] * (PLACES + 3)
        for source, rw in ranks:
            slots = self.by_rank.get(source)
            if not slots:
                continue
            for slot, gw in buckets:
                cell = slots.get(slot)
                if cell is None:
                    continue
                w = rw * gw
                for k in range(PLACES + 3):
                    c = cell[k]
                    if c:
                        total[k] += w * c
        # An hour or two out the chart has barely moved, and the history's
        # shortest gap may be longer than that, so the kernel can be left with
        # nothing to say.  Blend in "it stays where it is", fading out over the
        # first few hours, which is also the right answer at the limit.
        still = [0.0] * (PLACES + 3)
        place = int(round(rank))
        still[place if 1 <= place <= PLACES else PLACES + 1] = 1.0
        still[PLACES + 2] = 1.0
        weight = 0.0
        if IDENTITY_HOURS > 0.0:
            weight = max(0.0, 1.0 - hours / IDENTITY_HOURS)
        if total[PLACES + 2] <= 0.0:
            total = still
            weight = 1.0
        if weight > 0.0:
            total = [(1.0 - weight) * t + weight * s for t, s in zip(total, still)]
        self._cache[key] = total
        return total


# ---------------------------------------------------------------- ages

def first_seen(reads):
    seen = {}
    for stamp, games, _created in reads:
        for uid in games:
            if uid not in seen:
                seen[uid] = stamp
    return seen


def age_exit_probability(age_days, known):
    """Chance the game has already fallen off the chart through the 28-day cut."""
    ramp = AGE_RAMP_DAYS if known else AGE_RAMP_ESTIMATED
    over = (age_days - CHART_LIFETIME_DAYS) / ramp
    if over > 6.0:
        return 1.0
    if over < -6.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-over))


def build_ages(reads):
    """universe_id -> (age in days at the last read, whether that age is known)."""
    last_stamp = reads[-1][0]
    known = {}
    for _stamp, _games, created in reads:
        for uid, born in created.items():
            previous = known.get(uid)
            if previous is None or born > previous:
                known[uid] = born
    ages = {}
    for uid, stamp in first_seen(reads).items():
        born = known.get(uid)
        if born is not None:
            ages[uid] = ((last_stamp - born).total_seconds() / 86400.0, True)
        else:
            charted = (last_stamp - stamp).total_seconds() / 86400.0
            ages[uid] = (charted + AGE_AT_ENTRY_DAYS, False)
    return ages


# ---------------------------------------------------------------- the model

def slot_hours(task, origin):
    """Hours from the last history read to each scored slot."""
    window = task.get("window") or {}
    start = window.get("from")
    minutes = window.get("slot_minutes") or 60
    slots = window.get("slots") or 72
    if not start:
        return [1.0 + k for k in range(slots)]
    start = parse_ts(start)
    out = []
    for k in range(slots):
        stamp = start + timedelta(minutes=minutes * k)
        hours = (stamp - origin).total_seconds() / 3600.0
        out.append(min(max(hours, 0.25), 400.0))
    return out


class Interpolated(object):
    """Rank-transition distributions on a horizon grid, linearly interpolated."""

    def __init__(self, horizon, grid):
        self.horizon = horizon
        self.grid = grid
        self._cache = {}

    def at(self, rank, hours):
        grid = self.grid
        if hours <= grid[0]:
            lo = hi = grid[0]
        elif hours >= grid[-1]:
            lo = hi = grid[-1]
        else:
            lo = hi = None
            for a in range(len(grid) - 1):
                if grid[a] <= hours <= grid[a + 1]:
                    lo, hi = grid[a], grid[a + 1]
                    break
        key = (rank, round(hours, 3))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if lo == hi:
            dist = self.horizon.distribution(rank, lo)
        else:
            low = self.horizon.distribution(rank, lo)
            high = self.horizon.distribution(rank, hi)
            frac = (hours - lo) / (hi - lo)
            dist = [low[k] + frac * (high[k] - low[k]) for k in range(PLACES + 3)]
        self._cache[key] = dist
        return dist


def forecast(reads, task, verbose=False):
    origin_stamp, origin_games, _created = reads[-1]
    ages = build_ages(reads)
    hours_grid = slot_hours(task, origin_stamp)
    horizon = Horizon(reads)

    grid = [h for h in TIME_GRID if 0.4 <= h <= max(hours_grid)]
    if not grid or grid[0] > min(hours_grid):
        grid.insert(0, min(hours_grid))
    if grid[-1] < max(hours_grid):
        grid.append(max(hours_grid))
    grid = sorted(set(grid))
    table = Interpolated(horizon, grid)

    # per game, per place, averaged over the window's scored slots
    probs = {}
    for uid, entry in origin_games.items():
        rank = float(entry[0])
        age_days, known = ages.get(uid, (0.0, False))
        row = [0.0] * (PLACES + 1)
        for hours in hours_grid:
            survive = 1.0 - age_exit_probability(age_days + hours / 24.0, known)
            if survive <= 0.0:
                continue
            dist = table.at(rank, hours)
            mass = dist[PLACES + 2]
            if mass <= 0.0:
                continue
            gone = dist[0] / mass
            present = (1.0 - gone) * survive
            for place in range(1, PLACES + 1):
                row[place] += present * (dist[place] / mass)
        span = len(hours_grid)
        probs[uid] = [row[k] / span for k in range(PLACES + 1)]

    if SHRINK_WEIGHT > 0.0:
        # a flat fallback over the games now highest on the chart: the log
        # score punishes a forecast that is a little too sure of itself
        pool = [uid for uid, entry in
                sorted(origin_games.items(), key=lambda item: item[1][0])[:SHRINK_POOL]]
        flat = 1.0 / len(pool)
        for place in range(1, PLACES + 1):
            for uid in pool:
                probs[uid][place] = ((1.0 - SHRINK_WEIGHT) * probs[uid][place]
                                     + SHRINK_WEIGHT * flat)

    # a place's leftover is the share held by games not on the chart yet
    rows = []
    for place in range(1, PLACES + 1):
        entries = sorted(((values[place], uid) for uid, values in probs.items()
                          if values[place] > 1e-7), key=lambda item: (-item[0], item[1]))
        kept = entries[:MAX_ROWS_PER_PLACE]
        total = sum(p for p, _uid in kept)
        if total > 1.0:
            kept = [(p / total, uid) for p, uid in kept]
            total = 1.0
        for p, uid in kept:
            rows.append({"universe_id": uid, "place": place, "p": round(p, 6)})
        if verbose:
            print("place %2d  named %2d  sum %.4f  leftover %.4f"
                  % (place, len(kept), total, 1.0 - total))

    # the best-guess order: most likely to be charted at all, then best place
    order = sorted(probs.items(),
                   key=lambda item: (-sum(item[1][1:]),
                                     sum(k * item[1][k] for k in range(1, PLACES + 1)),
                                     item[0]))
    top10 = [uid for uid, _values in order[:PLACES]]

    return {"probabilities": rows, "top10": top10}, probs


# ---------------------------------------------------------------- entry point

def main():
    indir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    reads = load_history(indir)
    if not reads:
        raise SystemExit("no readable chart history in %s" % indir)
    task = load_task(indir)
    verbose = os.environ.get("MAKEFUN_VERBOSE") == "1"
    forecast_doc, _probs = forecast(reads, task, verbose=verbose)
    name = task.get("output") if isinstance(task.get("output"), str) else "forecast.json"
    name = os.path.basename(name) or "forecast.json"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(forecast_doc, handle, separators=(",", ":"))
        handle.write("\n")
    if verbose:
        print("wrote %s (%d rows, %d bytes)"
              % (path, len(forecast_doc["probabilities"]), os.path.getsize(path)))


if __name__ == "__main__":
    main()
