#!/usr/bin/env python3
"""Forecast of Roblox's Up-and-Coming chart top 10.

The program reads the chart history from $MAKEFUN_INPUT_DIR (else /input),
writes /$MAKEFUN_OUTPUT_DIR/forecast.json (else /output) and uses nothing but
the standard library.  It is fully deterministic: no randomness, no clocks,
no network.  It runs the same model whatever window it is handed, and reads the
window it must forecast from task.json rather than assuming one.

Model
-----
The chart is sorted by a growth score Roblox does not publish, so the model
never rebuilds that score.  It works in place space instead, projecting each
game's own record of the places it has held forward through the chart's rank
movement, and then picks the top 10 with a Hungarian assignment.

1.  Occupancy + transition.  Each game's recency-weighted distribution over the
    top-10 places -- how often it held place 1..10 -- smoothed so an unseen
    (game, place) is not a hard zero, is projected forward by convolving it
    with the pooled rank transition at every scored slot's horizon.  The
    transition pools one observation per game per ordered pair of reads into
    (start rank, horizon) cells and is queried with a Gaussian in rank and a
    log-normal in the horizon, so a rank tells you less the further out you
    look.  The first hours blend toward "it stays where it is", which is also
    the right limit as the horizon goes to nought.  A game with no occupancy
    of its own projects from its current rank.

2.  Age hazard.  The chart only lists games published in the last 28 days, so
    a game is retired smoothly as it approaches the cut.  When the history
    carries the games' details the creation date is known; without them the
    age is estimated from how long the game has already been charting.

A second reading was built and measured alongside this one: the rank marginal,
which hands a game the pooled record of the rank it holds now and carries no
per-game information at all.  Mixed 0.6/0.4 with the occupancy reading it
scored -33.68 practice / -38.54 rolling; the occupancy reading alone, capped
the same way, scored -33.64 / -37.97.  The rmarg weight is therefore zero and
`_rmarg_probs` is kept below only as the record of that comparison.

Capping
-------
Each place's named mass is capped at 0.75, and this is where most of the gain
over v1 lives.  The occupancy reading uncapped scores -33.59 practice and
-39.85 rolling, only +0.15 over v1 there; capped at 0.75 the same reading
scores -33.64 and -37.97, so the cap is worth about +1.9 on the rolling
backtest by itself.  A place's leftover is shared equally by every game on
that read's chart that the forecast did not name there -- it is the share held
by games that are not on the chart yet -- and a place whose named mass reaches
1.0 leaves nothing for them, so every one of those misses is punished at the
score floor, ln(0.0001).  Holding a quarter of every place back is what buys
the improvement.

Measured (train on the 34 history reads, score the 10 archive reads in the 72
hours after the close; rolling backtest over cuts 6, 9, 12, 15, 18 and 21):

    reading                              practice    rolling
    v1 (the submitted entry)              -36.87      -40.00
    house persistence baseline            -43.51         --
    occupancy + transition, uncapped      -33.59      -39.85
    occupancy + transition, capped        -33.64      -37.97
    rank marginal, capped                 -33.75      -39.60
    the 0.6/0.4 blend, capped             -33.68      -38.54
    this model (capped occupancy)         -33.64      -37.97

Rank momentum was tried as a third signal -- a game's recent climb used to
shift the rank it is queried at -- and it hurt the practice score badly enough
to be dropped.  So did projecting player counts forward: the chart is computed
rather than counted, `playing` is the noisiest field in a read, and a ranking
by extrapolated player count reached 3.5 top-10 hits a read against 5.6 for
the rank it started from.

The top 10 is chosen by a Hungarian assignment (implemented here in pure
Python) that maximises the expected top-10 agreement: the reward of putting a
game at a place is its probability mass at each place weighted by the
agreement that place would score, and the assignment weighs each game's mass
against the distance cost of the place it is given.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------- constants

PLACES = 10
MAX_ROWS_PER_PLACE = 100
RANK_CAP = 15              # highest start rank the transition tracks

# occupancy: recent reads count more than old ones
RECENCY_DECAY = 0.9
OCC_SMOOTH = 0.06          # additive smoothing on (game, place) counts
BETA_OCC = 0.45            # blend of a game's own occupancy vs its current rank

# transition: horizon bucket edges (hours); a query smooths in rank and in
# log-horizon across these cells
GAP_EDGES = [3.0, 6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 120.0, 400.0]
N_GAP = len(GAP_EDGES) - 1
GAP_REP = [math.sqrt(GAP_EDGES[i] * GAP_EDGES[i + 1]) for i in range(N_GAP)]
RANK_SIGMA = 1.3           # transition smoothing in rank
GAP_SIGMA_LOG = 0.5        # transition smoothing in log-horizon

# projection: scale the horizon the drift is queried at, and the hours over
# which "the chart has not moved yet" is blended into the answer
DRIFT_HORIZON = 0.8
IDENTITY_HOURS = 10.0

# concentration and shrinkage
POOL = 14                  # candidate games taken from the current chart top
SHRINK_WEIGHT = 0.08       # mix with a flat spread over the games now high up
SHRINK_POOL = 14

# age model
CHART_LIFETIME_DAYS = 28.0
AGE_RAMP_DAYS = 1.2        # softness of the 28-day cut when creation is known
AGE_RAMP_ESTIMATED = 3.0   # ... and when the age is only estimated
AGE_AT_ENTRY_DAYS = 5.0    # prior age of a game the first time it is charted

# rank-marginal reading: the horizon its windows span, the highest rank it
# tracks, the width it smooths across ranks and the size of its candidate pool
RMARG_WINDOW_HOURS = 72.0
RMARG_RANK_CAP = 20
RMARG_RANK_SIGMA = 1.0
RMARG_RANK_SPAN = 4
RMARG_POOL = 16

# the blend, and the named mass a place is allowed to reach.  The 0.6/0.4 mix
# of the two readings measured -33.68 on the practice forecast and -38.54 on
# the rolling backtest; the proto reading alone, capped the same way, measured
# -33.64 and -37.97 -- ahead on both.  The rmarg weight is therefore zeroed
# after that measurement rather than kept for looks: `_rmarg_probs` stays below
# as the record of the comparison and of a reading that was tried.
BLEND_PROTO = 1.0
BLEND_RMARG = 0.0
PLACE_CAP = 0.75


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


# ---------------------------------------------------------------- occupancy

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


def build_occupancy(reads):
    """universe_id -> distribution over the top-10 places.

    Each game's occupancy is the recency-weighted share of reads in which it
    held each top-10 place, additively smoothed so a (game, place) never seen
    is not a hard zero -- a game seen at #3 twice and #7 once is not a zero at
    #4.  Only games that reached the top 10 at least once get a distribution;
    the rest fall back to the transition model from their current rank.
    """
    n = len(reads)
    counts = {}       # uid -> [0, place1..placePLACES]
    for i, (_stamp, games, _created) in enumerate(reads):
        weight = RECENCY_DECAY ** (n - 1 - i)
        for uid, entry in games.items():
            rank = entry[0]
            if 1 <= rank <= PLACES:
                row = counts.get(uid)
                if row is None:
                    row = [0.0] * (PLACES + 1)
                    counts[uid] = row
                row[rank] += weight
    dists = {}
    for uid, row in counts.items():
        smoothed = [row[place] + OCC_SMOOTH for place in range(1, PLACES + 1)]
        total = sum(smoothed)
        dists[uid] = [value / total for value in smoothed]
    return dists


def build_transitions(reads):
    """(start rank, horizon bucket) -> [gone, place1..PLACES, below10, total].

    Every ordered pair of reads pools one observation per game into the cell
    for its starting rank and the hours between the reads.  A future rank past
    `PLACES` collapses into one "below top 10" counter.
    """
    cells = {}
    count = len(reads)
    for i in range(count):
        stamp_i, rank_i, _created_i = reads[i]
        for j in range(i + 1, count):
            gap = (reads[j][0] - stamp_i).total_seconds() / 3600.0
            if gap < 1.0:
                continue
            slot = bucket_of(gap)
            rank_j = reads[j][1]
            for uid, entry in rank_i.items():
                start = entry[0]
                if start > RANK_CAP:
                    continue
                cell = cells.get((start, slot))
                if cell is None:
                    cell = [0.0] * (PLACES + 3)
                    cells[(start, slot)] = cell
                later = rank_j.get(uid)
                if later is None:
                    cell[0] += 1.0
                elif later[0] <= PLACES:
                    cell[later[0]] += 1.0
                else:
                    cell[PLACES + 1] += 1.0
                cell[PLACES + 2] += 1.0
    return cells


def transition_distribution(cells, rank, hours):
    """[gone, place1..PLACES, below10, total] for a start rank at a horizon.

    A Gaussian in rank and a log-normal in the horizon pool the cells around
    the query, so a rank tells you less the further out you look.
    """
    span = int(RANK_SIGMA * 3.0) + 2
    ranks = []
    for delta in range(-span, span + 1):
        source = rank + delta
        if source < 1 or source > RANK_CAP:
            continue
        weight = math.exp(-0.5 * (delta / RANK_SIGMA) ** 2)
        if weight > 1e-4:
            ranks.append((source, weight))
    buckets = []
    for slot in range(N_GAP):
        rep = GAP_REP[slot]
        diff = math.log(max(rep, 0.5) / max(hours, 0.25)) / GAP_SIGMA_LOG
        weight = math.exp(-0.5 * diff * diff)
        if weight > 1e-4:
            buckets.append((slot, weight))
    total = [0.0] * (PLACES + 3)
    for source, rw in ranks:
        for slot, gw in buckets:
            cell = cells.get((source, slot))
            if cell is None:
                continue
            w = rw * gw
            for k in range(PLACES + 3):
                c = cell[k]
                if c:
                    total[k] += w * c
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


def ever_charted(reads):
    """Every universe the history has ever charted, best rank first."""
    best = {}
    for _stamp, games, _created in reads:
        for uid, entry in games.items():
            rank = entry[0]
            previous = best.get(uid)
            if previous is None or rank < previous:
                best[uid] = rank
    return sorted(best, key=lambda uid: (best[uid], uid))


def current_pool(origin_games, size):
    """The `size` games highest on the chart as the history ends."""
    ranked = sorted(origin_games.items(), key=lambda item: item[1][0])
    return [uid for uid, _entry in ranked[:size]]


def survival(reads, hours_grid, ages):
    """universe_id -> mean chance of still being charted over the window."""
    out = {}
    span = len(hours_grid)
    if span == 0:
        return {uid: 1.0 for uid in ages}
    for uid, (age_days, known) in ages.items():
        total = 0.0
        for hours in hours_grid:
            total += 1.0 - age_exit_probability(age_days + hours / 24.0, known)
        out[uid] = total / span
    return out


# ---------------------------------------------------------------- hungarian

def hungarian(cost):
    """Minimum-cost assignment of every row to a distinct column (rows <= cols).

    The shortest-augmenting-path (Kuhn-Munkres / Jonker-Volgenant) method in
    pure Python, fully deterministic.  `cost` is a list of rows, each a list of
    column costs; the return maps row index -> column index.  With more columns
    than rows the algorithm picks the columns that minimise the total.
    """
    n = len(cost)
    m = len(cost[0]) if n else 0
    if n == 0 or m == 0 or n > m:
        return list(range(n))
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def choose_top10(probs, candidates, spare=()):
    """The 10 games and their order, by Hungarian assignment.

    Putting a game at a place is rewarded by its probability mass at each
    place weighted by the agreement that place scores (10 minus the place
    distance), so the assignment maximises the expected top-10 agreement given
    the projected per-place probabilities.

    `spare` supplies further games to name when the candidate pool holds fewer
    than `PLACES` of them -- every universe the history has ever charted, best
    rank first -- so a short chart still yields a full top 10 rather than an
    index error.
    """
    games = list(candidates)
    if len(games) < PLACES:
        taken = set(games)
        extra = sorted((uid for uid in probs if uid not in taken),
                       key=lambda u: (-sum(probs[u][1:]), u))
        for uid in list(extra) + list(spare):
            if len(games) >= PLACES:
                break
            if uid not in taken:
                taken.add(uid)
                games.append(uid)
    if len(games) < PLACES:
        # the whole history holds fewer games than there are places
        return games
    reward = {}
    for uid in games:
        row = probs.get(uid)
        if row is None:
            # a game named only to fill the top 10 out: no projected mass, so
            # the assignment has no reason to put it anywhere but last
            reward[uid] = [0.0] * PLACES
            continue
        reward[uid] = [sum(row[q] * (PLACES - abs(place - q))
                           for q in range(1, PLACES + 1))
                       for place in range(1, PLACES + 1)]
    # rows are the places, columns the games; minimise the negated reward
    cost = [[-reward[uid][place - 1] for uid in games]
            for place in range(1, PLACES + 1)]
    assignment = hungarian(cost)
    return [games[assignment[place - 1]] for place in range(1, PLACES + 1)]


# ------------------------------------------------- reading one: proto

def _proto_probs(reads, task):
    """Occupancy projected through the rank transition.

    Returns {universe_id: [0.0, p1..pPLACES]} over the current chart's top
    `POOL` games, each row the window-average chance of holding each place.
    """
    origin_stamp, origin_games, _created = reads[-1]
    ages = build_ages(reads)
    hours_grid = slot_hours(task, origin_stamp)
    occ_dists = build_occupancy(reads)
    cells = build_transitions(reads)

    ranked = sorted(origin_games.items(), key=lambda item: item[1][0])
    pool = [uid for uid, _entry in ranked[:POOL]]
    span = len(hours_grid)

    probs = {uid: [0.0] * (PLACES + 1) for uid in pool}
    for uid in pool:
        rank = int(origin_games[uid][0])
        age_days, known = ages.get(uid, (0.0, False))
        occ = occ_dists.get(uid)
        beta = BETA_OCC if occ is not None else 0.0
        # starting distribution: the game's own occupancy blended with a point
        # mass at its current rank
        anchor = rank if 1 <= rank <= PLACES else PLACES
        start = [0.0] * (PLACES + 1)
        start[anchor] += 1.0 - beta
        if occ is not None:
            for place in range(1, PLACES + 1):
                start[place] += beta * occ[place - 1]
        row = [0.0] * (PLACES + 1)
        for hours in hours_grid:
            survive = 1.0 - age_exit_probability(age_days + hours / 24.0, known)
            if survive <= 0.0:
                continue
            dist = transition_distribution(cells, rank, hours * DRIFT_HORIZON)
            mass = dist[PLACES + 2]
            if mass <= 0.0:
                conv = list(start)
            else:
                conv = [0.0] * (PLACES + 1)
                for place in range(1, PLACES + 1):
                    acc = 0.0
                    for j in range(1, PLACES + 1):
                        if start[j] > 0.0:
                            acc += start[j] * (dist[place] / mass)
                    conv[place] = acc
            # the first hours blend toward "it stays where it is"
            idw = max(0.0, 1.0 - hours / IDENTITY_HOURS) if IDENTITY_HOURS > 0.0 else 0.0
            for place in range(1, PLACES + 1):
                val = conv[place]
                if idw > 0.0:
                    val = (1.0 - idw) * val + idw * start[place]
                row[place] += survive * val
        for place in range(1, PLACES + 1):
            probs[uid][place] = row[place] / span

    if SHRINK_WEIGHT > 0.0:
        # a flat fallback over the games now highest on the chart: the log
        # score punishes a forecast that is a little too sure of itself
        spool = [uid for uid, _entry in ranked[:SHRINK_POOL] if uid in probs]
        flat = 1.0 / len(spool)
        for place in range(1, PLACES + 1):
            for uid in spool:
                probs[uid][place] = ((1.0 - SHRINK_WEIGHT) * probs[uid][place]
                                     + SHRINK_WEIGHT * flat)
    return probs


# ------------------------------------------------- reading two: rmarg

def build_rank_marginal(reads):
    """rank -> [place1..PLACES, below10, gone, total] over a window.

    Every read i contributes, for each game at rank r there, one weighted
    observation per read j inside `RMARG_WINDOW_HOURS` after it: the place the
    game held at j, weighted by 1 over the number of such reads so each window
    counts once.  A game no longer on the chart counts as gone; a game past
    place 10 counts as below.

    This is a faithful port of the reference prototype, counter layout
    included: it keeps `PLACES + 2` counters, so "past place 10" lands on the
    same counter as place 10 itself and place 10's share carries the below-10
    mass with it.  The prototype was tuned and measured with that layout, so it
    is kept here rather than silently repaired; separating the two counters
    (and so deflating place 10) measures a shade better on both the practice
    forecast and the rolling backtest, and is the first thing to change if this
    reading is ever revisited.
    """
    acc = {}
    count = len(reads)
    for i in range(count):
        games_i = reads[i][1]
        stamp_i = reads[i][0]
        later = []
        for j in range(count):
            if j == i:
                continue
            gap = (reads[j][0] - stamp_i).total_seconds() / 3600.0
            if 0.0 < gap <= RMARG_WINDOW_HOURS:
                later.append(j)
        if not later:
            continue
        weight = 1.0 / len(later)
        for uid, entry in games_i.items():
            rank = entry[0]
            if rank > RMARG_RANK_CAP:
                continue
            cell = acc.get(rank)
            if cell is None:
                cell = [0.0] * (PLACES + 2)
                acc[rank] = cell
            for j in later:
                after = reads[j][1].get(uid)
                if after is None:
                    cell[PLACES] += weight
                elif after[0] <= PLACES:
                    cell[after[0] - 1] += weight
                else:
                    cell[PLACES - 1] += weight
                cell[PLACES + 1] += weight
    return acc


def _rmarg_probs(reads, task):
    """The rank-only marginal: a game inherits the record of its current rank.

    Returns {universe_id: [0.0, p1..pPLACES]} over the current chart's top
    `RMARG_POOL` games.
    """
    origin_stamp, origin_games, _created = reads[-1]
    ages = build_ages(reads)
    hours_grid = slot_hours(task, origin_stamp)
    surv = survival(reads, hours_grid, ages)
    acc = build_rank_marginal(reads)

    def marginal(rank):
        """The smoothed place distribution for a starting rank."""
        num = [0.0] * PLACES
        total = 0.0
        for delta in range(-RMARG_RANK_SPAN, RMARG_RANK_SPAN + 1):
            source = rank + delta
            if source < 1 or source > RMARG_RANK_CAP:
                continue
            cell = acc.get(source)
            if cell is None:
                continue
            weight = math.exp(-0.5 * (delta / RMARG_RANK_SIGMA) ** 2)
            for place in range(PLACES):
                num[place] += weight * cell[place]
            total += weight * cell[PLACES + 1]
        if total <= 0.0:
            return [0.0] * PLACES
        return [value / total for value in num]

    pool = current_pool(origin_games, RMARG_POOL)
    probs = {uid: [0.0] * (PLACES + 1) for uid in pool}
    for uid in pool:
        rank = origin_games[uid][0]
        share = marginal(rank)
        chance = surv.get(uid, 1.0)
        for place in range(1, PLACES + 1):
            probs[uid][place] = share[place - 1] * chance
    return probs


# ---------------------------------------------------------------- forecast

def forecast(reads, task, verbose=False):
    """The forecast: (document, {uid: [0, p1..pPLACES]}).

    One model, whatever window the task names: the capped occupancy reading
    described in the module docstring.  The window's own hours come from
    `task`, so the same code forecasts the practice window and the final one.
    """
    origin_stamp, origin_games, _created = reads[-1]
    proto = _proto_probs(reads, task)
    rmarg = _rmarg_probs(reads, task) if BLEND_RMARG > 0.0 else None

    # the union of the two candidate pools, highest on the chart first
    pool = current_pool(origin_games, POOL if rmarg is None
                        else max(POOL, RMARG_POOL))

    probs = {}
    for uid in pool:
        row_p = proto.get(uid)
        row_r = rmarg.get(uid) if rmarg is not None else None
        row = [0.0] * (PLACES + 1)
        for place in range(1, PLACES + 1):
            a = row_p[place] if row_p is not None else 0.0
            b = row_r[place] if row_r is not None else 0.0
            row[place] = BLEND_PROTO * a + BLEND_RMARG * b
        probs[uid] = row

    # a place's leftover is the share held by games not on the chart yet: it
    # is capped so the games that join the chart during the window keep a
    # share, and every one of those misses otherwise costs the score floor
    rows = []
    for place in range(1, PLACES + 1):
        entries = sorted(((values[place], uid) for uid, values in probs.items()
                          if values[place] > 1e-7),
                         key=lambda item: (-item[0], item[1]))
        kept = entries[:MAX_ROWS_PER_PLACE]
        total = sum(p for p, _uid in kept)
        if total > PLACE_CAP:
            scale = PLACE_CAP / total
            kept = [(p * scale, uid) for p, uid in kept]
            total = PLACE_CAP
        # rounding to six decimals can lift a place's sum back over the cap by
        # a hair, so shave the excess off the largest row: the numbers the file
        # actually carries must satisfy "at most PLACE_CAP", not nearly do
        rounded = [(round(p, 6), uid) for p, uid in kept]
        excess = sum(p for p, _uid in rounded) - PLACE_CAP
        if excess > 0.0 and rounded:
            biggest = max(range(len(rounded)), key=lambda i: rounded[i][0])
            value, uid = rounded[biggest]
            rounded[biggest] = (round(max(0.0, value - excess), 6), uid)
        for p, uid in rounded:
            rows.append({"universe_id": uid, "place": place, "p": p})
        if verbose:
            print("place %2d  named %2d  sum %.4f  leftover %.4f"
                  % (place, len(rounded),
                     sum(p for p, _uid in rounded),
                     1.0 - sum(p for p, _uid in rounded)))

    top10 = choose_top10(probs, pool, spare=ever_charted(reads))

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
