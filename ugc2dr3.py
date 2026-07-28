#!/usr/bin/env python3
"""
ugc2dr3.py: Convert UMIGURI chart .zip(s) to DanceRail3 .zip(s).

Usage:
    python ugc2dr3.py INPUT [INPUT ...] [-o OUTDIR] [--name BASE] [--level N]
                            [--ln-density 1/4] [--flick-tap] [--no-head-tap]
                            [--offset-sign +|-]

Each INPUT is a .zip (containing a .ugc plus its audio & jacket) or a folder of
zips. Every input zip becomes  DR_<originalname>.zip  containing the four DR3
files. Pillow is auto-installed if missing; ffmpeg must be in PATH.

The LN-centre code (cmd_addmiddle), head-tap overlay (cmd_forceln), timing math
(m2t) and bugged-note detection (find_bugged) are lifted VERBATIM from
dr3editor.py so behaviour matches the editor exactly.
"""

import os, re, sys, math, argparse, shutil, subprocess, glob

def ensure_package(import_name, pip_name=None):
    """Import a package, pip-installing it into THIS interpreter if missing.
    (Pillow is a Python library, not a PATH tool - it must live in the same
    Python that runs this script.)"""
    try:
        return __import__(import_name)
    except ImportError:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pip_name or import_name],
                           check=True)
            return __import__(import_name)
        except Exception:
            return None



# ============================================================================
#  PART 1: code reused verbatim from dr3editor.py (do not edit; keep in sync)
# ============================================================================
HIGHWAY_WIDTH = 16.0
OVERALL_MIN_WIDTH = 0.5
MAX_NOTE_WIDTH = 48.0

CHAIN_STARTS = {3, 5, 10, 19, 21, 23}
CHAIN_MIDS   = {6, 11, 12, 17, 19, 21, 23}
CHAIN_ENDS   = {4, 7, 18, 20, 22, 24}
CHAIN_START_TO_MID = {5: 6, 3: 11, 10: 17, 19: 19, 21: 21, 23: 23}
CHAIN_END_TO_MID   = {4: 11, 7: 6, 18: 17, 20: 19, 22: 21, 24: 23}

def clamp_width(w):
    return max(OVERALL_MIN_WIDTH, min(MAX_NOTE_WIDTH, w))

def notes_in_range(notes, m1, m2):
    if m1 == 0 and m2 == 0:
        return list(range(len(notes)))
    return [i for i, n in enumerate(notes) if m1 <= n['beat'] < m2]

def is_ln_child(notes, i):
    n = notes[i]; p = n['parent']
    if p < 0 or p >= len(notes) or p == i:
        return False
    if n['type'] not in (CHAIN_MIDS | CHAIN_ENDS):
        return False
    return notes[p]['beat'] <= n['beat'] + 0.001

def build_ln_children(notes):
    ch = {}
    for i in range(len(notes)):
        if is_ln_child(notes, i):
            ch.setdefault(notes[i]['parent'], []).append(i)
    return ch

def follow_chain(notes, start):
    children = {}
    for i, n in enumerate(notes):
        if is_ln_child(notes, i) and i != start:
            children[n['parent']] = i
    chain = [start]; cur = start
    while cur in children:
        cur = children[cur]; chain.append(cur)
    return chain

def select_densify_heads(notes, density):
    """Decide which LN heads cmd_addmiddle should re-centre.

    addmiddle deletes an LN's existing centre notes and lays fresh ones on a
    uniform `density` grid. For an LN whose UMIGURI centres are ALREADY denser
    than that grid, that would REDUCE the centre count and blockify its shape, so
    we only densify an LN when addmiddle would add MORE centres than it already
    has (this also covers start/end-only LNs, which have 0 centres and need some
    to be holdable). Returns (heads_to_densify, n_preserved).

    O(N): builds the child map once and walks each linear chain once."""
    ch = build_ln_children(notes)              # parent_idx -> [child_idx, ...]
    children = set()
    for lst in ch.values(): children.update(lst)
    heads_to_densify, preserved, N = [], 0, len(notes)
    for h in ch.keys():
        if h in children:                      # not a real chain head
            continue
        length, cur, last, steps = 1, h, h, 0
        while cur in ch:                       # walk the linear chain
            cur = ch[cur][0]; length += 1; last = cur; steps += 1
            if steps > N: break                # cycle guard
        existing_mids = length - 2             # head + mids + end
        sb, eb = notes[h]['beat'], notes[last]['beat']
        new_mids, p = 0, sb + density          # exactly what addmiddle's loop creates
        while p < eb - 0.0001 and new_mids <= N:
            new_mids += 1; p += density
        if new_mids > existing_mids:
            heads_to_densify.append(h)
        else:
            preserved += 1
    return heads_to_densify, preserved

def m2t(m, bpms, off):
    """Measure -> time (seconds) with multiple BPM changes. Verbatim from
    dr3editor.py.  bpms: sorted list of {'beat': measure, 'bpm': value}."""
    if not bpms: return m * 2.0 + off
    t = off
    for i, bp in enumerate(bpms):
        seg_start = bp['beat']; seg_bpm = bp['bpm']
        seg_end = bpms[i + 1]['beat'] if i + 1 < len(bpms) else float('inf')
        if m <= seg_start: break
        if m <= seg_end:
            t += (m - seg_start) * 240.0 / seg_bpm
            return t
        t += (seg_end - seg_start) * 240.0 / seg_bpm
    else:
        return t
    t += (m - bpms[0]['beat']) * 240.0 / bpms[0]['bpm']
    return t

def find_bugged(notes, bpms, off, has_audio=True):
    """Reproduces dr3editor.py's bugged-note detection (selection filter
    '...are bugged' + save-time warning): a chain mid/end TYPE that is neither a
    genuine LN child nor an LN head, an invalid NSC string, or a note that lands
    strictly before audio start (t < -0.001).  t == 0 is NOT bugged."""
    chain_role = CHAIN_MIDS | CHAIN_ENDS
    heads = set(build_ln_children(notes).keys())
    bugged = set()
    for i, n in enumerate(notes):
        tt = n['type']
        if tt in chain_role:
            if not is_ln_child(notes, i) and i not in heads:
                bugged.add(i); continue
        nsc = n.get('nsc', '0')
        if nsc and nsc != '0' and nsc != '1':
            bad = False
            if ':' not in nsc:
                try: float(nsc)
                except: bad = True
            else:
                try:
                    for pair in nsc.split(';'):
                        a, b = pair.split(':'); float(a); float(b)
                except: bad = True
            if bad: bugged.add(i); continue
        if has_audio and m2t(n['beat'], bpms, off) < -0.001:
            bugged.add(i); continue
    return bugged

def delete_bugged(notes, bpms, off):
    """Remove bugged notes so the chart can't crash DR3. If a bugged note is part
    of an LN, the WHOLE chain is removed (deleting one link would orphan the rest,
    which are then bugged too). Iterates until the chart is clean. Returns
    (clean_notes, n_removed)."""
    removed = 0
    for _ in range(20):
        bugged = find_bugged(notes, bpms, off, has_audio=True)
        if not bugged:
            break
        ch = build_ln_children(notes)
        drop = set()
        for i in bugged:
            root, guard = i, 0                       # climb to the chain's head
            while is_ln_child(notes, root) and 0 <= notes[root]['parent'] < len(notes):
                root = notes[root]['parent']; guard += 1
                if guard > len(notes): break
            stack = [root]                           # then drop the whole chain
            while stack:
                x = stack.pop()
                if x in drop: continue
                drop.add(x); stack.extend(ch.get(x, []))
        remap, out = {}, []
        for k, n in enumerate(notes):
            if k in drop: continue
            remap[k] = len(out); out.append(n)
        for n in out:
            n['parent'] = remap.get(n['parent'], -1) if n['parent'] >= 0 else -1
        for k, n in enumerate(out): n['idx'] = k
        notes = out; removed += len(drop)
    return notes, removed

def dedupe_flicks(notes):
    """Delete type-9 (any-direction) flicks that perfectly overlap a directional
    flick (13/14/15/16) at the same beat + x + width. In DR3 the type-9 there is
    redundant - the directional flick already needs a flick - so it does nothing
    but clutter the visuals. O(N). Returns (notes, n_removed)."""
    DIRQ = (13, 14, 15, 16)
    key = lambda n: (round(n['beat'], 3), round(n['x'], 3), round(n['width'], 3))
    dir_keys = {key(n) for n in notes if n['type'] in DIRQ}
    if not dir_keys:
        return notes, 0
    drop = {i for i, n in enumerate(notes)
            if n['type'] == 9 and key(n) in dir_keys and '_tower' not in n}
    if not drop:
        return notes, 0
    remap, out = {}, []
    for k, n in enumerate(notes):
        if k in drop: continue
        remap[k] = len(out); out.append(n)
    for n in out:
        n['parent'] = remap.get(n['parent'], -1) if n['parent'] >= 0 else -1
    for k, n in enumerate(out): n['idx'] = k
    return out, len(drop)

BOMB_WINDOW_S = 0.0075      # +-7.5 ms; a red blocks the lane for this long either side

def dedupe_bombs(notes, bpms, offset):
    """Delete red bomb notes (type 10) that make another note impossible to hit.

    A red is treated as a 2-D box: its lane span [x, x+width] across a time window of
    +-BOMB_WINDOW_S around its own hit time. Ordinary notes stay 1-dimensional - a
    lane span at a single instant. If a non-bomb note's lane span falls entirely
    inside a red's box, that note cannot be touched without hitting the bomb, so the
    red is removed. Reds overlapping only other reds are left alone, as is a red
    sitting inside a WIDER note (there is still bomb-free lane to hit).

    The window is measured in SECONDS via m2t, not in measures, so local BPM spikes
    are handled correctly: notes a few thousandths of a measure apart during a
    5000 BPM burst are only microseconds apart and are correctly seen as coincident.
    Red notes are never part of an LN, so removal is always safe.
    Returns (notes, n_removed)."""
    times = [m2t(n['beat'], bpms, offset) for n in notes]
    others = sorted((times[i], n['x'], n['x'] + n['width'])
                    for i, n in enumerate(notes) if n['type'] != 10)
    if not others:
        return notes, 0
    otimes = [o[0] for o in others]
    drop = set()
    for i, n in enumerate(notes):
        if n['type'] != 10:
            continue
        r0, r1 = n['x'], n['x'] + n['width']
        t = times[i]
        lo = _bisect.bisect_left(otimes, t - BOMB_WINDOW_S)
        hi = _bisect.bisect_right(otimes, t + BOMB_WINDOW_S)
        for k in range(lo, hi):
            _tk, a, b = others[k]
            if r0 - 1e-6 <= a and b <= r1 + 1e-6:      # note span sits inside the bomb
                drop.add(i); break
    if not drop:
        return notes, 0
    remap, out = {}, []
    for k, n in enumerate(notes):
        if k in drop: continue
        remap[k] = len(out); out.append(n)
    for n in out:
        n['parent'] = remap.get(n['parent'], -1) if n['parent'] >= 0 else -1
    for k, n in enumerate(out): n['idx'] = k
    return out, len(drop)

NPS_LIMIT = 250          # notes/sec the game struggles past
NPS_MIN_RUN = 3          # consecutive quarter-note beats that must all exceed it

def _drop_notes(notes, drop):
    """Remove the note indices in `drop`, remapping parent array-indices and idx."""
    remap, out = {}, []
    for k, n in enumerate(notes):
        if k in drop: continue
        remap[k] = len(out); out.append(n)
    for n in out:
        n['parent'] = remap.get(n['parent'], -1) if n['parent'] >= 0 else -1
    for k, n in enumerate(out): n['idx'] = k
    return out

def _beat_nps(notes, bpms):
    """Per quarter-note-beat NPS, matching dr3editor's filter: bucket = floor(beat*4)/4,
    NPS = notes_in_bucket / (60/bpm_at_bucket). Returns {bucket: nps}."""
    import math
    from collections import Counter
    cnt = Counter(round(math.floor(n['beat'] * 4) / 4, 3) for n in notes)
    def bpm_at(bk):
        b = bpms[0]['bpm'] if bpms else 120.0
        for e in bpms:
            if e['beat'] <= bk: b = e['bpm']
        return b
    return {bk: c / (60.0 / bpm_at(bk)) for bk, c in cnt.items()}

def _hot_runs(nps):
    """[start,end) beat ranges of >= NPS_MIN_RUN consecutive quarter-beats at >= NPS_LIMIT."""
    hot = sorted(bk for bk, v in nps.items() if v >= NPS_LIMIT)
    runs, i = [], 0
    while i < len(hot):
        j = i
        while j + 1 < len(hot) and abs(hot[j + 1] - hot[j] - 0.25) < 1e-6:
            j += 1
        if j - i + 1 >= NPS_MIN_RUN:
            runs.append((hot[i], hot[j] + 0.25))
        i = j + 1
    return runs

def reduce_stacked_nps(notes, bpms, half_density):
    """Tame impossibly dense passages made of perfectly-stacked LNs. In every sustained
    hot region (>= NPS_MIN_RUN consecutive beats at >= NPS_LIMIT nps):
      A) delete HALF of the fully-redundant LNs there - an LN qualifies only if EVERY
         one of its notes overlaps a note of another LN (same beat+x+width). Deleting
         whole chains is safe: the head's tap is a separate note and survives.
      B) if the region is still hot, coarsen every LN that intersects it to 1/2 centres
         (cmd_addmiddle replaces the existing centres, halving them).
    Returns (notes, n_lns_deleted, n_lns_coarsened). ~O(N)."""
    from collections import Counter
    runs = _hot_runs(_beat_nps(notes, bpms))
    if not runs:
        return notes, 0, 0

    def chains(ns):
        ch = build_ln_children(ns)
        hs = [i for i in ch if not is_ln_child(ns, i)]
        def walk(h):
            c = [h]; cur = h
            while cur in ch: cur = ch[cur][-1]; c.append(cur)
            return c
        return ch, hs, walk
    key = lambda ns, i: (round(ns[i]['beat'], 3), round(ns[i]['x'], 3), round(ns[i]['width'], 3))
    def hits(runlist, b0, b1):
        return any(b0 < r1 and b1 > r0 for r0, r1 in runlist)

    _, heads, walk = chains(notes)
    allc = {h: walk(h) for h in heads}                    # walk each chain once
    cand = [c for c in allc.values()
            if hits(runs, notes[c[0]]['beat'], notes[c[-1]]['beat'])]
    deleted = 0
    if cand:
        keycount = Counter(key(notes, i) for c in allc.values() for i in c)   # over all LN nodes
        redundant = [c for c in cand if all(keycount[key(notes, i)] >= 2 for i in c)]
        target = len(redundant) // 2                      # delete half -> ~halve the nps
        drop = set()
        for c in redundant:
            if deleted >= target: break
            if all(keycount[key(notes, i)] >= 2 for i in c):   # keeps every lane covered >=1
                for i in c:
                    keycount[key(notes, i)] -= 1; drop.add(i)
                deleted += 1
        if drop:
            notes = _drop_notes(notes, drop)

    coarsened = 0
    runs2 = _hot_runs(_beat_nps(notes, bpms))
    if runs2:
        _, heads2, walk2 = chains(notes)
        allc2 = {h: walk2(h) for h in heads2}
        hot_heads = [h for h, c in allc2.items()
                     if hits(runs2, notes[c[0]]['beat'], notes[c[-1]]['beat'])]
        if hot_heads:
            ok, notes, _ = cmd_addmiddle(notes, 0, 0, half_density, indices=hot_heads)
            if ok: coarsened = len(hot_heads)
    return notes, deleted, coarsened

def cmd_addmiddle(notes, m1, m2, density, indices=None):
    if density <= 0: return False, notes, "Invalid density"
    if indices is None: indices = notes_in_range(notes, m1, m2)
    ch = build_ln_children(notes)
    parents_set = set(ch.keys())
    children_set = set()
    for lst in ch.values(): children_set.update(lst)
    starts = [i for i in indices if i in parents_set and i not in children_set]
    if not starts: return False, notes, "No chain starts"
    all_remove = set(); all_new = []
    for si in starts:
        sn = notes[si]
        chain = [si]; cur = si           # walk via the already-built `ch` map (O(chain)
        while cur in ch:                 # per start, not O(N) like follow_chain) so that
            cur = ch[cur][-1]; chain.append(cur)   # huge charts (100k+ notes) stay fast
        if len(chain) < 2: continue
        li = chain[-1]; ln = notes[li]
        sb, eb = sn['beat'], ln['beat']
        existing_mid_types = [notes[ci]['type'] for ci in chain[1:-1]
                              if notes[ci]['type'] in CHAIN_MIDS]
        if existing_mid_types:
            mt = existing_mid_types[0]
        elif sn['type'] in CHAIN_START_TO_MID:
            mt = CHAIN_START_TO_MID[sn['type']]
        elif ln['type'] in CHAIN_END_TO_MID:
            mt = CHAIN_END_TO_MID[ln['type']]
        else:
            mt = 6
        shape_pts = [(notes[ci]['beat'], notes[ci]['x'], notes[ci]['width'])
                     for ci in chain]
        shape_pts.sort(key=lambda p: p[0])
        def interp_at(b):
            if b <= shape_pts[0][0]: return shape_pts[0][1], shape_pts[0][2]
            if b >= shape_pts[-1][0]: return shape_pts[-1][1], shape_pts[-1][2]
            for k in range(len(shape_pts) - 1):
                b0, x0, w0 = shape_pts[k]
                b1, x1, w1 = shape_pts[k + 1]
                if b0 <= b <= b1:
                    if b1 - b0 < 0.0001: return x0, w0
                    t = (b - b0) / (b1 - b0)
                    return x0 + (x1 - x0) * t, w0 + (w1 - w0) * t
            return shape_pts[-1][1], shape_pts[-1][2]
        all_remove |= set(chain[1:-1])
        pos = []; p = sb + density
        while p < eb - 0.0001: pos.append(round(p, 5)); p += density
        mids_list = []
        for b in pos:
            ix, iw = interp_at(b)
            mids_list.append({'idx': -1, 'type': mt, 'beat': b,
                'x': round(ix, 5), 'width': round(iw, 5),
                'nsc': '0', 'attr': '', 'parent': -1})
        all_new.append((si, li, mids_list))
    nn = []; pos_map = {}
    for i, n in enumerate(notes):
        if i not in all_remove:
            pos_map[i] = len(nn); nn.append(n)
    base = max((n['idx'] for n in nn), default=-1) + 1
    next_file_idx = max((n.get('file_idx', n['idx']) for n in nn), default=-1) + 1
    for si, li, mids in all_new:
        sp, lp = pos_map[si], pos_map[li]
        if mids:
            for j, m in enumerate(mids):
                m['idx'] = base; m['file_idx'] = next_file_idx
                base += 1; next_file_idx += 1
                nn.append(m)
            mids[0]['parent'] = nn[sp]['idx']
            for j in range(1, len(mids)): mids[j]['parent'] = mids[j - 1]['idx']
            nn[lp]['parent'] = mids[-1]['idx']
        else:
            nn[lp]['parent'] = nn[sp]['idx']
    nn.sort(key=lambda n: (n['beat'], n['type'], n['x']))
    remap = {n['idx']: i for i, n in enumerate(nn)}
    for n in nn:
        if n['parent'] >= 0: n['parent'] = remap.get(n['parent'], -1)
    for i, n in enumerate(nn): n['idx'] = i
    return True, nn, f"Processed {len(starts)} chains"

def cmd_forceln(notes, m1, m2, indices=None, tap_type=1):
    if indices is None: indices = notes_in_range(notes, m1, m2)
    excluded = {1, 2, 25, 26, 27}
    eligible = []
    for i in indices:
        t = notes[i]['type']
        if t in excluded: continue
        if is_ln_child(notes, i): continue
        eligible.append(i)
    starts = eligible
    if not starts: return False, notes, "No eligible notes in range"
    # precompute existing tap_type positions once (O(N)) so the dup check is O(1) per head
    key = lambda n: (round(n['beat'], 3), round(n['x'], 3), round(n['width'], 3))
    existing = {key(n) for n in notes if n['type'] == tap_type}
    new_taps = []
    for si in starts:
        s = notes[si]
        if key(s) in existing: continue   # matches editor: check original notes only
        new_taps.append({'idx': -1, 'type': tap_type, 'beat': s['beat'], 'x': s['x'],
            'width': s['width'], 'nsc': '0', 'attr': '', 'parent': -1})
    if not new_taps: return False, notes, f"All chain starts already have type-{tap_type} taps"
    base = max((n['idx'] for n in notes), default=-1) + 1
    for j, t in enumerate(new_taps): t['idx'] = base + j
    nn = list(notes) + new_taps
    nn.sort(key=lambda n: (n['beat'], n['type'], n['x']))
    om = {n['idx']: i for i, n in enumerate(nn)}
    for n in nn:
        if n['parent'] >= 0: n['parent'] = om.get(n['parent'], -1)
    for i, n in enumerate(nn): n['idx'] = i
    return True, nn, f"Added {len(new_taps)} forced type-{tap_type} taps"

def t2m(t, bpms, off):
    """Time (seconds) -> measure position; exact inverse of m2t."""
    if not bpms: return (t - off) / 2.0
    cur_t = off
    for i, bp in enumerate(bpms):
        seg_start, seg_bpm = bp['beat'], bp['bpm']
        seg_end = bpms[i + 1]['beat'] if i + 1 < len(bpms) else float('inf')
        seg_dur = (seg_end - seg_start) * 240.0 / seg_bpm
        if t <= cur_t + seg_dur or i == len(bpms) - 1:
            return seg_start + (t - cur_t) * seg_bpm / 240.0
        cur_t += seg_dur
    return bpms[-1]['beat']

def format_chart(header, notes):
    def fmt(v):
        if isinstance(v, float) and v == int(v): return str(int(v))
        if isinstance(v, float): return f"{v:.5f}".rstrip('0').rstrip('.')
        return str(v)
    lines = list(header)
    for i, n in enumerate(notes):
        p = n['parent'] if n['parent'] >= 0 else i
        nsc = n.get('nsc', '0'); attr = n.get('attr', '')
        suffix = f"<{attr}>" if attr else ""
        lines.append(f"<{i}><{n['type']}><{fmt(n['beat'])}><{fmt(n['x'])}>"
                     f"<{fmt(n['width'])}><{nsc}><{p}>{suffix}")
    return '\r\n'.join(lines) + '\r\n'

# ============================================================================
#  PART 2: UMIGURI (.ugc) parser
# ============================================================================
def _dec(ch):
    """Single-char lane/width code: 0-9, A-G  ->  0-16 (base 17)."""
    try: return int(ch, 17)
    except ValueError: return 0

class UgcChart:
    def __init__(self):
        self.meta = {}            # raw @ string fields
        self.ticks_per_beat = 480
        self.beats = []           # list of (start_measure, num, den)
        self.bpms = []            # list of (abs_tick, value)
        self.bgmofs = 0.0
        self.groups = []          # list of dicts: {head_tick, body, children:[(abs_tick, body)], til}
        self.tils = {}            # timeline id -> [(abs_tick, speed)]  (@TIL keyframes)
        self.maintil = 0          # @MAINTIL
        self.spdmod = []          # [(abs_tick, speed)]  (@SPDMOD keyframes)
        self.has_spdfld = False   # @SPDDEF/@SPDFLD seen (unsupported, rare)
        self.bpm0_dropped = 0     # duplicate measure-0 BPM entries removed
        self.bpm0_kept = None
        self.bpm0_differed = False

    # measure'tick -> absolute tick, honouring @BEAT time-signature segments
    def mt_to_tick(self, measure, tick):
        """Absolute tick of `measure`'@'`tick`. An "@BEAT m num den" line sets the
        time signature FROM measure m onward, so every measure before the first
        @BEAT keeps the 4/4 default. Getting that wrong silently rescales the
        whole chart: top-level notes (which come through here) shift while LN
        children (offset from their head by a raw tick delta) do not, so note
        types drift apart from each other."""
        full = self.ticks_per_beat * 4          # one 4/4 measure, in ticks
        if not self.beats:
            return measure * full + tick
        segs = sorted(self.beats, key=lambda b: b[0])
        total = 0
        cur = 0
        num, den = 4, 4                          # default until the first @BEAT
        for sm, snum, sden in segs:
            if sm >= measure:
                break                            # this signature starts after us
            if sm > cur:
                total += (sm - cur) * (full * num // den)
                cur = sm
            num, den = snum, sden
        if measure > cur:
            total += (measure - cur) * (full * num // den)
        return total + tick

    # DR3 measure position ("ichi") for an absolute tick. DR3 measure = 4 beats,
    # so ichi = tick / (ticks_per_beat * 4). This yields correct AUDIO TIMING
    # for any time signature because both engines reduce to tick/(8*bpm) seconds.
    def ichi(self, abs_tick):
        return abs_tick / (self.ticks_per_beat * 4.0)


def parse_ugc(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    lines = text.split('\n')
    c = UgcChart()
    in_notes = False
    cur_group = None
    bpm_lines = []
    til_lines, spd_lines = [], []
    cur_til = 0
    for raw in lines:
        line = raw.rstrip('\r')
        if not line.strip():
            continue
        if not in_notes:
            if line.startswith('@ENDHEAD'):
                in_notes = True
                continue
            if not line.startswith('@'):
                continue
            parts = line.split('\t')
            tag = parts[0][1:]
            val = parts[1] if len(parts) > 1 else ''
            if tag == 'TICKS':
                c.ticks_per_beat = int(val)
            elif tag == 'BEAT':
                # @BEAT <measure> <num> <den>
                c.beats.append((int(parts[1]), int(parts[2]), int(parts[3])))
            elif tag == 'BPM':
                bpm_lines.append(parts)          # resolve ticks after BEAT known
            elif tag == 'BGMOFS':
                c.bgmofs = float(val)
            elif tag == 'TIL':
                til_lines.append(parts)          # @TIL id bar'tick speed
            elif tag == 'MAINTIL':
                try: c.maintil = int(val)
                except ValueError: pass
            elif tag == 'SPDMOD':
                spd_lines.append(parts)          # @SPDMOD bar'tick speed
            elif tag in ('SPDDEF', 'SPDFLD'):
                c.has_spdfld = True
            else:
                c.meta[tag] = parts[1:] if len(parts) > 1 else ['']
            continue
        # ----- note section -----
        if line.startswith('@USETIL'):
            pp = line.split('\t')
            try: cur_til = int(pp[1]) if len(pp) > 1 else 0
            except ValueError: cur_til = 0
            continue
        m = re.match(r"^#(\d+)'(\d+):(.+)$", line)
        if m:
            at = c.mt_to_tick(int(m.group(1)), int(m.group(2)))
            cur_group = {'tick': at, 'body': m.group(3), 'children': [], 'til': cur_til}
            c.groups.append(cur_group)
            continue
        m = re.match(r"^#(\d+)>(.+)$", line)
        if m and cur_group is not None:
            # child reltick is cumulative from the head's absolute tick
            if cur_group['til'] != cur_til: cur_group['til_mixed'] = True
            cur_group['children'].append((cur_group['tick'] + int(m.group(1)), m.group(2)))
            continue
    # resolve BPM change positions now that BEAT segments are known
    for parts in bpm_lines:
        mm = re.match(r"(\d+)'(\d+)", parts[1])
        at = c.mt_to_tick(int(mm.group(1)), int(mm.group(2)))
        c.bpms.append((at, float(parts[2])))
    if not c.bpms:
        c.bpms.append((0, 120.0))
    c.bpms.sort()
    # DR3 treats the BPM entry at measure 0 as the special "initial" tempo. A second
    # entry there (UMIGURI tolerates duplicates; some charts ship them) makes every
    # note spawn stuck at the judgment line - the chart becomes unplayable. Collapse
    # measure-0 entries into one, keeping the FASTEST. Duplicates anywhere else are
    # harmless in DR3 and are left exactly as they are.
    at_zero = [v for t, v in c.bpms if t == 0]
    if len(at_zero) > 1:
        c.bpm0_dropped = len(at_zero) - 1
        c.bpm0_kept = max(at_zero)
        c.bpm0_differed = (min(at_zero) != max(at_zero))
        c.bpms = [(0, c.bpm0_kept)] + [(t, v) for t, v in c.bpms if t != 0]
    # resolve @TIL / @SPDMOD positions (bar'tick) the same way
    for parts in til_lines:
        try:
            tid = int(parts[1])
            mm = re.match(r"(\d+)'(\d+)", parts[2])
            c.tils.setdefault(tid, []).append(
                (c.mt_to_tick(int(mm.group(1)), int(mm.group(2))), float(parts[3])))
        except (ValueError, AttributeError, IndexError):
            pass
    for kf in c.tils.values(): kf.sort()
    for parts in spd_lines:
        try:
            mm = re.match(r"(\d+)'(\d+)", parts[1])
            c.spdmod.append((c.mt_to_tick(int(mm.group(1)), int(mm.group(2))), float(parts[2])))
        except (ValueError, AttributeError, IndexError):
            pass
    c.spdmod.sort()
    return c


def parse_body(body):
    """Return (type_char, lane, width, extra) ; lane/width None for bare s/c."""
    tc = body[0]
    if len(body) >= 3 and body[1] in '0123456789ABCDEFG' and body[2] in '0123456789ABCDEFG':
        return tc, _dec(body[1]), _dec(body[2]), body[3:]
    return tc, None, None, body[1:]

# ============================================================================
#  PART 3: mapping UMIGURI notes -> DR3 notes
# ============================================================================
def sc_at_beat(scs, beat):
    """SC (scroll-speed) multiplier active at a measure position. scs is the
    converter\'s [(speed, measure), ...] list (ascending). Defaults to 1.0."""
    val = 1.0
    if scs:
        val = scs[0][0] if scs[0][1] <= beat + 1e-9 else 1.0
        for speed, m in scs:
            if m <= beat + 1e-9: val = speed
            else: break
    return val

TOWER_MAX_LAYERS = 5        # past ~5 layers the DR3 NSC tower illusion breaks down

def stack_notes_3d(layers, scs):
    """Port of dr3editor.py\'s "3D Stacker": turn a set of perfectly-overlapping
    notes into a vertical 3D stack via a per-layer NSC illusion. layers is the
    note dicts in stacking order (index 0 = ground). Mutates them in place.

    The math is copied verbatim from the editor so the in-game look matches:
    ground layer gets an identity curve (scaled by SC) so it scrolls like a
    normal note; each higher layer k gets a height h, a scroll rate s=1/(1-h),
    a hover c, a piecewise NSC curve, an outward centre shift and a width
    scale-up for the extra perceived distance. Every apparent distance is
    multiplied by the local SC so the stack tracks surrounding notes."""
    import math
    N = len(layers)
    if N < 2:
        return
    C0, H0 = 0.0125, 0.30
    HMAX = 0.30 if N == 2 else (0.35 if N == 3 else 0.40)
    T    = 0.6  if N == 2 else 1.0
    ref_shift = 0.25
    SHIFT_CAL = ref_shift / (math.sqrt(1.0 / 0.7) - 1.0)
    def _f(v):
        t = f"{v:.5f}".rstrip('0').rstrip('.')
        return t if t else "0"
    SC = sc_at_beat(scs, layers[0]['beat'])
    for k, n in enumerate(layers):
        if k == 0:
            n['nsc'] = f"10:{_f(round(10.0 * SC, 5))};-1:{_f(round(-SC, 5))}"
            continue
        frac = k / (N - 1)
        h = HMAX * frac
        s = 1.0 / (1.0 - h)
        c = C0 * (h / H0)
        kg = math.sqrt(s)
        n['nsc'] = (f"2:{_f(round(2.0 * SC, 5))};"
                    f"{_f(T)}:{_f(round(SC * (T * s + c), 5))};"
                    f"0:{_f(round(SC * c, 5))}")
        ow = n['width']; oc = n['x'] + ow / 2.0
        nw = clamp_width(ow * (1.0 + 0.8 * (kg - 1.0)))
        nc = oc + SHIFT_CAL * (kg - 1.0) * (oc - 8.0)
        n['width'] = round(nw, 5)
        n['x'] = round(nc - nw / 2.0, 5)

LN_KINDS = {'h': (3, 11, 4), 'H': (3, 11, 4),      # holds, air-holds -> orange hold LN
            's': (5, 6, 7),  'S': (5, 6, 7)}        # slides, air-slides -> blue slide LN

CRUSH_SLAB_SPAN = 0.02      # measures; shorter than this a density-0 air-crush is a slab
CRUSH_TOWER_SPAN = 0.25     # measures; a static slab ladder longer than this is a stream,
                            #            not a vertical tower (the 3D stack is instantaneous)
TOWER_INSTANT_SPAN = 0.03   # measures; a height-changing ribbon this short is drawn all at
                            #           once, so its joints stack up as a vertical tower

_B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _b36(c):
    return _B36.index(c) if c in _B36 else 0

def crush_height(extra, new_fmt):
    """Height of an AIR-CRUSH head/joint from the chars after its lane+width.

    Two payload layouts exist and they use different units:
      new (UMIGURI v8): 2-char height, colour, optional ",interval" - the pair is
                        height x 10, so '30' -> 108 -> height 10.8
      old (pre-v8):     1-char height then colour - the char IS the height
    Returns the height in real units, or None when the joint carries no height
    (a bare joint, which inherits the previous point's)."""
    head = extra.split(',', 1)[0]
    if not head:
        return None
    if new_fmt:
        return (_b36(head[0]) * 36 + _b36(head[1])) / 10.0 if len(head) >= 2 else None
    return float(_b36(head[0]))

def _crush_geo(g):
    """Joint points of an AIR-CRUSH group.

    An air-crush is a ribbon through its joints; PenguinTools' converter turns
    every consecutive pair into one segment carrying a start and end height, so
    the joints are the ribbon's structural points. Bare joints (no lane given)
    inherit the previous point's position and height.

    Returns (points, dens) where points is [(tick, x, w, height), ...] starting
    with the head, and dens is the ",interval" token (None when absent)."""
    tc, lane, width, extra = parse_body(g['body'])
    dens = extra.split(',', 1)[1].strip() if ',' in extra else None
    # new layout carries 2 height chars + a colour before any comma
    new_fmt = len(extra.split(',', 1)[0]) >= 3
    x = lane if lane is not None else 0
    w = width if width is not None else 1
    h = crush_height(extra, new_fmt)
    if h is None: h = 0.0
    pts = [(g['tick'], x, w, h)]
    for ctick, cbody in sorted(g['children']):              # joints in time order
        _t, cl, cw, cextra = parse_body(cbody)
        if cl is not None:
            x, w = cl, cw                                   # bare joints inherit position
            ch = crush_height(cextra, new_fmt)
            if ch is not None: h = ch
        pts.append((ctick, x, w, h))
    return pts, dens

def air_crush_plan(chart):
    """Decide, for every AIR-CRUSH ('C') note, how it becomes DR3 notes.

    An air-crush is an airborne ribbon drawn through its joints. How many slabs
    it shows depends on the ",interval" field and on whether it changes height:

      density 0    -> AIR-TRACE, a bare guide line: decorative, dropped
                      (unless short + static + unlinked, i.e. really one slab)
      density N>0  -> a slab every N ticks: duration // N + 1 slabs
      density $ / no field:
          height varies across the joints -> the ribbon is a stack, and each
              JOINT POINT is one slab (head + children), with lane, width and
              height interpolated between head and endpoint
          height constant -> a flat bar: one slab

    Each slab becomes a type-9 flick. A ribbon whose slabs are stacked at
    essentially one instant is a vertical TOWER, rendered with the 3D illusion:
      * a static slab ladder (interval-driven) shorter than CRUSH_TOWER_SPAN
      * a height-changing ribbon shorter than TOWER_INSTANT_SPAN
    The stack may lean (centre drifting) or taper (width changing) exactly as the
    source does, because both are carried by the interpolation. A ribbon that
    genuinely advances in time is left as separate flicks along its path.

    Returns {group_index: {'slabs': [(tick, x, w, height), ...], 'tower': bool}}.
    Dropped decorative lines are absent from the result.
    """
    tpm = chart.ticks_per_beat * 4
    geo = {}
    for i, g in enumerate(chart.groups):
        if not g['body'] or g['body'][0] != 'C':
            continue
        geo[i] = _crush_geo(g)
    # chaining test (density-0 polylines) reuses head/endpoint coincidence
    starts, ends = {}, {}
    for i, (pts, dens) in geo.items():
        starts[(pts[0][0], pts[0][1], pts[0][2])] = True
        ends[(pts[-1][0], pts[-1][1], pts[-1][2])] = True
    plan = {}
    for i, (pts, dens) in geo.items():
        t0, x, w, h = pts[0]
        end, ex, ew, eh = pts[-1]
        dur = end - t0
        span = dur / tpm if tpm else 0.0
        static = (x == ex and w == ew)
        moves_h = any(abs(pt[3] - h) > 1e-9 for pt in pts)
        n_slabs = 1
        tower = False
        if dens == '0':
            linked = ends.get((t0, x, w)) or starts.get((end, ex, ew))
            if linked or not static or span >= CRUSH_SLAB_SPAN:
                continue                                    # decorative line: drop
        elif dens not in (None, '$'):
            try:
                interval = int(dens)
            except ValueError:
                interval = 0
            if interval > 0:
                n_slabs = max(1, dur // interval + 1)
                tower = static and span < CRUSH_TOWER_SPAN and n_slabs >= 2
        if n_slabs == 1 and moves_h:
            # a height-changing ribbon: one slab per joint point
            n_slabs = max(1, len(pts))
            tower = span < TOWER_INSTANT_SPAN and n_slabs >= 2
        # --- lay the slabs down along the ribbon ---
        slabs = []
        for k in range(n_slabs):
            f = 0.0 if n_slabs == 1 else k / (n_slabs - 1)
            slabs.append((int(round(t0 + f * dur)), x + f * (ex - x),
                          w + f * (ew - w), h + f * (eh - h)))
        plan[i] = {'slabs': slabs, 'tower': tower}
    return plan

def air_to_flick(extra):
    """extra like 'UCN','DLN','URN' -> DR3 flick type."""
    vdir = extra[0] if len(extra) >= 1 else 'U'
    hdir = extra[1] if len(extra) >= 2 else 'C'
    if hdir == 'L': return 13          # Flick L
    if hdir == 'R': return 14          # Flick R
    return 16 if vdir == 'D' else 15   # down / up


# ============================================================================
#  PART 3b: UGC scroll-speed gimmicks (@TIL/@USETIL/@SPDMOD) -> DR3 SC + NSC
#
#  Both engines share the same model: scroll speed is a STEP FUNCTION over time,
#  and a note's visual position is the time-integral of that speed from "now" to
#  its hit time. DR3 offers three tools (semantics verified in TheGameManager.cs
#  and TheOnpu.cs):
#    * #SC/#SCI      - the global step function. Applies to every normal note.
#    * simple NSC    - a constant multiplier on the SC-integrated distance.
#    * advanced NSC  - per-note piecewise-LINEAR curve "A:B;..." mapping real time
#                      to visual approach; keyframe time=BPMCurve(ichi-A), value=
#                      BPMCurve(ichi-B). It BYPASSES SC entirely, and such notes
#                      only spawn within 10s of their hit (z=300 before the first
#                      pair). LN links have their nsc force-reset by the game, so
#                      whole LNs can only ever follow SC.
#
#  Because UGC speeds are step functions, remaining-distance D(t) is piecewise
#  linear in t, which advanced NSC represents EXACTLY (one A:B pair per speed
#  boundary). So: one timeline becomes the SC lane; standalone notes on other
#  timelines get simple NSC (constant speed ratio) or exact advanced NSC curves.
# ============================================================================
import bisect as _bisect

NSC_WINDOW_S = 9.8        # advanced-NSC notes may exist only <10s before hit
NSC_MAX_PAIRS = 60        # cap per note (simplified with increasing tolerance)

class ScrollFX:
    """Effective scroll-speed step functions per timeline (@TIL x @SPDMOD),
    precomputed in both measure- and time-space for fast queries."""
    PRE = -64.0           # a boundary far before the chart; speed defaults to 1.0

    def __init__(self, chart, bpms, offset):
        self.ch, self.bpms, self.off = chart, bpms, offset
        ids = set(chart.tils.keys()) | {0} | {g.get('til', 0) for g in chart.groups}
        spd = [(chart.ichi(t), v) for t, v in chart.spdmod]
        self.tl = {}
        for tid in ids:
            til = [(chart.ichi(t), v) for t, v in chart.tils.get(tid, [])]
            bounds = sorted({self.PRE} | {m for m, _ in til} | {m for m, _ in spd})
            def at(kf, m):                               # step-function lookup
                v = 1.0
                for mm, vv in kf:
                    if mm <= m + 1e-9: v = vv
                    else: break
                return v
            segs = [(m, at(til, m) * at(spd, m)) for m in bounds]
            merged = [segs[0]]                           # merge equal neighbours
            for m, v in segs[1:]:
                if abs(v - merged[-1][1]) > 1e-9: merged.append((m, v))
            tms = [m2t(m, bpms, offset) for m, _ in merged]
            integ = [0.0]
            for i in range(1, len(tms)):
                integ.append(integ[-1] + merged[i - 1][1] * (tms[i] - tms[i - 1]))
            self.tl[tid] = {'m': [m for m, _ in merged], 'v': [v for _, v in merged],
                            't': tms, 'I': integ}

    def _seg(self, tid, t):
        d = self.tl[tid]
        return max(0, _bisect.bisect_right(d['t'], t) - 1)

    def integral(self, tid, t):
        d = self.tl[tid]; i = self._seg(tid, t)
        return d['I'][i] + d['v'][i] * (t - d['t'][i])

    def remaining(self, tid, t, T):
        """Seconds of scroll left at time t for a note hitting at time T."""
        return self.integral(tid, T) - self.integral(tid, t)

    def boundaries(self, tid, t1, t2):
        d = self.tl[tid]
        i = _bisect.bisect_right(d['t'], t1)
        out = []
        while i < len(d['t']) and d['t'][i] < t2 - 1e-9:
            out.append(d['t'][i]); i += 1
        return out

    def equal(self, a, b):
        da, db = self.tl[a], self.tl[b]
        bounds = sorted(set(da['m']) | set(db['m']))
        def at(d, m):
            i = max(0, _bisect.bisect_right(d['m'], m + 1e-9) - 1)
            return d['v'][i]
        return all(abs(at(da, m) - at(db, m)) < 1e-9 for m in bounds)

    def const_ratio(self, a, sc):
        """Speed_a / speed_sc if constant over the whole chart, else None."""
        da, ds = self.tl[a], self.tl[sc]
        bounds = sorted(set(da['m']) | set(ds['m']))
        r = None
        def at(d, m):
            i = max(0, _bisect.bisect_right(d['m'], m + 1e-9) - 1)
            return d['v'][i]
        for m in bounds:
            va, vs = at(da, m), at(ds, m)
            if abs(va) < 1e-9 and abs(vs) < 1e-9: continue
            if abs(vs) < 1e-9: return None
            rr = va / vs
            if r is None: r = rr
            elif abs(rr - r) > 1e-6: return None
        return r

    def nsc_pairs(self, tid, ichi):
        """Exact advanced-NSC 'A:B;...' string for a note at measure `ichi` on
        timeline `tid`, or None if it can't be expressed. Also returns pop_in:
        True if the note should already be visible before the 10s window."""
        T = m2t(ichi, self.bpms, self.off)
        t_zero = m2t(0.0, self.bpms, self.off)
        w0 = max(T - NSC_WINDOW_S, t_zero + 1e-4)   # keyframes can't precede measure 0
        if w0 >= T: w0 = max(t_zero + 1e-4, T - 1e-3)
        times = [w0] + self.boundaries(tid, w0, T) + [T]
        # the game's BPMCurve clamps visual positions at measure 0, capping the
        # representable distance. Where D(t) crosses that cap INSIDE a segment,
        # a linear pair-to-pair interpolation would smear the kink across the
        # screen - so insert an extra pair exactly at each crossing.
        cap = T - t_zero
        Ds = [self.remaining(tid, t, T) for t in times]
        times2 = []
        for i in range(len(times)):
            times2.append(times[i])
            if i + 1 < len(times):
                da, db = Ds[i] - cap, Ds[i + 1] - cap
                if da * db < 0:
                    f = da / (da - db)
                    times2.append(times[i] + f * (times[i + 1] - times[i]))
        pts = []
        for t in times2:
            if pts and t - pts[-1][0] < 1e-4: continue
            D = self.remaining(tid, t, T)
            tgt = max(T - D, t_zero)                    # game clamps <measure 0
            A = ichi - t2m(t, self.bpms, self.off)
            B = ichi - t2m(tgt, self.bpms, self.off)
            if not (math.isfinite(A) and math.isfinite(B)): return None, False
            pts.append((t, A, B, D))
        pop_in = pts[0][3] < 2.0                    # <2s of scroll left at window edge
        # simplify collinear (t, B-as-time) points; A stays exact per point
        keep = self._simplify(pts)
        last = keep[-1]
        keep[-1] = (last[0], 0.0, 0.0, 0.0)         # exact 0:0 at the hit moment
        def f(v):
            out = f"{round(v, 5):.5f}".rstrip('0').rstrip('.')
            return out if out not in ('', '-0') else '0'
        return ';'.join(f"{f(a)}:{f(b)}" for _, a, b, _ in keep), pop_in

    def _simplify(self, pts):
        if len(pts) <= NSC_MAX_PAIRS: return list(pts)
        tol = 0.002
        cur = list(pts)
        while len(cur) > NSC_MAX_PAIRS:
            nxt, i = [cur[0]], 1
            while i < len(cur) - 1:
                t0, _, _, d0 = nxt[-1]; t1, _, _, d1 = cur[i]; t2_, _, _, d2 = cur[i + 1]
                interp = d0 + (d2 - d0) * ((t1 - t0) / (t2_ - t0)) if t2_ > t0 else d1
                if abs(interp - d1) < tol: i += 1     # drop this point
                else: nxt.append(cur[i]); i += 1
                if i == len(cur) - 1: break
            nxt.append(cur[-1]); cur = nxt; tol *= 2.0
            if tol > 1.0: break
        return cur

def choose_sc_timeline(chart, fx):
    """Pick the timeline that becomes DR3's global #SC lane. Priorities:
    (1) a timeline whose standalone notes NEED SC (they'd pop in mid-screen
        because their motion keeps them visible >10s before the hit, which
        advanced NSC cannot represent);
    (2) most LN content (LN bodies can only follow SC - the game resets nsc
        on every LN link);
    (3) most notes; (4) @MAINTIL; (5) lowest id."""
    stats = {}
    for g in chart.groups:
        tid = g.get('til', 0)
        st = stats.setdefault(tid, {'ln': 0, 'n': 0, 'need': 0})
        st['n'] += 1
        tc = g['body'][0] if g['body'] else '?'
        if tc in ('h', 'H', 's', 'S'):
            st['ln'] += len(g['children']) + 2
        else:
            T = m2t(chart.ichi(g['tick']), fx.bpms, fx.off)
            if fx.remaining(tid, T - NSC_WINDOW_S, T) < 2.0 and \
               not fx.equal(tid, chart.maintil if chart.maintil in fx.tl else 0):
                st['need'] += 1
    best = max(stats.keys(), key=lambda t: (
        stats[t]['need'] > 0, stats[t]['ln'], stats[t]['n'],
        t == chart.maintil, -t))
    return best, stats

class Converter:
    def __init__(self, chart, ln_density, head_tap, flick_tap, offset):
        self.c = chart
        self.density = ln_density
        self.head_tap = head_tap
        self.flick_tap = flick_tap
        self.offset = offset
        self.notes = []           # master list (parent = array index, -1 = none)
        self.warnings = []
        self.last_note_time = 0.0
        self.ln_densified = self.ln_preserved = 0
        self.bugged_removed = 0
        self.flicks_deduped = 0
        self.bombs_deduped = 0
        self.nps_lns_removed = 0
        self.nps_lns_coarsened = 0
        self.crush_lines_dropped = 0
        self.towers_built = 0
        self.tower_layers = 0
        self.towers_capped = 0
        self.towers_grounded = 0
        self.towers_fattened = 0
        self.mover_added = False
        self._cur_til = 0
        self.scs = None           # [(speed, measure)] for #SC/#SCI, None = default
        self.nsc_simple = self.nsc_adv = 0
        self.sc_til = None

    def _detect_towers(self, plan):
        """Identify AIR-CRUSH towers from the crush plan. Three encodings all
        produce a vertical stack of coincident slabs:

        (A) several separate \'C\' groups sharing (tick, lane, width) at different
            heights - the classic tower (each contributes one slab);
        (B) one \'C\' group whose plan is a short, static interval-driven ladder;
        (C) one \'C\' group whose height changes across a near-instant ribbon -
            its joint points are the stacked slabs.

        Returns {group_index: (tower_key, [slab_heights])}, one height per slab
        the group contributes, so the stack can be ordered bottom-up. Groups not
        in any tower are absent."""
        from collections import defaultdict
        shared = defaultdict(list)                        # (tick,x,w) -> [(gi, height)]
        out = {}
        for gi, pl in plan.items():
            g = self.c.groups[gi]
            _tc, lane, width, extra = parse_body(g['body'])
            if pl['tower']:                              # (B)/(C): stack within one group
                out[gi] = (('stack', gi), [sl[3] for sl in pl['slabs']])
                continue
            if len(pl['slabs']) == 1:                    # candidate for a shared stack (A)
                shared[(g['tick'], lane, width)].append((gi, pl['slabs'][0][3]))
        for key, members in shared.items():
            heights = sorted({h for _gi, h in members})
            if len(heights) < 2:                         # need real vertical spread
                continue
            for gi, h in members:
                out[gi] = (('shared',) + key, [h])
        return out

    def _add(self, t, ichi, x, w, parent=-1, ex_head=False):
        i = len(self.notes)
        self.notes.append({'idx': i, 'file_idx': i, 'type': t,
            'beat': round(ichi, 5), 'x': round(float(x), 5),
            'width': round(clamp_width(float(w)), 5),
            'nsc': '0', 'attr': '', 'parent': parent, '_ex_head': ex_head,
            '_til': self._cur_til})
        return i

    def _path(self, group):
        """Build [(ichi, x, width), ...] for an LN group, inheriting position
        for bare s/c continuation points."""
        _, lane, width, _ = parse_body(group['body'])
        if lane is None: lane, width = 0, 1
        pts = [(self.c.ichi(group['tick']), float(lane), float(max(width, 1)))]
        px, pw = float(lane), float(max(width, 1))
        for ctick, cbody in group['children']:
            _, cl, cw, _ = parse_body(cbody)
            if cl is None: cl, cw = px, pw           # bare s/c -> inherit
            cw = max(cw, 1)
            pts.append((self.c.ichi(ctick), float(cl), float(cw)))
            px, pw = float(cl), float(cw)
        return pts

    def _add_ln(self, group, kind, ex_head=False):
        head_t, mid_t, end_t = kind
        pts = self._path(group)
        if len(pts) < 2:                              # degenerate -> single tap
            ichi, x, w = pts[0]
            self._add(1, ichi, x, w)
            return
        hi = self._add(head_t, *pts[0], ex_head=ex_head)
        prev = hi
        for p in pts[1:-1]:
            prev = self._add(mid_t, *p, parent=prev)
        self._add(end_t, *pts[-1], parent=prev)

    def convert(self):
        crush_plan = air_crush_plan(self.c)
        tower_of = self._detect_towers(crush_plan)    # group-index -> (tower_key, [heights])
        n_crush = sum(1 for g in self.c.groups if g['body'] and g['body'][0] == 'C')
        self.crush_lines_dropped = n_crush - len(crush_plan)
        for gi, g in enumerate(self.c.groups):
            tc, lane, width, extra = parse_body(g['body'])
            ichi = self.c.ichi(g['tick'])
            self._cur_til = g.get('til', 0)
            if tc == 't':                              # Tap
                self._add(1, ichi, lane or 0, width or 1)
            elif tc == 'x':                            # ExTap -> always-perfect tap
                self._add(2, ichi, lane or 0, width or 1)
            elif tc == 'f':                            # Flick (any direction)
                self._add(9, ichi, lane or 0, width or 1)
            elif tc == 'd':                            # Damage / mine -> red avoid
                self._add(10, ichi, lane or 0, width or 1)
            elif tc == 'a':                            # Air -> directional flick
                self._add(air_to_flick(extra), ichi, lane or 0, width or 1)
            elif tc == 'C':                            # Air-Crush -> slab flick(s)
                pl = crush_plan.get(gi)
                if pl is None:
                    pass                                   # decorative line: dropped
                else:
                    tw = tower_of.get(gi)
                    for si, (stick, sx, sw, _sh) in enumerate(pl['slabs']):
                        if tw is not None:
                            # tower slabs stack at the head instant (coincident)
                            ni = self._add(9, ichi, sx, sw)
                            self.notes[ni]['_tower'] = (tw[0], tw[1][si])
                        else:
                            # ordinary ladder: each slab at its own time/lane
                            ni = self._add(9, self.c.ichi(stick), sx, sw)
            elif tc in ('h', 'H'):                     # Hold / Air-Hold -> hold LN
                self._add_ln(g, LN_KINDS[tc])
            elif tc in ('s', 'S'):                     # Slide / Air-Slide -> slide LN
                self._add_ln(g, LN_KINDS[tc])
            # unknown tokens are silently skipped

        # 0) drop type-9 flicks that perfectly overlap a directional flick (13-16);
        #    the type-9 is redundant in DR3 and only muddies the visuals.
        self.notes, self.flicks_deduped = dedupe_flicks(self.notes)

        # 1) centre notes on LNs, reusing the editor code (shape-preserving).
        #    Only densify LNs where the uniform grid would ADD centres; LNs whose
        #    UMIGURI centres are already denser are left untouched so their shape
        #    isn't blockified.
        density_meas = self.density
        heads, preserved = select_densify_heads(self.notes, density_meas)
        self.ln_densified, self.ln_preserved = len(heads), preserved
        if heads:
            ok, self.notes, _ = cmd_addmiddle(self.notes, 0, 0, density_meas, indices=heads)

        # 2) overlay a strict tap on every LN head (UMIGURI hold/slide starts are
        #    judged like taps). Skip heads that already carry a coincident ExTap
        #    (the "fake-ex" pattern), which already makes them justice-critical.
        if self.head_tap:
            ch = build_ln_children(self.notes)
            heads = [p for p in ch.keys()
                     if p not in {c for lst in ch.values() for c in lst}]
            ex2 = {(round(n['beat'], 3), round(n['x'], 3), round(n['width'], 3))
                   for n in self.notes if n['type'] == 2}
            heads = [h for h in heads
                     if (round(self.notes[h]['beat'], 3), round(self.notes[h]['x'], 3),
                         round(self.notes[h]['width'], 3)) not in ex2]
            if heads:
                ok, self.notes, _ = cmd_forceln(self.notes, 0, 0, indices=heads, tap_type=1)

        # 3) optional strict-tap overlay on flicks (default OFF: UMIGURI flicks
        #    are movement-based and chained into flick-slides).
        if self.flick_tap:
            flicks = [i for i, n in enumerate(self.notes) if n['type'] in (9, 13, 14, 15, 16)]
            if flicks:
                ok, self.notes, _ = cmd_forceln(self.notes, 0, 0, indices=flicks, tap_type=1)

        # 3b) delete red bomb notes (type 10) stacked exactly on a non-bomb note -
        #     that combination is impossible to hit. Runs after the tap overlays so
        #     bombs sitting on LN heads (now carrying a tap) are caught too.
        bpms = [{'beat': self.c.ichi(t), 'bpm': v} for t, v in self.c.bpms]
        self.notes, self.bombs_deduped = dedupe_bombs(self.notes, bpms, self.offset)

        # 3c) tame impossibly dense stacked-LN passages (halve perfectly-redundant LNs,
        #     then coarsen survivors to 1/2 centres if still over the NPS limit).
        self.notes, self.nps_lns_removed, self.nps_lns_coarsened = reduce_stacked_nps(
            self.notes, bpms, parse_density('1/2'))

        # 4) delete bugged notes (before audio start / broken chains) so the chart
        #    can't crash DR3. Whole LN chains go if any link is bugged.
        self.notes, self.bugged_removed = delete_bugged(self.notes, bpms, self.offset)

        # 5) scroll-speed gimmicks: translate @TIL/@USETIL/@SPDMOD into SC + NSC
        self._apply_scrollfx(bpms)

        # 6) rebuild AIR-CRUSH towers as DR3 3D stacks (must follow SC: the stack
        #    NSC curves scale by the local SC value). Groups the surviving tower
        #    flicks by their shared key and stacks them by ascending source height.
        self._build_towers()

        # 7) force at least one mover so DR3 never zooms out (UMIGURI never does).
        #    The head of a slide LN (type 5) can become a mover centre (type 23)
        #    with no visual or gameplay change - confirmed safe.
        self._ensure_mover()

        self._validate()
        return self.notes

    def _ensure_mover(self):
        """DR3 only allows the camera to zoom out when a chart has zero mover
        notes; UMIGURI charts never zoom out, so guarantee a mover exists. The
        first slide-LN head (type 5) is retyped to a mover centre (type 23),
        which is visually and mechanically identical here. No-op if the chart
        has no type-5 head."""
        for n in self.notes:
            if n['type'] == 5:
                n['type'] = 23
                self.mover_added = True
                return

    def _build_towers(self):
        """Convert tagged AIR-CRUSH tower members into DR3 3D stacks using the
        editor\'s stacker math. Runs after SC is decided so curves scale to it.

        If a tower sits on top of an ordinary note at the same beat and lane (an
        up-flick, ExTap, tap, ...), that note becomes the stack\'s GROUND layer so
        every slab floats ABOVE it: nothing type-9 is left on the ground beside
        the real note, and no non-type-9 note is ever lifted into the air. The
        slab count is preserved (a 6-height tower shows 6 airborne slabs over the
        ground note). Without such a ground note the lowest slab is the ground,
        as before.

        Air slabs are capped at TOWER_MAX_LAYERS: past ~5 the NSC illusion breaks
        down, so a taller tower keeps that many evenly-spaced slabs (lowest + top)
        and drops the surplus entirely."""
        from collections import defaultdict
        towers = defaultdict(list)
        for n in self.notes:
            tw = n.get('_tower')
            if tw is not None:
                towers[tw[0]].append((tw[1], n))          # (source_height, note)
        # index ordinary (non-tower) notes by beat for ground-note lookup.
        # LN-involved notes are excluded: the game force-clears NSC on LN links,
        # so they can\'t carry the ground layer\'s identity curve.
        ln_involved = set()
        ch_children = build_ln_children(self.notes)
        for parent, kids in ch_children.items():
            ln_involved.add(parent)
            ln_involved.update(kids)
        ground_by_beat = defaultdict(list)
        for gi, n in enumerate(self.notes):
            if '_tower' not in n and gi not in ln_involved:
                ground_by_beat[round(n['beat'], 5)].append(n)
        built = layers_total = capped = grounded = fattened = 0
        drop = set()
        used_ground = set()
        for tkey, members in towers.items():
            if len(members) < 2:
                continue                                  # tower lost members; skip
            members.sort(key=lambda hn: hn[0])            # lowest slab first
            slabs = [n for _h, n in members]
            # look for an ordinary note under this tower to act as the ground so
            # the slabs are erected above it (matched on the tower's lane span,
            # taken from the lowest slab before any stacker shift is applied).
            tx = min(n['x'] for n in slabs)
            tw_ = max(n['x'] + n['width'] for n in slabs) - tx
            g = None
            for cand in ground_by_beat.get(round(slabs[0]['beat'], 5), ()):
                if id(cand) in used_ground:
                    continue
                cl, cr = cand['x'], cand['x'] + cand['width']
                if cl <= tx + 1e-6 and tx + (tw_ or 1) <= cr + 1e-6 or \
                   tx <= cl + 1e-6 and cl + cand['width'] <= tx + (tw_ or 1) + 1e-6:
                    g = cand; break
            # cap TOTAL layers (ground note included) at TOWER_MAX_LAYERS: past
            # ~5 the NSC illusion breaks down. A grounded tower therefore keeps at
            # most TOWER_MAX_LAYERS-1 airborne slabs; a free-standing one keeps
            # TOWER_MAX_LAYERS. Surplus slabs (evenly spaced, lowest+top kept) are
            # dropped from the chart entirely.
            max_slabs = TOWER_MAX_LAYERS - (1 if g is not None else 0)
            if len(slabs) > max_slabs:
                capped += 1
                keep = {round(i * (len(slabs) - 1) / (max_slabs - 1))
                        for i in range(max_slabs)}
                for j, n in enumerate(slabs):
                    if j not in keep:
                        drop.add(id(n))
                slabs = [n for j, n in enumerate(slabs) if j in keep]
            if g is not None:
                used_ground.add(id(g)); grounded += 1
                layers = [g] + slabs                      # real note grounds the stack
            else:
                layers = slabs                            # lowest slab is the ground
            stack_notes_3d(layers, self.scs)
            # The outward lean can push a flick fully off the highway (no part of
            # it in [0, HIGHWAY_WIDTH]). If so, fatten the WHOLE tower uniformly -
            # only the type-9 flicks, never a non-flick ground - by the amount
            # that brings the furthest-out flick 1 unit back in bounds. Fattening
            # keeps each note's centre fixed, so the stack's shape is preserved.
            if self._fatten_tower(layers):
                fattened += 1
            built += 1; layers_total += len(slabs)
        if drop:
            remap, kept = {}, []
            for old_i, n in enumerate(self.notes):
                if id(n) in drop:
                    continue
                remap[old_i] = len(kept)
                kept.append(n)
            self.notes = kept
            for k, n in enumerate(self.notes):
                n['parent'] = remap.get(n['parent'], -1) if n['parent'] >= 0 else -1
                n['idx'] = k
        self.towers_built = built
        self.tower_layers = layers_total
        self.towers_capped = capped
        self.towers_grounded = grounded
        self.towers_fattened = fattened

    def _fatten_tower(self, layers):
        """If any type-9 flick in this tower is fully off the highway (no part
        within [0, HIGHWAY_WIDTH]), widen every type-9 flick in the tower by one
        shared delta - the smallest that pulls the furthest-out flick 1 unit back
        in bounds. Uses the editor\'s fatten semantics (centre kept fixed). The
        ground note is only widened if it is itself a type-9 flick. Returns True
        if any fattening was applied."""
        flicks = [n for n in layers if n['type'] == 9]
        if not flicks:
            return False
        need = 0.0
        for n in flicks:
            l, r = n['x'], n['x'] + n['width']
            if r <= 0.0:                                   # fully off the left
                need = max(need, 2.0 * (1.0 - r))          # bring right edge to +1
            elif l >= HIGHWAY_WIDTH:                        # fully off the right
                need = max(need, 2.0 * (l - (HIGHWAY_WIDTH - 1.0)))
        if need <= 0.0:
            return False
        for n in flicks:
            cx = n['x'] + n['width'] / 2.0
            nw = round(clamp_width(n['width'] + need), 5)
            n['width'] = nw
            n['x'] = round(cx - nw / 2.0, 5)
        return True
        self.towers_built = built
        self.tower_layers = layers_total
        self.towers_capped = capped

    def _apply_scrollfx(self, bpms):
        c = self.c
        if not c.tils and not c.spdmod:
            return                                        # no gimmicks: keep default header
        if c.has_spdfld:
            self.warnings.append("chart uses @SPDDEF/@SPDFLD (v2.01 spatial speed) - unsupported, ignored")
        fx = ScrollFX(c, bpms, self.offset)
        sc_til, _ = choose_sc_timeline(c, fx)
        self.sc_til = sc_til
        # ---- emit the SC lane: effective curve of the chosen timeline ----
        d = fx.tl[sc_til]
        scs = [(d['v'][max(0, _bisect.bisect_right(d['m'], 1e-9) - 1)], 0.0)]
        for m, v in zip(d['m'], d['v']):
            if m > 1e-9 and abs(v - scs[-1][0]) > 1e-9:
                scs.append((v, m))
        if len(scs) > 1 or abs(scs[0][0] - 1.0) > 1e-9:
            self.scs = scs
        # ---- per-note NSC for standalone notes on other timelines ----
        ch_map = build_ln_children(self.notes)
        involved = set(ch_map.keys()) | {x for lst in ch_map.values() for x in lst}
        ln_off, pop_ins = 0, 0
        seen_ln_tils = set()
        for i, n in enumerate(self.notes):
            tid = n.get('_til')
            if tid is None or tid not in fx.tl or tid == sc_til: continue
            if fx.equal(tid, sc_til): continue
            if i in involved:                             # LN link/head: engine forces SC
                if is_ln_child(self.notes, i): continue   # count chains once, via heads
                ln_off += 1; seen_ln_tils.add(tid); continue
            r = fx.const_ratio(tid, sc_til)
            if r is not None and r > 1e-9:
                if abs(r - 1.0) > 1e-9:
                    n['nsc'] = f"{round(r, 5):.5f}".rstrip('0').rstrip('.')
                    self.nsc_simple += 1
                continue
            pairs, pop = fx.nsc_pairs(tid, n['beat'])
            if pairs:
                n['nsc'] = pairs; self.nsc_adv += 1
                if pop: pop_ins += 1
        if ln_off:
            self.warnings.append(
                f"{ln_off} LNs ride timeline(s) {sorted(seen_ln_tils)} but DR3 LNs can only "
                f"follow the global SC lane (timeline {sc_til}) - their scroll motion is approximated")
        if pop_ins:
            self.warnings.append(
                f"{pop_ins} notes should be visible more than 10s before their hit; DR3 spawns "
                f"advanced-NSC notes only inside 10s, so they will pop in")
        mixed = sum(1 for g in c.groups if g.get('til_mixed'))
        if mixed:
            self.warnings.append(f"{mixed} slides switch timelines mid-chain (@USETIL between joints) - "
                                 f"DR3 cannot vary speed within one LN; using the head's timeline")

    def _validate(self):
        bpms = [{'beat': self.c.ichi(t), 'bpm': v} for t, v in self.c.bpms]
        self.last_note_time = max((m2t(n['beat'], bpms, self.offset) for n in self.notes),
                                  default=0.0)
        if self.c.bpm0_dropped:
            self.warnings.append(
                f"removed {self.c.bpm0_dropped} duplicate BPM event(s) at measure 0 "
                f"(kept {self.c.bpm0_kept:g})"
                + (" - they disagreed, so the fastest was kept" if self.c.bpm0_differed else "")
                + "; DR3 freezes every note if measure 0 has more than one BPM")
        if self.bugged_removed:
            self.warnings.append(f"deleted {self.bugged_removed} bugged notes "
                                 f"(before audio start / broken chains) to prevent a DR3 crash")
        # safety: should be clean now
        left = len(find_bugged(self.notes, bpms, self.offset, has_audio=True))
        if left:
            self.warnings.append(f"{left} bugged notes remain after cleanup (please report)")

# ============================================================================
#  PART 4: header / data / asset output
# ============================================================================
def build_header(chart, offset, scs=None):
    bpms = [(chart.ichi(t), v) for t, v in chart.bpms]
    def fnum(v):
        return f"{v:.5f}".rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
    h = [f"#OFFSET={fnum(round(offset, 5))};", "#BEAT=1;",
         f"#BPM_NUMBER={len(bpms)};"]
    for i, (pos, val) in enumerate(bpms):
        h.append(f"#BPM [{i}]={fnum(float(val))};")
        h.append(f"#BPMS[{i}]={fnum(float(pos))};")
    if scs:                                    # translated @TIL/@SPDMOD scroll lane
        h.append(f"#SCN={len(scs)};")
        for i, (val, pos) in enumerate(scs):
            h.append(f"#SC [{i}]={fnum(round(float(val), 5))};")
            h.append(f"#SCI[{i}]={fnum(round(float(pos), 5))};")
    else:
        h += ["#SCN=1;", "#SC [0]=1.0;", "#SCI[0]=0.0;"]
    return h


def meta1(chart, tag, default=''):
    v = chart.meta.get(tag)
    return (v[0] if v else default)

def write_data_txt(chart, path):
    """title / artist always present; bpm forced to a positive non-zero integer;
    no preview field; copyright = 'Chart by <DESIGN>' only if a charter is named."""
    title  = (meta1(chart, 'TITLE')  or '').strip() or 'Unknown'
    artist = (meta1(chart, 'ARTIST') or '').strip() or 'Unknown'
    bpm = chart.bpms[0][1] if chart.bpms else 0
    bpm_int = int(round(bpm)) if bpm and bpm > 0 else 0
    if bpm_int < 1: bpm_int = 1                      # must be valid non-zero positive int
    lines = [f"title:{title}", f"artist:{artist}", f"bpm:{bpm_int}"]
    designer = (meta1(chart, 'DESIGN') or '').strip()
    if designer:                                     # omit entirely if no charter name
        lines.append(f"copyright:Chart by {designer}")
    open(path, 'w', encoding='utf-8', newline='').write('\r\n'.join(lines) + '\r\n')

def write_data_txt_multi(items, path):
    """One shared data.txt for a multi-difficulty song. Metadata comes from the
    hardest difficulty; copyright credits every distinct charter (hardest first)."""
    hardest = max(items, key=lambda x: x['const'])
    ch = hardest['chart']
    title  = (meta1(ch, 'TITLE')  or '').strip() or 'Unknown'
    artist = (meta1(ch, 'ARTIST') or '').strip() or 'Unknown'
    bpm = ch.bpms[0][1] if ch.bpms else 0
    bpm_int = int(round(bpm)) if bpm and bpm > 0 else 0
    if bpm_int < 1: bpm_int = 1
    lines = [f"title:{title}", f"artist:{artist}", f"bpm:{bpm_int}"]
    designers = []
    for it in sorted(items, key=lambda x: -x['const']):   # hardest first
        d = (meta1(it['chart'], 'DESIGN') or '').strip()
        if d and d not in designers:
            designers.append(d)
    if designers:
        lines.append("copyright:Chart by " + " / ".join(designers))
    open(path, 'w', encoding='utf-8', newline='').write('\r\n'.join(lines) + '\r\n')

AUDIO_EXTS = ('.mp3', '.ogg', '.wav', '.flac', '.m4a', '.aac', '.opus', '.wma', '.aiff', '.aif')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')

def _norm_name(s):
    """Fold case and treat spaces/underscores/hyphens as interchangeable, for
    matching filenames that drifted from what @BGM/@JACKET recorded."""
    return re.sub(r'[\s_\-]+', '', s.lower())

def find_asset(src_dir, name, exts):
    """Resolve a file referenced by @BGM/@JACKET, tolerating filename drift
    (space<->underscore, case, a changed extension) and finally falling back to the
    sole file of that kind in the folder. Returns an absolute path, or '' if nothing
    sensible matches. Chart zips almost always hold exactly one audio + one image,
    so the fallback is safe."""
    if not os.path.isdir(src_dir):
        return ''
    try:
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    except OSError:
        return ''
    base = os.path.basename(name) if name else ''
    if base:
        if base in files:                                   # 1) exact
            return os.path.join(src_dir, base)
        low = base.lower()                                  # 2) case-insensitive
        for f in files:
            if f.lower() == low:
                return os.path.join(src_dir, f)
        nb = _norm_name(base)                               # 3) space/underscore/hyphen + case
        for f in files:
            if _norm_name(f) == nb:
                return os.path.join(src_dir, f)
        nstem = _norm_name(os.path.splitext(base)[0])       # 4) same stem, different extension
        cand = [f for f in files if f.lower().endswith(exts)
                and _norm_name(os.path.splitext(f)[0]) == nstem]
        if len(cand) == 1:
            return os.path.join(src_dir, cand[0])
    kind = [f for f in files if f.lower().endswith(exts)]   # 5) the only file of this kind
    if len(kind) == 1:
        return os.path.join(src_dir, kind[0])
    return ''

def audio_duration(path):
    if not shutil.which('ffprobe'):
        return 0.0
    try:
        out = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                              '-of', 'csv=p=0', path], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0

def has_video_stream(path):
    """True if the file carries a video stream (e.g. embedded album art). Such a
    stream becomes a Theora video in the ogg and stops DR3 from playing it."""
    if not shutil.which('ffprobe'):
        return False
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v',
                              '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', path],
                             capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()
        return 'video' in out
    except Exception:
        return False

def analyze_audio(path):
    """Return (sample_rate, channels, peak_dbfs) for the first audio stream, using
    ffprobe + ffmpeg's volumedetect (peak in dBFS, like dr3editor.py's analysis)."""
    sr = ch = 0
    if shutil.which('ffprobe'):
        try:
            out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                                  '-show_entries', 'stream=sample_rate,channels',
                                  '-of', 'csv=p=0', path], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout.strip()
            p = out.split(',')
            sr = int(p[0]); ch = int(p[1])
        except Exception:
            pass
    peak_db = 0.0
    if shutil.which('ffmpeg'):
        try:
            r = subprocess.run(['ffmpeg', '-i', path, '-vn', '-af', 'volumedetect', '-f', 'null', '-'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
            m = re.search(r'max_volume:\s*(-?[\d.]+) dB', r.stderr)
            if m: peak_db = float(m.group(1))
        except Exception:
            pass
    return sr, ch, peak_db

# DR3-expected audio (verbatim from dr3editor.py "Fix audio problems")
REC_SR = 44100
REC_CH = 2
TARGET_PEAK_DB = 0.0
NORM_THRESHOLD_DB = -0.5

def prepare_audio(src, dst, need_seconds, tail=2.0):
    """Write `src` to `dst` as a DR3-ready .ogg in ONE ffmpeg pass:
      * -vn                 strip embedded album art (else Theora video breaks DR3)
      * volume=(0-peak)dB   peak-normalize to 0 dBFS when peak < -0.5 dB
                            (flat gain - dynamics untouched, exactly like the editor)
      * -ar 44100           resample if the source isn't 44.1 kHz
      * -ac 2               upmix to stereo if not already
      * apad                pad trailing silence if the chart outlasts the music
    A clean .ogg that needs none of these is copied losslessly."""
    if not shutil.which('ffmpeg'):
        return False, "ffmpeg not found in PATH (audio not written)"
    dur = audio_duration(src)
    target = need_seconds + tail
    pad = dur > 0 and target > dur + 0.05
    sr, ch, peak_db = analyze_audio(src)
    sr_bad   = bool(sr) and sr != REC_SR
    ch_bad   = bool(ch) and ch != REC_CH
    peak_bad = peak_db < NORM_THRESHOLD_DB

    filters, out_opts, fixes = [], [], []
    if peak_bad:
        gain = round(TARGET_PEAK_DB - peak_db, 2)        # = -peak_db ; flat gain
        filters.append(f'volume={gain}dB'); fixes.append(f"normalized {peak_db:+.1f}->0 dB")
    if sr_bad:
        out_opts += ['-ar', str(REC_SR)]; fixes.append(f"{sr}->{REC_SR} Hz")
    if ch_bad:
        out_opts += ['-ac', str(REC_CH)]
        fixes.append(("mono" if ch == 1 else f"{ch}ch") + "->stereo")
    if pad:
        filters.append(f'apad=whole_dur={target:.3f}'); fixes.append(f"padded {dur:.1f}->{target:.1f}s")

    # lossless fast path: a clean ogg with nothing to fix and no video
    if not fixes and src.lower().endswith('.ogg') and not has_video_stream(src):
        shutil.copyfile(src, dst)
        return True, "audio -> ogg (copied; already DR3-ready)"

    cmd = ['ffmpeg', '-y', '-i', src, '-vn']
    if filters: cmd += ['-af', ','.join(filters)]
    cmd += out_opts + ['-c:a', 'libvorbis', '-q:a', '6', dst]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return False, "ffmpeg failed: " + e.stderr.decode('utf-8', 'replace')[-200:]
    return True, "audio -> ogg (-vn" + ("; " + "; ".join(fixes) if fixes else "") + ")"

def convert_jacket(src, dst):
    if ensure_package('PIL', 'Pillow') is None:
        return False, "Pillow not installed and auto-install failed (jacket not written)"
    from PIL import Image
    im = Image.open(src).convert('RGB')
    w, h = im.size
    s = min(w, h)                                    # centre-crop to a square
    left, top = (w - s) // 2, (h - s) // 2
    im.crop((left, top, left + s, top + s)).save(dst, 'PNG')
    return True, f"jacket -> png ({s}x{s})"

def sanitize(name):
    name = re.sub(r'[^A-Za-z0-9_-]', '', (name or '').replace(' ', ''))
    return name or ''

# ============================================================================
#  PART 5: CLI  (input .zip  ->  output DR_<name>.zip)
# ============================================================================
def parse_density(s):
    s = s.strip()
    if '/' in s:
        num, den = s.split('/'); beats = float(num) / float(den)
    else:
        beats = float(s)
    return beats / 4.0                               # editor convention: beats -> measures

def find_all_in_zip(root, exts):
    """Return all files under `root` whose extension is in `exts` (walks subfolders)."""
    found = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() in exts:
                found.append(os.path.join(dirpath, fn))
    return found

def select_single_bgm(items):
    """A multi-.ugc zip normally means one song with several difficulties sharing one
    audio file. Some packs instead ship speed/BPM variants of the same chart, each with
    its OWN audio (e.g. 240.ugc+240.ogg, 260.ugc+260.ogg ...). Merging those would pair
    every chart with a single audio file, so all but one would be badly desynced.

    The test is which audio file each difficulty actually RESOLVES to, not what @BGM
    says: if the zip only holds one audio file, every chart uses it (names may simply
    have drifted) and nothing is skipped. Only when the difficulties genuinely resolve
    to DIFFERENT audio files do we keep the highest-tier chart (by @LEVEL, then @CONST)
    and drop the rest. Returns (items, note_or_None)."""
    if len(items) < 2:
        return items, None
    resolved = {}
    for it in items:
        p = find_asset(it['src_dir'], meta1(it['chart'], 'BGM'), AUDIO_EXTS)
        resolved[id(it)] = os.path.normcase(os.path.abspath(p)) if p else ''
    distinct = {v for v in resolved.values() if v}
    if len(distinct) < 2:
        return items, None                       # one shared audio -> keep every difficulty
    keep = max(items, key=lambda x: (x['level_int'], x['const']))
    dropped = [it for it in items if it is not keep]
    names = ', '.join(sorted(os.path.basename(it['path']) for it in dropped))
    note = (f"difficulties use different audio files ({len(distinct)} of them). "
            f"Converting only the highest-tier chart ({os.path.basename(keep['path'])}, "
            f"Lv {meta1(keep['chart'], 'LEVEL', '?')}); ignoring {names}")
    return [keep], note

DR3_MIN_TIER = 1        # DR3 has no tier 0
DR3_MAX_TIER = 25       # ...and nothing above 25
FALLBACK_TIER = 15      # unrated charts (@LEVEL 0 / missing) land here

def dr3_tier(level_int):
    """Map a UMIGURI @LEVEL onto a tier DR3 actually has."""
    if level_int < DR3_MIN_TIER:        # 0 (or junk) = unrated -> default to 15
        return FALLBACK_TIER
    return min(level_int, DR3_MAX_TIER)  # nothing above 25

def worldsend_tier(level_int):
    """WORLD'S END charts (@DIFF 4) carry no real level - @LEVEL holds the star
    rating, encoded as odd numbers 1,3,5,7,9 = 1..5 stars. Calibrate stars to tiers:
    1->12, 2->13, 3->14, 4->15, 5->16. Anything outside 1..5 stars is treated as a
    literal level (e.g. a hand-set LEVEL 11 -> tier 11)."""
    star = (level_int + 1) // 2
    if 1 <= star <= 5:
        return 11 + star
    return dr3_tier(level_int)

def _diff_stage(ch):
    d = meta1(ch, 'DIFF', '').strip()
    return int(d) if d.isdigit() else 3      # no @DIFF -> treat like a normal (MASTER-rank) chart

def assign_levels(items):
    """Give each chart a UNIQUE integer DR3 tier in 1..25, ordered by difficulty stage
    (@DIFF: BASIC<ADVANCED<EXPERT<MASTER<WORLD'S END<ULTIMA). Normal stages and ULTIMA
    use their real @LEVEL; WORLD'S END uses the star calibration. Tiers are forced
    strictly increasing along that order, so MASTER < WORLD'S END < ULTIMA always holds
    (WE ends up above MASTER, ULTIMA above WE) even when a raw level wouldn't. Mutates
    items, adding 'dr3_level'."""
    def desired(it):
        if _diff_stage(it['chart']) == 4:            # WORLD'S END
            return worldsend_tier(it['level_int'])
        return dr3_tier(it['level_int'])             # normal stages + ULTIMA use real level
    order = sorted(items, key=lambda it: (_diff_stage(it['chart']), it['const'], it['level_int']))
    tiers, cur = [], 0
    for it in order:                                  # strictly increasing by difficulty
        t = max(desired(it), cur + 1)
        tiers.append(t); cur = t
    cap = DR3_MAX_TIER                                # pull back under 25 from the hardest end
    for i in range(len(order) - 1, -1, -1):
        tiers[i] = min(tiers[i], cap); cap = tiers[i] - 1
    floor = DR3_MIN_TIER                              # and keep >=1, preserving the order
    for i in range(len(order)):
        tiers[i] = max(tiers[i], floor); floor = tiers[i] + 1
    for it, t in zip(order, tiers):
        it['dr3_level'] = t

def diff_label(ch):
    """Short human label for a chart's difficulty stage + rating, e.g. 'MASTER 13+'
    or 'WORLD\'S END 2★', for the conversion log."""
    stage = _diff_stage(ch)
    names = {0: 'BASIC', 1: 'ADVANCED', 2: 'EXPERT', 3: 'MASTER', 4: "WORLD'S END", 5: 'ULTIMA'}
    lvl = meta1(ch, 'LEVEL', '?')
    if stage == 4:                                   # show WORLD'S END rating as stars
        try:
            star = (int(re.sub(r'\D', '', lvl) or '0') + 1) // 2
            lvl = f"{star}★" if 1 <= star <= 5 else lvl
        except ValueError:
            pass
    nm = names.get(stage)
    return f"{nm} {lvl}" if nm and meta1(ch, 'DIFF', '').strip().isdigit() else f"Lv {lvl}"

def convert_one(zip_path, args):
    """Convert one input zip (which may contain SEVERAL .ugc difficulties of the
    same song) into a single DR_<name>.zip with one chart file per difficulty,
    sharing one ogg / png / data.txt. Returns (out_zip, summary_lines, warnings)."""
    import tempfile, zipfile
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"'{zip_path}' is not a valid .zip file")
    summary, msgs, all_warnings = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        ugc_paths = find_all_in_zip(tmp, {'.ugc'})
        if not ugc_paths:
            raise ValueError("no .ugc chart found anywhere in the zip")

        # ---- parse every difficulty ----
        items = []
        for p in ugc_paths:
            ch = parse_ugc(p)
            lvl_int = int(re.sub(r'\D', '', meta1(ch, 'LEVEL', '')) or '1')
            try: const = float(meta1(ch, 'CONST', '') or lvl_int)
            except ValueError: const = float(lvl_int)
            items.append({'chart': ch, 'path': p, 'src_dir': os.path.dirname(p),
                          'level_int': lvl_int, 'const': const})

        # speed-variant packs (each difficulty has its own audio): keep the top tier only
        items, bgm_note = select_single_bgm(items)
        if bgm_note:
            msgs.append("NOTE: " + bgm_note)

        # shared base name (all difficulties share one stem so DR3 groups them)
        title = meta1(items[0]['chart'], 'TITLE')
        base = (sanitize(args.name) or sanitize(title)
                or sanitize(os.path.splitext(os.path.basename(zip_path))[0]) or 'song')

        # unique DR3 tiers (only honour --level for a single-difficulty zip)
        if args.level and len(items) == 1:
            items[0]['dr3_level'] = dr3_tier(int(re.sub(r'\D', '', args.level) or '0'))
        else:
            if args.level and len(items) > 1:
                msgs.append("note: --level ignored (zip has multiple difficulties)")
            assign_levels(items)

        # ---- convert each difficulty's notes ----
        stage = os.path.join(tmp, '_dr3out'); os.makedirs(stage, exist_ok=True)
        density = parse_density(args.ln_density)
        max_need = 0.0
        for it in items:
            ch = it['chart']
            offset = ch.bgmofs * (1.0 if args.offset_sign == '+' else -1.0)
            conv = Converter(ch, density, head_tap=not args.no_head_tap,
                             flick_tap=args.flick_tap, offset=offset)
            notes = conv.convert()
            it['offset'] = offset
            max_need = max(max_need, conv.last_note_time)
            fname = f"{base}.{it['dr3_level']}.txt"
            open(os.path.join(stage, fname), 'w', encoding='utf-8', newline='').write(
                format_chart(build_header(ch, offset, conv.scs), notes))
            summary.append(f"[{diff_label(ch)} -> tier {it['dr3_level']}] {fname}: "
                           f"{len(notes)} notes, offset {offset:+.3f}s, "
                           f"LNs {conv.ln_densified} densified / {conv.ln_preserved} kept"
                           + (f", {conv.flicks_deduped} dup flicks removed" if conv.flicks_deduped else "")
                           + (f", {conv.bombs_deduped} dup bombs removed" if conv.bombs_deduped else "")
                   + (f", {conv.crush_lines_dropped} air-crush lines skipped"
                      if conv.crush_lines_dropped else "")
                           + (f", NPS: {conv.nps_lns_removed} stacked LNs cut" if conv.nps_lns_removed else "")
                           + (f", {conv.nps_lns_coarsened} LNs thinned" if conv.nps_lns_coarsened else "")
                           + (f", SC {len(conv.scs)} kf (til {conv.sc_til})" if conv.scs else "")
                           + (f", NSC {conv.nsc_adv} curves/{conv.nsc_simple} simple"
                              if (conv.nsc_adv or conv.nsc_simple) else ""))
            all_warnings += [f"(tier {it['dr3_level']}) {w}" for w in conv.warnings]

        # ---- shared assets (use the hardest difficulty's metadata) ----
        hardest = max(items, key=lambda x: x['const'])
        ch = hardest['chart']; src_dir = hardest['src_dir']
        write_data_txt_multi(items, os.path.join(stage, f"{base}.data.txt"))

        bgm = meta1(ch, 'BGM'); bgm_src = find_asset(src_dir, bgm, AUDIO_EXTS)
        if bgm_src:
            _, m = prepare_audio(bgm_src, os.path.join(stage, f"{base}.ogg"),
                                 need_seconds=max_need); msgs.append(m)
        else:
            msgs.append(f"audio '{bgm}' not found in the zip (skipped)")
        jak = meta1(ch, 'JACKET'); jak_src = find_asset(src_dir, jak, IMAGE_EXTS)
        if jak_src:
            _, m = convert_jacket(jak_src, os.path.join(stage, f"{base}.png")); msgs.append(m)
        else:
            msgs.append(f"jacket '{jak}' not found in the zip (skipped)")

        zip_stem = os.path.splitext(os.path.basename(zip_path))[0]
        out_dir = args.out or os.path.dirname(os.path.abspath(zip_path))
        os.makedirs(out_dir, exist_ok=True)
        out_zip = os.path.join(out_dir, f"DR_{zip_stem}.zip")
        with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fn in sorted(os.listdir(stage)):
                zf.write(os.path.join(stage, fn), fn)

        summary = [f"{len(items)} difficulty(ies), base '{base}', bpm {ch.bpms[0][1]}"] + summary + msgs
    return out_zip, summary, all_warnings

def web_convert(workdir, base_override='', level_override='', ln_density='1/4',
                head_tap=True, flick_tap=False, offset_sign='-'):
    """Pyodide entry point. `workdir` holds the already-extracted zip contents.
    Does ALL the chart logic (identical to convert_one) and returns a JSON plan.
    Audio + jacket are handled in JS (ffmpeg.wasm / canvas), so this only resolves
    which zip entries to use for them and how long the audio must be."""
    import json
    ugc_paths = find_all_in_zip(workdir, {'.ugc'})
    if not ugc_paths:
        return json.dumps({'error': 'no .ugc chart found anywhere in the zip'})

    items = []
    for p in ugc_paths:
        ch = parse_ugc(p)
        lvl_int = int(re.sub(r'\D', '', meta1(ch, 'LEVEL', '')) or '1')
        try: const = float(meta1(ch, 'CONST', '') or lvl_int)
        except ValueError: const = float(lvl_int)
        items.append({'chart': ch, 'path': p, 'src_dir': os.path.dirname(p),
                      'level_int': lvl_int, 'const': const})

    # speed-variant packs (each difficulty has its own audio): keep the top tier only
    items, bgm_note = select_single_bgm(items)

    title = meta1(items[0]['chart'], 'TITLE')
    base = sanitize(base_override) or sanitize(title) or 'song'
    if level_override and len(items) == 1:
        items[0]['dr3_level'] = dr3_tier(int(re.sub(r'\D', '', level_override) or '0'))
    else:
        assign_levels(items)

    density = parse_density(ln_density)
    stage = os.path.join(workdir, '_dr3out'); os.makedirs(stage, exist_ok=True)
    out_text, log, warnings, max_need = {}, [], [], 0.0
    if bgm_note:
        warnings.append(bgm_note)
    for it in items:
        ch = it['chart']
        offset = ch.bgmofs * (1.0 if offset_sign == '+' else -1.0)
        conv = Converter(ch, density, head_tap=head_tap, flick_tap=flick_tap, offset=offset)
        notes = conv.convert()
        max_need = max(max_need, conv.last_note_time)
        fname = f"{base}.{it['dr3_level']}.txt"
        out_text[fname] = format_chart(build_header(ch, offset, conv.scs), notes)
        log.append(f"[{diff_label(ch)} -> tier {it['dr3_level']}] {fname}: {len(notes)} notes, "
                   f"offset {offset:+.3f}s, LNs {conv.ln_densified} densified / {conv.ln_preserved} kept"
                   + (f", {conv.flicks_deduped} dup flicks removed" if conv.flicks_deduped else "")
                   + (f", {conv.bombs_deduped} dup bombs removed" if conv.bombs_deduped else "")
                   + (f", {conv.crush_lines_dropped} air-crush lines skipped"
                      if conv.crush_lines_dropped else "")
                   + (f", NPS: {conv.nps_lns_removed} stacked LNs cut" if conv.nps_lns_removed else "")
                   + (f", {conv.nps_lns_coarsened} LNs thinned" if conv.nps_lns_coarsened else "")
                   + (f", SC {len(conv.scs)} kf (til {conv.sc_til})" if conv.scs else "")
                   + (f", NSC {conv.nsc_adv} curves/{conv.nsc_simple} simple"
                      if (conv.nsc_adv or conv.nsc_simple) else ""))
        warnings += [f"(tier {it['dr3_level']}) {w}" for w in conv.warnings]

    data_name = f"{base}.data.txt"
    data_path = os.path.join(stage, data_name)
    write_data_txt_multi(items, data_path)
    out_text[data_name] = open(data_path, encoding='utf-8', newline='').read()

    hardest = max(items, key=lambda x: x['const'])
    ch, src_dir = hardest['chart'], hardest['src_dir']
    def rel(name, exts):
        full = find_asset(src_dir, name, exts)
        return os.path.relpath(full, workdir) if full else ''
    return json.dumps({
        'base': base,
        'out_text': out_text,                       # {filename: text} chart files + data.txt
        'ogg_name': f"{base}.ogg",  'audio_rel': rel(meta1(ch, 'BGM'), AUDIO_EXTS),
        'png_name': f"{base}.png",  'jacket_rel': rel(meta1(ch, 'JACKET'), IMAGE_EXTS),
        'need_seconds': max_need,
        'log': log, 'warnings': warnings,
    }, ensure_ascii=True)

def expand_inputs(inputs):
    """Turn file/folder arguments into a flat, de-duplicated list of .zip paths."""
    zips = []
    for p in inputs:
        if os.path.isdir(p):
            zips += sorted(glob.glob(os.path.join(p, '*.zip')))
        elif p.lower().endswith('.zip'):
            zips.append(p)
        else:
            print(f"  skip (not a .zip or folder): {p}")
    seen, out = set(), []
    for z in zips:
        ap = os.path.abspath(z)
        if ap not in seen and not os.path.basename(z).startswith('DR_'):
            seen.add(ap); out.append(z)
    return out

def main():
    ap = argparse.ArgumentParser(description="Convert UMIGURI chart .zip(s) to DanceRail3 .zip(s).")
    ap.add_argument('inputs', nargs='+', help=".zip file(s), and/or folder(s) of zips, to convert")
    ap.add_argument('-o', '--out', default=None,
                    help="directory for the output zips (default: next to each input)")
    ap.add_argument('--name', default=None,
                    help="internal base filename (single-file use only; from title by default)")
    ap.add_argument('--level', default=None, help="DR3 tier number for the chart filename")
    ap.add_argument('--ln-density', default='1/4',
                    help="LN centre spacing as a beat fraction (editor arg, default 1/4)")
    ap.add_argument('--flick-tap', action='store_true',
                    help="overlay a strict type-1 tap on every flick (default off)")
    ap.add_argument('--no-head-tap', action='store_true',
                    help="do NOT overlay strict taps on LN heads")
    ap.add_argument('--offset-sign', choices=['+', '-'], default='-',
                    help="sign applied to @BGMOFS for #OFFSET; default '-' (= -BGMOFS). flip if desynced")
    args = ap.parse_args()

    zips = expand_inputs(args.inputs)
    if not zips:
        sys.exit("error: no .zip inputs found")
    if len(zips) > 1 and args.name:
        print("  note: --name ignored for batches of more than one zip")
        args.name = None

    ok = 0
    for i, zp in enumerate(zips, 1):
        print(f"[{i}/{len(zips)}] {os.path.basename(zp)}")
        try:
            out_zip, summary, warnings = convert_one(zp, args)
            for line in summary: print("    " + line)
            print("    -> " + out_zip)
            for w in warnings: print("    WARNING: " + w)
            ok += 1
        except Exception as e:
            print(f"    ERROR: {e}")
    print(f"\nDone: {ok}/{len(zips)} converted.")


if __name__ == '__main__':
    main()
