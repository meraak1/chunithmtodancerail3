#!/usr/bin/env python3
"""
ugc2dr3.py  --  Convert UMiGuri / Margrete Chunithm chart .zip(s) to DanceRail3 .zip(s).

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
#  PART 1 -- code reused verbatim from dr3editor.py (do not edit; keep in sync)
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
    uniform `density` grid. For an LN whose Chunithm centres are ALREADY denser
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
    drop = {i for i, n in enumerate(notes) if n['type'] == 9 and key(n) in dir_keys}
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
#  PART 2 -- UMiGuri (.ugc) parser
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
        self.groups = []          # list of dicts: {head_tick, body, children:[(abs_tick, body)]}

    # measure'tick -> absolute tick, honouring @BEAT time-signature segments
    def mt_to_tick(self, measure, tick):
        if not self.beats:
            return measure * self.ticks_per_beat * 4 + tick
        # accumulate full-measure lengths up to `measure`
        segs = sorted(self.beats, key=lambda b: b[0])
        abs_meas_tick = 0
        cur = 0
        for k, (sm, num, den) in enumerate(segs):
            nxt = segs[k + 1][0] if k + 1 < len(segs) else measure
            seg_end = min(nxt, measure)
            if seg_end > cur:
                mlen = self.ticks_per_beat * 4 * num // den
                abs_meas_tick += (seg_end - cur) * mlen
                cur = seg_end
            if cur >= measure: break
        return abs_meas_tick + tick

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
            else:
                c.meta[tag] = parts[1:] if len(parts) > 1 else ['']
            continue
        # ----- note section -----
        m = re.match(r"^#(\d+)'(\d+):(.+)$", line)
        if m:
            at = c.mt_to_tick(int(m.group(1)), int(m.group(2)))
            cur_group = {'tick': at, 'body': m.group(3), 'children': []}
            c.groups.append(cur_group)
            continue
        m = re.match(r"^#(\d+)>(.+)$", line)
        if m and cur_group is not None:
            # child reltick is cumulative from the head's absolute tick
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
    return c


def parse_body(body):
    """Return (type_char, lane, width, extra) ; lane/width None for bare s/c."""
    tc = body[0]
    if len(body) >= 3 and body[1] in '0123456789ABCDEFG' and body[2] in '0123456789ABCDEFG':
        return tc, _dec(body[1]), _dec(body[2]), body[3:]
    return tc, None, None, body[1:]

# ============================================================================
#  PART 3 -- mapping Chunithm notes -> DR3 notes
# ============================================================================
LN_KINDS = {'h': (3, 11, 4), 'H': (3, 11, 4),      # holds, air-holds -> orange hold LN
            's': (5, 6, 7),  'S': (5, 6, 7)}        # slides, air-slides -> blue slide LN

def air_to_flick(extra):
    """extra like 'UCN','DLN','URN' -> DR3 flick type."""
    vdir = extra[0] if len(extra) >= 1 else 'U'
    hdir = extra[1] if len(extra) >= 2 else 'C'
    if hdir == 'L': return 13          # Flick L
    if hdir == 'R': return 14          # Flick R
    return 16 if vdir == 'D' else 15   # down / up


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

    def _add(self, t, ichi, x, w, parent=-1, ex_head=False):
        i = len(self.notes)
        self.notes.append({'idx': i, 'file_idx': i, 'type': t,
            'beat': round(ichi, 5), 'x': round(float(x), 5),
            'width': round(clamp_width(float(w)), 5),
            'nsc': '0', 'attr': '', 'parent': parent, '_ex_head': ex_head})
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
        for g in self.c.groups:
            tc, lane, width, extra = parse_body(g['body'])
            ichi = self.c.ichi(g['tick'])
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
            elif tc == 'C':                            # Air-Crush -> flick at start
                self._add(9, ichi, lane or 0, width or 1)
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
        #    Chunithm centres are already denser are left untouched so their shape
        #    isn't blockified.
        density_meas = self.density
        heads, preserved = select_densify_heads(self.notes, density_meas)
        self.ln_densified, self.ln_preserved = len(heads), preserved
        if heads:
            ok, self.notes, _ = cmd_addmiddle(self.notes, 0, 0, density_meas, indices=heads)

        # 2) overlay a strict tap on every LN head (Chunithm hold/slide starts are
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

        # 3) optional strict-tap overlay on flicks (default OFF: Chunithm flicks
        #    are movement-based and chained into flick-slides).
        if self.flick_tap:
            flicks = [i for i, n in enumerate(self.notes) if n['type'] in (9, 13, 14, 15, 16)]
            if flicks:
                ok, self.notes, _ = cmd_forceln(self.notes, 0, 0, indices=flicks, tap_type=1)

        # 4) delete bugged notes (before audio start / broken chains) so the chart
        #    can't crash DR3. Whole LN chains go if any link is bugged.
        bpms = [{'beat': self.c.ichi(t), 'bpm': v} for t, v in self.c.bpms]
        self.notes, self.bugged_removed = delete_bugged(self.notes, bpms, self.offset)

        self._validate()
        return self.notes

    def _validate(self):
        bpms = [{'beat': self.c.ichi(t), 'bpm': v} for t, v in self.c.bpms]
        self.last_note_time = max((m2t(n['beat'], bpms, self.offset) for n in self.notes),
                                  default=0.0)
        if self.bugged_removed:
            self.warnings.append(f"deleted {self.bugged_removed} bugged notes "
                                 f"(before audio start / broken chains) to prevent a DR3 crash")
        # safety: should be clean now
        left = len(find_bugged(self.notes, bpms, self.offset, has_audio=True))
        if left:
            self.warnings.append(f"{left} bugged notes remain after cleanup (please report)")
        oob = sum(1 for n in self.notes
                  if n['x'] < -0.5 or n['x'] + n['width'] > HIGHWAY_WIDTH + 0.5)
        if oob:
            self.warnings.append(f"{oob} notes extend out of the 0-16 highway")

# ============================================================================
#  PART 4 -- header / data / asset output
# ============================================================================
def build_header(chart, offset):
    bpms = [(chart.ichi(t), v) for t, v in chart.bpms]
    def fnum(v):
        return f"{v:.5f}".rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
    h = [f"#OFFSET={fnum(round(offset, 5))};", "#BEAT=1;",
         f"#BPM_NUMBER={len(bpms)};"]
    for i, (pos, val) in enumerate(bpms):
        h.append(f"#BPM [{i}]={fnum(float(val))};")
        h.append(f"#BPMS[{i}]={fnum(float(pos))};")
    h += ["#SCN=1;", "#SC [0]=1.0;", "#SCI[0]=0.0;"]   # TIL/soflan ignored in v1
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
#  PART 5 -- CLI  (input .zip  ->  output DR_<name>.zip)
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

def assign_levels(items):
    """Give each chart a UNIQUE integer DR3 tier. items: list of dicts with
    'level_int' and 'const'. Processed easiest-first so collisions push the
    harder chart to the higher number. Mutates items, adding 'dr3_level'."""
    used = set()
    for it in sorted(items, key=lambda x: x['const']):
        lv = it['level_int']
        while lv in used:
            lv += 1
        it['dr3_level'] = lv; used.add(lv)

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

        # shared base name (all difficulties share one stem so DR3 groups them)
        title = meta1(items[0]['chart'], 'TITLE')
        base = (sanitize(args.name) or sanitize(title)
                or sanitize(os.path.splitext(os.path.basename(zip_path))[0]) or 'song')

        # unique DR3 tiers (only honour --level for a single-difficulty zip)
        if args.level and len(items) == 1:
            items[0]['dr3_level'] = int(re.sub(r'\D', '', args.level) or '1')
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
                format_chart(build_header(ch, offset), notes))
            summary.append(f"[{meta1(ch,'LEVEL','?')} -> tier {it['dr3_level']}] {fname}: "
                           f"{len(notes)} notes, offset {offset:+.3f}s, "
                           f"LNs {conv.ln_densified} densified / {conv.ln_preserved} kept"
                           + (f", {conv.flicks_deduped} dup flicks removed" if conv.flicks_deduped else ""))
            all_warnings += [f"(tier {it['dr3_level']}) {w}" for w in conv.warnings]

        # ---- shared assets (use the hardest difficulty's metadata) ----
        hardest = max(items, key=lambda x: x['const'])
        ch = hardest['chart']; src_dir = hardest['src_dir']
        write_data_txt_multi(items, os.path.join(stage, f"{base}.data.txt"))

        bgm = meta1(ch, 'BGM'); bgm_src = os.path.join(src_dir, bgm) if bgm else ''
        if bgm and os.path.exists(bgm_src):
            _, m = prepare_audio(bgm_src, os.path.join(stage, f"{base}.ogg"),
                                 need_seconds=max_need); msgs.append(m)
        else:
            msgs.append(f"audio '{bgm}' not found in the zip (skipped)")
        jak = meta1(ch, 'JACKET'); jak_src = os.path.join(src_dir, jak) if jak else ''
        if jak and os.path.exists(jak_src):
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

    title = meta1(items[0]['chart'], 'TITLE')
    base = sanitize(base_override) or sanitize(title) or 'song'
    if level_override and len(items) == 1:
        items[0]['dr3_level'] = int(re.sub(r'\D', '', level_override) or '1')
    else:
        assign_levels(items)

    density = parse_density(ln_density)
    stage = os.path.join(workdir, '_dr3out'); os.makedirs(stage, exist_ok=True)
    out_text, log, warnings, max_need = {}, [], [], 0.0
    for it in items:
        ch = it['chart']
        offset = ch.bgmofs * (1.0 if offset_sign == '+' else -1.0)
        conv = Converter(ch, density, head_tap=head_tap, flick_tap=flick_tap, offset=offset)
        notes = conv.convert()
        max_need = max(max_need, conv.last_note_time)
        fname = f"{base}.{it['dr3_level']}.txt"
        out_text[fname] = format_chart(build_header(ch, offset), notes)
        log.append(f"[Lv {meta1(ch,'LEVEL','?')} -> tier {it['dr3_level']}] {fname}: {len(notes)} notes, "
                   f"offset {offset:+.3f}s, LNs {conv.ln_densified} densified / {conv.ln_preserved} kept"
                   + (f", {conv.flicks_deduped} dup flicks removed" if conv.flicks_deduped else ""))
        warnings += [f"(tier {it['dr3_level']}) {w}" for w in conv.warnings]

    data_name = f"{base}.data.txt"
    data_path = os.path.join(stage, data_name)
    write_data_txt_multi(items, data_path)
    out_text[data_name] = open(data_path, encoding='utf-8', newline='').read()

    hardest = max(items, key=lambda x: x['const'])
    ch, src_dir = hardest['chart'], hardest['src_dir']
    def rel(name):
        full = os.path.join(src_dir, name)
        return os.path.relpath(full, workdir) if name and os.path.exists(full) else ''
    return json.dumps({
        'base': base,
        'out_text': out_text,                       # {filename: text} chart files + data.txt
        'ogg_name': f"{base}.ogg",  'audio_rel': rel(meta1(ch, 'BGM')),
        'png_name': f"{base}.png",  'jacket_rel': rel(meta1(ch, 'JACKET')),
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
    ap = argparse.ArgumentParser(description="Convert UMiGuri chart .zip(s) to DanceRail3 .zip(s).")
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
