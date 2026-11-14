"""Physical clearance check: no two parts on different connections may come closer than 3 mm.
Positions in mm: rows 2.54 mm apart, columns 2.54 mm, centre channel 7.62 mm, rails as on the GL-12."""
import math, itertools
from layout_v import *
from verify_v import node, f as find, JUMPERS as _J
XMM = {"L+": -8.3, "L-": -5.4, "a": 0, "b": 2.54, "c": 5.08, "d": 7.62, "e": 10.16, "f": 17.78, "g": 20.32, "h": 22.86, "i": 25.4, "j": 27.94, "R-": 33.3, "R+": 36.2}
def xy(h): return (XMM[h[0]], h[1] * 2.54)
def seg_dist(p, q, r, s):
    def pd(a, b, c):
        bx, by = c[0] - b[0], c[1] - b[1]; L = bx * bx + by * by
        t = 0 if L == 0 else max(0, min(1, ((a[0] - b[0]) * bx + (a[1] - b[1]) * by) / L))
        return math.hypot(a[0] - b[0] - t * bx, a[1] - b[1] - t * by)
    return min(pd(p, r, s), pd(q, r, s), pd(r, p, q), pd(s, p, q))
segs = []
for pid, kind, val, step, legs, note in PARTS:
    if kind in ("piezo", "buzzer"): continue                 # insulated wire leads
    hs = [h for _, h in legs]
    if kind == "button": hs = [legs[0][1], legs[3][1]]
    if kind == "npn": hs = [hs[0], hs[2]]
    nets = {find(node(h)) for _, h in legs}
    segs.append((pid, xy(hs[0]), xy(hs[1]), nets))
for (h, p) in U1.items(): segs.append((f'U1.{p}', xy(h), xy(h), {find(node(h))}))
bad = []
for (a, p1, p2, n1), (b, q1, q2, n2) in itertools.combinations(segs, 2):
    if n1 & n2 or (a.startswith('U1') and b.startswith('U1')): continue                                      # share a connection: touching is harmless
    d = seg_dist(p1, p2, q1, q2)
    if d < 3.0: bad.append((round(d, 2), a, b))
near = sorted((round(seg_dist(p1,p2,q1,q2),1),a,b) for (a,p1,p2,n1),(b,q1,q2,n2) in itertools.combinations(segs,2) if not (n1&n2) and not (a.startswith('U1') and b.startswith('U1')))
print('closest pairs:', near[:5])
print("parts closer than 3 mm on different connections:", sorted(bad) if bad else "none")

used = {}
for pid, kind, val, step, legs, note in PARTS:
    for _, h in legs: used[h] = pid
for h, p in U1.items(): used[h] = f"U1.{p}"
for jid, step, a, b, col, note in JUMPERS: used[a] = jid; used[b] = jid
blocked = []
for pid, kind, val, step, legs, note in PARTS + [(j[0], "jumper", "", 0, [("1", j[2]), ("2", j[3])], "") for j in JUMPERS]:
    if kind not in ("res", "jumper") or len(legs) != 2: continue
    (_, h1), (_, h2) = legs
    if h1[0] == "ESP" or h2[0] == "ESP" or h1[0] != h2[0]: continue
    lo, hi = sorted((h1[1], h2[1]))
    for r in range(lo + 1, hi):
        if (h1[0], r) in used: blocked.append((pid, h1[0] + str(r), used[(h1[0], r)]))
print("flat parts lying over a hole that is in use:", blocked if blocked else "none")
