from layout_v import *
RAILS = ("L+", "L-", "R+", "R-")
def node(h):
    c, r = h
    if c == "ESP": return "ESP." + r
    if c in RAILS: return "RAIL" + c
    return ("LEFT" if c in "abcde" else "RIGHT") + str(r)
used = {}
def take(h, who):
    if h[0] in RAILS or h[0] == "ESP": return
    assert h not in used, f"hole {h} used by {used[h]} and {who}"; used[h] = who
for h, p in U1.items(): take(h, f"U1.{p}")
for pid, kind, val, step, legs, note in PARTS:
    for t, h in legs: take(h, f"{pid}.{t}")
for jid, step, a, b, col, note in JUMPERS: take(a, jid); take(b, jid)
parent = {}
def f(x):
    parent.setdefault(x, x)
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def u(a, b): parent[f(a)] = f(b)
for jid, step, a, b, col, note in JUMPERS: u(node(a), node(b))
u("ESP.GND1", "ESP.GND2")          # both GND pins are joined on the ESP32 board
def nets(pairs, pressed):
    global parent; saved = dict(parent)
    B = dict(next(p for p in PARTS if p[0] == "BTN")[4])
    for a, b in pairs: u(node(B[a]), node(B[b]))
    if pressed: u(node(B["A"]), node(B["D"]))
    term = {}
    for nm in ["3V3", "GND1", "GND2", "VIN", "D34", "D25", "D26", "D27", "D14"]: term.setdefault(f("ESP." + nm), set()).add("ESP." + nm)
    for h, p in U1.items(): term.setdefault(f(node(h)), set()).add(f"U1.{p}")
    for pid, kind, val, step, legs, note in PARTS:
        if pid == "BTN": continue
        for t, h in legs: term.setdefault(f(node(h)), set()).add(f"{pid}.{t}")
    parent = saved
    return sorted(sorted(v) for v in term.values())
EXP = [
 {"ESP.VIN","R_SUP.1","BZ.+"}, {"R_SUP.2","U1.8","C_SUP.+","C_SUP2.1"},
 {"ESP.GND1","ESP.GND2","C_REF.-","C_REF2.2","R_BOT.2","U1.4","C_SUP.-","C_SUP2.2","C_2.2","R_LY.2","R_LR.2","Q1.E"},
 {"ESP.3V3","R_TOP.1"}, {"R_TOP.2","R_BOT.1","C_REF.+","C_REF2.1","PZ.black","R_BIAS.1","C_G.-"},
 {"PZ.red","R_BIAS.2","R_IN.1"}, {"R_IN.2","U1.3"}, {"C_G.+","R_G.1"}, {"R_G.2","U1.2","R_F.2"},
 {"U1.1","R_F.1","R_1.1"}, {"R_1.2","R_2.1","C_1.1"}, {"R_2.2","U1.5","C_2.1"}, {"U1.7","U1.6","C_1.2","R_OUT.1"},
 {"R_OUT.2","ESP.D34"}, {"ESP.D27","LED_Y.A"}, {"LED_Y.K","R_LY.1"}, {"ESP.D26","LED_R.A"}, {"LED_R.K","R_LR.1"},
 {"ESP.D25","R_B.1"}, {"R_B.2","Q1.B"}, {"Q1.C","BZ.-"}, {"ESP.D14"}]
exp = sorted(sorted(s) for s in EXP); ok = True
for pairs in ([("A","C"),("B","D")], [("A","B"),("C","D")]):
    for pressed in (False, True):
        want = [list(x) for x in exp]
        if pressed:
            want.remove(["ESP.D14"]); g = [w for w in want if "ESP.GND1" in w][0]; want.remove(g); want.append(sorted(g + ["ESP.D14"])); want = sorted(want)
        got = nets(pairs, pressed)
        if got != want:
            ok = False; print("MISMATCH", pairs, pressed)
            for x in got:
                if x not in want: print("  got ", x)
            for x in want:
                if x not in got: print("  want", x)
print("netlist OK (both button orientations, pressed and released)" if ok else "NETLIST ERRORS")
rows = [r for (c, r) in used]; print("rows used:", min(rows), "to", max(rows))
