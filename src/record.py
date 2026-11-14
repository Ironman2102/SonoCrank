"""record.py <port> <seconds> <out.npy> - capture raw ADC counts streamed by the logger sketch."""
import os, sys, serial, numpy as np
port, secs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)     # create data/<condition>/ if missing
s = serial.Serial(port, 921600, timeout=1)
need, got, chunks, last, drops = secs * 16000, 0, [], None, 0
while got < need:
    if s.read(1) != b"\xA5" or s.read(1) != b"\x5A": continue
    seq = s.read(1); data = s.read(1024)
    if len(data) < 1024 or not seq: continue
    if last is not None and seq[0] != (last + 1) % 256: drops += 1
    last = seq[0]
    chunks.append(np.frombuffer(data, "<u2") & 0x0FFF)   # keep the 12-bit ADC value only
    got += 512
x = np.concatenate(chunks)[:need]
np.save(out, x); print("saved", out, len(x), "samples | dropped blocks:", drops,
      "| RMS counts: %.1f | min/max: %d/%d" % (x.astype(float).std(), x.min(), x.max()))
