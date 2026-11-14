#!/bin/bash
# Double-click to open the EngineSense app in Google Chrome (served from this folder on http://localhost:8765).
cd "$(dirname "$0")"
lsof -ti tcp:8765 >/dev/null || (python3 -m http.server 8765 >/dev/null 2>&1 &)
sleep 1
open -a "Google Chrome" "http://localhost:8765/engine_app.html" || open "http://localhost:8765/engine_app.html"
