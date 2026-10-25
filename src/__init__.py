import sys

for _stream in (sys.stdout, sys.stderr):          # Windows consoles default to cp1252, which cannot print Devanagari
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
