#!/usr/bin/env python3
"""Serve the activation webapp on localhost (stdlib only).

Usage::

    .venv/bin/python paper/activation_webapp/serve.py
    # then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    os.chdir(HERE)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {HERE}")
        print(f"open http://127.0.0.1:{args.port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
