#!/usr/bin/env python3
"""Serve the activation webapp on localhost (stdlib only).

Supports HTTP Range requests so ``<video>`` can scrub/play MP4s.

Usage::

    .venv/bin/python paper/activation_webapp/serve.py
    # then open http://127.0.0.1:8765/demo.html
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
from pathlib import Path

HERE = Path(__file__).resolve().parent


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTP + Accept-Ranges / 206 Partial Content for media scrubbing."""

    def send_head(self):  # noqa: D401 — mirror stdlib signature
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return None

        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        size = fs.st_size
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f

        # bytes=start-end
        try:
            units, _, rng = range_header.partition("=")
            if units.strip() != "bytes":
                raise ValueError("only bytes ranges")
            start_s, _, end_s = rng.strip().partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            if end >= size:
                end = size - 1
            if start > end or start < 0:
                raise ValueError("bad range")
        except ValueError:
            f.close()
            self.send_error(416, "Invalid Range")
            return None

        length = end - start + 1
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()

        # Wrap so copyfile reads only the requested length.
        class _Limited:
            def __init__(self, fp, remaining):
                self._fp = fp
                self._remaining = remaining

            def read(self, n=-1):
                if self._remaining <= 0:
                    return b""
                if n < 0 or n > self._remaining:
                    n = self._remaining
                data = self._fp.read(n)
                self._remaining -= len(data)
                return data

            def close(self):
                self._fp.close()

        return _Limited(f, length)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    os.chdir(HERE)
    handler = functools.partial(RangeRequestHandler, directory=str(HERE))

    class ReuseTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReuseTCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"serving {HERE}")
        print(f"open http://127.0.0.1:{args.port}/demo.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
