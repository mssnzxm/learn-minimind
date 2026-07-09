#!/usr/bin/env python3
"""Serve this static tutorial site locally."""

from __future__ import annotations

import argparse
import functools
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


SITE_ROOT = Path(__file__).resolve().parent


class StaticSiteHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".py": "text/plain; charset=utf-8",
    }

    def _route_home(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path in {"", "/", "/tutorial-site", "/tutorial-site/"}:
            self.path = "/index.html"
            if parsed.query:
                self.path += f"?{parsed.query}"
        elif parsed.path.startswith("/tutorial-site/"):
            self.path = parsed.path.removeprefix("/tutorial-site")
            if parsed.query:
                self.path += f"?{parsed.query}"

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        self._route_home()
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib hook name
        self._route_home()
        super().do_HEAD()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Learn MiniMind-O static site.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Use 0.0.0.0 to expose on the LAN.")
    parser.add_argument("--port", default=8765, type=int, help="Port to bind.")
    parser.add_argument("--open", action="store_true", help="Open the site URL in the default browser.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (SITE_ROOT / "index.html").exists():
        print("index.html was not found.", file=sys.stderr)
        return 1

    handler = functools.partial(StaticSiteHandler, directory=str(SITE_ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{url_host}:{args.port}/"

    print(f"Serving Learn MiniMind-O from: {SITE_ROOT}")
    print(f"URL: {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
