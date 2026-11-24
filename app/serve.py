"""Serve the Saarth demo page:  python -m app.serve   (then open http://localhost:8000)

Python's built-in http.server cannot send part of a file, which browsers need to jump around
in a video, so the seek bar would not work with it. This small server adds that (HTTP Range).
"""
import argparse
import functools
import os
import re
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        if not m or (not m.group(1) and not m.group(2)):
            return super().send_head()
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        else:                                            # last N bytes
            start, end = max(0, size - int(m.group(2))), size - 1
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass                                         # the browser moved on (normal when seeking)
        finally:
            self._remaining = None

    def end_headers(self):
        if not self.headers.get("Range"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, *args):
        pass                                             # keep the terminal quiet


def main():
    p = argparse.ArgumentParser(description="Serve the Saarth demo page")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--run", help="open this run straight away")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(RangeHandler, directory=WEB))
    url = f"http://localhost:{args.port}/" + (f"?run={args.run}" if args.run else "")
    print(f"Saarth demo at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
