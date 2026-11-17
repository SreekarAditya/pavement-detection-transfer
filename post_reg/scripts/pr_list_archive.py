"""List the member names of a remote zip archive without downloading its images.

Reads only the zip central directory through HTTP range requests, so no image or label byte is fetched.
Used before the rules tag to fill images_dir, labels_dir and parent_regex in exp18_candidates.csv.

    python pr_list_archive.py URL [--out members.csv]
"""
import argparse, collections, io, sys, urllib.request, zipfile

import pandas as pd


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests; counts the bytes fetched."""

    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        # A one-byte ranged GET (some hosts refuse HEAD on signed redirect targets); the total size comes from
        # Content-Range, and the redirect target is kept for the later requests.
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()
            if r.status != 206 or "Content-Range" not in r.headers:
                sys.exit("STOP: server ignored the range request; listing would download the whole archive")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            self.fetched += len(r.read())

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206:
                sys.exit("STOP: server ignored the range request; listing would download the whole archive")
            b = r.read()
        self.pos += len(b); self.fetched += len(b)
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("--out")
    a = ap.parse_args()
    f = RangeFile(a.url)
    z = zipfile.ZipFile(f)
    rows = [dict(name=i.filename, bytes=i.file_size) for i in z.infolist() if not i.is_dir()]
    M = pd.DataFrame(rows)
    M["folder"] = M.name.str.rsplit("/", n=1).str[0].where(M.name.str.contains("/"), "")
    M["ext"] = M.name.str.rsplit(".", n=1).str[-1].str.lower()
    if a.out:
        M.to_csv(a.out, index=False)
    print(f"archive {f.size} bytes; fetched {f.fetched} bytes for the listing; {len(M)} files")
    t = M.groupby(["folder", "ext"]).size().rename("files").reset_index()
    print(t.to_string(index=False))
    for folder, g in M.groupby("folder"):
        print(f"\n{folder or '(root)'}: e.g.", ", ".join(g.name.str.rsplit("/", n=1).str[-1].head(4)))
    c = collections.Counter(M.ext)
    print("\nextensions:", dict(c))


if __name__ == "__main__":
    main()
