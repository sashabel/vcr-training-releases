# vcr-training-releases

Published VCR training artifacts are exposed as static files under GitHub Pages. The model file to consume is always `best.pt`.

## What To Download

Each published release directory contains:

- `best.pt`: the trained model to use
- `metadata.json`: release metadata, including the published checksum for `best.pt`
- `checksums.txt`: checksum file for the published artifacts

The release index is published at:

- `https://sashabel.github.io/vcr-training-releases/`

## Recommended Consume Logic

When consuming a published model, use this flow:

1. Calculate the checksum of the current local `best.pt` if it already exists.
2. Resolve the requested published version:
   - either a specific release id such as `2026_03_28__14_28_29`
   - or the latest published release whose `Run Name` matches a regex prefix such as `^hybrid_unified_`
3. Read the published SHA256 checksum for `best.pt`.
4. If the local checksum is different, download the published `best.pt`.
5. Log each step so it is clear which release was selected and whether the local model was reused or replaced.

## Reference Script

The script below implements that logic using only the Python standard library.

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, urlretrieve


BASE_URL = "https://sashabel.github.io/vcr-training-releases"


@dataclass
class ReleaseEntry:
    trained_at: datetime
    run_name: str
    release_id: str
    sha256: str

    @property
    def release_dir_url(self) -> str:
        return f"{BASE_URL}/{self.release_id}"

    @property
    def metadata_url(self) -> str:
        return f"{self.release_dir_url}/metadata.json"

    @property
    def model_url(self) -> str:
        return f"{self.release_dir_url}/best.pt"


class ReleaseIndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_td = False
        self.current_td: list[str] = []
        self.current_row: list[dict[str, str]] = []
        self.rows: list[list[dict[str, str]]] = []
        self.current_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag == "td":
            self.in_td = True
            self.current_td = []
            self.current_href = None
        elif tag == "a" and self.in_td:
            attr_map = dict(attrs)
            self.current_href = attr_map.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_td:
            self.current_row.append(
                {"text": "".join(self.current_td).strip(), "href": self.current_href or ""}
            )
            self.in_td = False
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)

    def handle_data(self, data: str) -> None:
        if self.in_td:
            self.current_td.append(data)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def load_release_index() -> list[ReleaseEntry]:
    parser = ReleaseIndexParser()
    parser.feed(read_text(f"{BASE_URL}/"))

    releases: list[ReleaseEntry] = []
    for row in parser.rows:
        if len(row) < 12:
            continue

        trained_at = row[0]["text"]
        run_name = row[1]["text"]
        release_href = row[6]["href"].strip("/")
        sha256 = row[11]["text"]

        if not trained_at or not run_name or not release_href or not sha256:
            continue

        releases.append(
            ReleaseEntry(
                trained_at=datetime.strptime(trained_at, "%Y-%m-%d %H:%M:%S"),
                run_name=run_name,
                release_id=release_href,
                sha256=sha256,
            )
        )

    return sorted(releases, key=lambda item: item.trained_at, reverse=True)


def resolve_release(release_id: str | None, run_name_regex: str | None) -> ReleaseEntry:
    releases = load_release_index()
    if not releases:
        raise RuntimeError("No published releases found in the index.")

    if release_id:
        for release in releases:
            if release.release_id == release_id:
                return release
        raise RuntimeError(f"Requested release_id not found: {release_id}")

    if not run_name_regex:
        raise RuntimeError("Either --release-id or --run-name-regex is required.")

    pattern = re.compile(run_name_regex)
    for release in releases:
        if pattern.search(release.run_name):
            return release

    raise RuntimeError(f"No published release matched run name regex: {run_name_regex}")


def fetch_published_sha256(release: ReleaseEntry) -> str:
    metadata = json.loads(read_text(release.metadata_url))
    return metadata["checksums"]["best.pt"]["sha256"]


def download_if_needed(release: ReleaseEntry, target_path: Path) -> None:
    published_sha256 = fetch_published_sha256(release)
    local_sha256 = sha256_file(target_path)

    logging.info("Selected release_id=%s run_name=%s", release.release_id, release.run_name)
    logging.info("Published best.pt sha256=%s", published_sha256)

    if local_sha256:
        logging.info("Local best.pt sha256=%s", local_sha256)
    else:
        logging.info("Local best.pt does not exist: %s", target_path)

    if local_sha256 == published_sha256:
        logging.info("Local file already matches the published version. Reusing %s", target_path)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s -> %s", release.model_url, target_path)
    urlretrieve(release.model_url, target_path)

    downloaded_sha256 = sha256_file(target_path)
    if downloaded_sha256 != published_sha256:
        raise RuntimeError(
            "Downloaded file checksum mismatch: "
            f"expected {published_sha256}, got {downloaded_sha256}"
        )

    logging.info("Download complete. Using %s", target_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume a published best.pt model.")
    parser.add_argument("--release-id", help="Specific published release id to use.")
    parser.add_argument(
        "--run-name-regex",
        help="Regex used to select the latest published release by Run Name.",
    )
    parser.add_argument(
        "--output",
        default="models/best.pt",
        help="Local path where best.pt should be stored.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        release = resolve_release(args.release_id, args.run_name_regex)
        download_if_needed(release, Path(args.output))
    except (RuntimeError, HTTPError, URLError, ValueError, KeyError) as exc:
        logging.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Usage Examples

Download a specific published version:

```bash
python3 consume_best_pt.py \
  --release-id 2026_03_28__14_28_29 \
  --output models/best.pt
```

Download the latest published version whose `Run Name` starts with `hybrid_unified_`:

```bash
python3 consume_best_pt.py \
  --run-name-regex '^hybrid_unified_' \
  --output models/best.pt
```

## Notes

- The file you should actually run with is `best.pt`.
- The decision to download should be based on `SHA256`, not only file presence.
- If the GitHub Pages base URL changes, update `BASE_URL` in the script.
