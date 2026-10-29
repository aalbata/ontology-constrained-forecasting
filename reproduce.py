"""Reproduces every result, table value, and figure of the study.

Fast path (default): runs the experiments from the shipped auction-level snapshot table.
Full path (--from-raw): downloads the bid-level archive from Zenodo (or uses --zip),
rebuilds the snapshot table, checks it against the shipped table, then runs the experiments.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import experiment
import figures
import snapshots

HERE = Path(__file__).resolve().parent
TABLE = HERE / "data" / "auction_snapshots.csv"
RAW_URL = "https://zenodo.org/api/records/11259259/files/Online%20Auctions%20Dataset.zip/content"
RAW_SHA256 = "a25dfdd0c4204d9f1b62983c3991ed5deef18bfdfe10c534852749a242807e4b"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rebuild(zip_path):
    zip_path = Path(zip_path) if zip_path else HERE / "data" / "raw" / "Online_Auctions_Dataset.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {RAW_URL}")
        urllib.request.urlretrieve(RAW_URL, zip_path)
    digest = sha256(zip_path)
    assert digest == RAW_SHA256, f"unexpected archive checksum {digest}"
    rebuilt = HERE / "data" / "auction_snapshots_rebuilt.csv"
    snapshots.write_table(snapshots.build(zip_path), rebuilt)
    if TABLE.exists():
        assert rebuilt.read_bytes() == TABLE.read_bytes(), "rebuilt table differs from the shipped table"
        rebuilt.unlink()
        print("rebuilt snapshot table is identical to the shipped table")
    else:
        rebuilt.replace(TABLE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-raw", action="store_true", help="rebuild the snapshot table from the Zenodo archive")
    ap.add_argument("--zip", help="path to a local copy of the Zenodo archive")
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    args = ap.parse_args()
    if args.from_raw or not TABLE.exists():
        rebuild(args.zip)
    table = snapshots.read_table(TABLE)
    experiment.run(table, args.out)
    figures.make_all(args.out, HERE / "figures")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
