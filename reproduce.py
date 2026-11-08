"""Reproduces every result, table value, and figure of the study.

Fast path (default): runs the experiments from the shipped auction-level and daily tables.
Full path (--from-raw): downloads the raw archives (or uses --raw-dir), rebuilds every table,
checks that each rebuilt table is identical to the shipped one, then runs the experiments.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import domains
import experiment
import figures
import rolling
import snapshots

HERE = Path(__file__).resolve().parent
TABLE = HERE / "data" / "auction_snapshots.csv"
RAW_URL = "https://zenodo.org/api/records/11259259/files/Online%20Auctions%20Dataset.zip/content"
RAW_SHA256 = "a25dfdd0c4204d9f1b62983c3991ed5deef18bfdfe10c534852749a242807e4b"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _replace_or_check(rebuilt, shipped):
    if shipped.exists():
        assert rebuilt.read_bytes() == shipped.read_bytes(), f"rebuilt table differs from {shipped.name}"
        rebuilt.unlink()
        print(f"rebuilt {shipped.name} is identical to the shipped table")
    else:
        rebuilt.replace(shipped)


def rebuild_auctions(zip_path):
    zip_path = Path(zip_path) if zip_path else HERE / "data" / "raw" / "Online_Auctions_Dataset.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {RAW_URL}")
        urllib.request.urlretrieve(RAW_URL, zip_path)
    digest = sha256(zip_path)
    assert digest == RAW_SHA256, f"unexpected archive checksum {digest}"
    rebuilt = HERE / "data" / "auction_snapshots_rebuilt.csv"
    snapshots.write_table(snapshots.build(zip_path), rebuilt)
    _replace_or_check(rebuilt, TABLE)


def rebuild_days(raw_dir):
    for dataset, builder in domains.BUILDERS.items():
        local = Path(raw_dir) / f"{dataset}.zip" if raw_dir else None
        zip_path = domains.fetch(dataset, local if local and local.exists() else None)
        rebuilt = domains.TABLES[dataset].with_name(domains.TABLES[dataset].stem + "_rebuilt.csv")
        domains.write_table(builder(zip_path), rebuilt)
        _replace_or_check(rebuilt, domains.TABLES[dataset])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-raw", action="store_true", help="rebuild every table from the raw archives")
    ap.add_argument("--zip", help="path to a local copy of the Zenodo auction archive")
    ap.add_argument("--raw-dir", help="folder with local copies electricity.zip, bike.zip, weather.zip")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from existing results")
    ap.add_argument("--out", default=str(HERE / "results"), help="output directory")
    args = ap.parse_args()
    if not args.figures_only:
        if args.from_raw or not TABLE.exists():
            rebuild_auctions(args.zip)
        if args.from_raw or not all(p.exists() for p in domains.TABLES.values()):
            rebuild_days(args.raw_dir)
        experiment.run(snapshots.read_table(TABLE), args.out)
        rolling.run(args.out)
    figures.make_all(args.out, HERE / "figures")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
