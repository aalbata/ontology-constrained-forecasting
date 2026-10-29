"""Builds auction-level snapshot tables from the bid-level Online Auctions Dataset.

A snapshot at fraction s of the scheduled duration uses only bids placed at or before
s * duration. Individual proxy amounts are private under the second-price proxy mechanism,
so the only price information taken from a snapshot is the standing price: the
second-highest bidder maximum among the bids placed so far, or the opening price when
fewer than two bidders have bid.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.75)
PREDICTION_POINTS = (0.25, 0.5, 0.75)
ITEM_CODES = {
    "Cartier wristwatch": "CartierWristwatch",
    "Palm Pilot M515 PDA": "PalmPilotM515",
    "Xbox game console": "XboxGameConsole",
}
SUPERCLASS = {"CartierWristwatch": "LuxuryGoods", "PalmPilotM515": "Electronics",
              "XboxGameConsole": "Electronics"}


def load_bids(zip_path):
    """Reads auction.csv from the Zenodo archive."""
    with zipfile.ZipFile(zip_path) as z, z.open("auction.csv") as f:
        bids = pd.read_csv(f)
    bids["duration"] = bids["auction_type"].str.extract(r"(\d+)")[0].astype(int)
    # Bids without a bidder identifier are treated as distinct anonymous bidders.
    missing = bids["bidder"].isna()
    bids.loc[missing, "bidder"] = [f"_anon{i}" for i in range(int(missing.sum()))]
    return bids


def auction_table(bids):
    """One row per auction with static attributes and the closing price."""
    g = bids.groupby("auctionid", sort=True)
    out = pd.DataFrame({
        "item": g["item"].first().map(ITEM_CODES),
        "duration": g["duration"].first(),
        "open": g["openbid"].min(),
        "price": g["price"].first(),
        "n_records": g["bid"].size(),
        "max_record": g["bid"].max(),
    })
    out["superclass"] = out["item"].map(SUPERCLASS)
    return out


def snapshot(bids, auctions, s):
    """Public state of every auction at fraction s of its scheduled duration."""
    sub = bids[bids["bidtime"] <= s * bids["duration"]]
    per_bidder = sub.groupby(["auctionid", "bidder"])["bid"].max().reset_index()
    per_bidder = per_bidder.sort_values(["auctionid", "bid"], ascending=[True, False])
    second = per_bidder.groupby("auctionid")["bid"].nth(1)
    second = pd.Series(second.to_numpy(), index=per_bidder.loc[second.index, "auctionid"].to_numpy())
    rating = sub.groupby(["auctionid", "bidder"])["bidderrate"].first().groupby("auctionid").mean()
    g = sub.groupby("auctionid")
    snap = pd.DataFrame(index=auctions.index)
    snap["bids"] = g["bid"].size().reindex(auctions.index).fillna(0).astype(int)
    snap["bidders"] = g["bidder"].nunique().reindex(auctions.index).fillna(0).astype(int)
    snap["standing"] = np.maximum(auctions["open"], second.reindex(auctions.index).fillna(-np.inf))
    snap["last_bid"] = (g["bidtime"].max().reindex(auctions.index) / auctions["duration"]).fillna(0.0)
    snap["feedback"] = rating.reindex(auctions.index)
    return snap


def build(zip_path):
    """Auction table joined with snapshots at every fraction in FRACTIONS."""
    bids = load_bids(zip_path)
    auctions = auction_table(bids)
    parts = [auctions]
    for s in FRACTIONS:
        snap = snapshot(bids, auctions, s)
        snap.columns = [f"{c}@{s:g}" for c in snap.columns]
        parts.append(snap)
    table = pd.concat(parts, axis=1)
    table.index.name = "auctionid"
    return table


def write_table(table, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, float_format="%.6f", lineterminator="\n")


def read_table(path):
    return pd.read_csv(path, index_col="auctionid")


def standing_bound(table, t):
    """Largest standing price public by fraction t (opening price included)."""
    cols = [f"standing@{s:g}" for s in FRACTIONS if s <= t]
    return np.maximum(table["open"].to_numpy(), table[cols].max(axis=1).to_numpy())


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def features(table, t):
    """Engineered features for prediction point t from the snapshots at t/2 and t."""
    X = pd.DataFrame(index=table.index)
    X["log_open"] = np.log1p(table["open"])
    X["duration"] = table["duration"]
    for tag, s in (("half", t / 2), ("now", t)):
        k = f"@{s:g}"
        X[f"bids_{tag}"] = table["bids" + k]
        X[f"bidders_{tag}"] = table["bidders" + k]
        X[f"log_standing_{tag}"] = np.log1p(table["standing" + k])
        X[f"uplift_{tag}"] = X[f"log_standing_{tag}"] - X["log_open"]
        X[f"last_bid_{tag}"] = table["last_bid" + k]
        X[f"idle_{tag}"] = np.where(table["bids" + k] > 0, s - table["last_bid" + k], s)
        X[f"bid_rate_{tag}"] = table["bids" + k] / (s * table["duration"])
        X[f"feedback_{tag}"] = _signed_log(table["feedback" + k].fillna(0.0))
    X["new_bids"] = X["bids_now"] - X["bids_half"]
    X["new_bidders"] = X["bidders_now"] - X["bidders_half"]
    X["standing_gain"] = X["log_standing_now"] - X["log_standing_half"]
    X["item"] = table["item"]
    return X


FEATURE_GROUPS = {
    "Price state and trajectory": ("log_open", "log_standing_half", "log_standing_now",
                                   "uplift_half", "uplift_now", "standing_gain"),
    "Product semantics": ("item",),
    "Temporal context": ("duration", "last_bid_half", "last_bid_now", "idle_half", "idle_now"),
    "Competition dynamics": ("bids_half", "bids_now", "bidders_half", "bidders_now",
                             "bid_rate_half", "bid_rate_now", "new_bids", "new_bidders"),
    "Bidder reputation": ("feedback_half", "feedback_now"),
}


def state_labels(table, t):
    """Ontology state classes of each auction at prediction point t."""
    k, h = f"@{t:g}", f"@{t / 2:g}"
    bidders, bids, half_bids = table["bidders" + k], table["bids" + k], table["bids" + h]
    fb = table["feedback" + k]
    uplift = np.log1p(table["standing" + k]) - np.log1p(table["open"])
    later = bids - half_bids
    lab = pd.DataFrame(index=table.index)
    lab["DurationState"] = table["duration"].map({3: "ShortAuction", 5: "MediumAuction", 7: "LongAuction"})
    lab["CompetitionState"] = np.select([bidders >= 4, bidders >= 2], ["HighCompetition", "ModerateCompetition"],
                                        "LowCompetition")
    lab["ReputationState"] = np.select([fb.isna() | (fb <= 0), fb <= 10, fb <= 100],
                                       ["UnratedBidders", "EmergingReputation", "EstablishedReputation"],
                                       "HighReputation")
    lab["MomentumState"] = np.select([bids == 0, half_bids == 0, later > half_bids],
                                     ["DormantAuction", "ActivatedAuction", "AcceleratingAuction"],
                                     "SteadyAuction")
    lab["UpliftState"] = np.select([uplift < np.log(1.10), uplift < np.log(1.75)], ["LowUplift", "ModerateUplift"],
                                   "HighUplift")
    lab["ProductSuperclass"] = table["superclass"]
    return lab
