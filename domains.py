"""Daily tables, observable features, and rule files for the electricity, bike-sharing, and weather tasks.

Each table has one row per unit (a day, or a station-day) holding the hourly readings of that unit.
The values that become public during the day (energy used so far, rentals so far, highest and lowest
temperature so far) and the daily targets are derived from the hourly readings by `prepare`, so that
every bound holds exactly by construction.
"""
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import Namespace

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ONTOLOGY_DIR = HERE / "ontology"
POINTS = (0.25, 0.5, 0.75)
HOURS = {0.25: 6, 0.5: 12, 0.75: 18}
H24 = [f"{h:02d}" for h in range(24)]

EL = Namespace("http://example.org/household-energy#")
BK = Namespace("http://example.org/bike-sharing#")
WX = Namespace("http://example.org/station-weather#")

RAW = {
    "electricity": ("https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
                    "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"),
    "bike": ("https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
             "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"),
    "weather": ("https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
                "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"),
}
TABLES = {"electricity": DATA / "electricity_days.csv", "bike": DATA / "bike_days.csv",
          "weather": DATA / "weather_station_days.csv"}


@dataclass(frozen=True)
class Task:
    """One forecasting task: its data, rule files, target, bound columns, and evaluation settings."""
    name: str
    dataset: str
    ontology: str
    shapes: str
    namespace: Namespace
    unit_class: object
    prediction_class: object
    predicted: object
    target: object
    target_column: str
    columns: dict
    scale: str
    unit: str
    first_test_month: str


def _bound_columns(ns, stem, col_stem):
    return {ns[f"{stem}By{h:02d}"]: f"{col_stem}_by{h:02d}" for h in HOURS.values()}


TASKS = {
    "electricity": Task("electricity", "electricity", "electricity_ontology.ttl", "electricity_shapes.ttl", EL,
                        EL.Day, EL.EnergyForecast, EL.predictedDailyEnergy, EL.dailyEnergy, "daily_energy",
                        {**_bound_columns(EL, "energy", "energy"), EL.dailyEnergy: "daily_energy"},
                        "log", "kWh", "2008-01"),
    "bike": Task("bike", "bike", "bike_ontology.ttl", "bike_shapes.ttl", BK,
                 BK.Day, BK.RentalForecast, BK.predictedDailyRentals, BK.dailyRentals, "daily_rentals",
                 {**_bound_columns(BK, "rentals", "rentals"), BK.dailyRentals: "daily_rentals"},
                 "log", "rentals", "2012-01"),
    "weather_max": Task("weather_max", "weather", "weather_ontology.ttl", "weather_shapes.ttl", WX,
                        WX.StationDay, WX.TemperatureForecast, WX.predictedTemperature, WX.dailyMaxTemperature,
                        "daily_tmax", {**_bound_columns(WX, "maxTemperature", "tmax"), WX.dailyMaxTemperature: "daily_tmax"},
                        "identity", "deg C", "2014-03"),
    "weather_min": Task("weather_min", "weather", "weather_ontology.ttl", "weather_shapes.ttl", WX,
                        WX.StationDay, WX.TemperatureForecast, WX.predictedTemperature, WX.dailyMinTemperature,
                        "daily_tmin", {**_bound_columns(WX, "minTemperature", "tmin"), WX.dailyMinTemperature: "daily_tmin"},
                        "identity", "deg C", "2014-03"),
}


# ---------------------------------------------------------------- raw archives to daily tables
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(dataset, path=None):
    """Local copy of a raw archive, downloaded from the UCI repository when absent; checksum verified."""
    url, digest = RAW[dataset]
    path = Path(path) if path else DATA / "raw" / f"{dataset}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    found = sha256(path)
    assert found == digest, f"unexpected checksum {found} for {path}"
    return path


def electricity_table(zip_path):
    """Hourly active energy and sub-metered energy (kWh) for every day with all 1,440 minutes recorded."""
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open("household_power_consumption.txt"), sep=";", na_values="?", low_memory=False)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.assign(date=ts.dt.strftime("%Y-%m-%d"), hour=ts.dt.hour)
    complete = df.groupby("date")["Global_active_power"].count() == 1440
    df = df[df["date"].isin(complete[complete].index)]
    parts = {"e": df["Global_active_power"] / 60.0}
    for i in (1, 2, 3):
        parts[f"s{i}"] = df[f"Sub_metering_{i}"] / 1000.0
    frames = []
    for stem, values in parts.items():
        wide = values.groupby([df["date"], df["hour"]]).sum().unstack("hour")
        wide.columns = [f"{stem}{h:02d}" for h in wide.columns]
        frames.append(wide)
    table = pd.concat(frames, axis=1)
    table.index.name = "unit"
    return table


def bike_table(zip_path):
    """Hourly rentals and weather of every day; hours absent from hour.csv had no rentals."""
    with zipfile.ZipFile(zip_path) as z:
        hours = pd.read_csv(z.open("hour.csv"))
        days = pd.read_csv(z.open("day.csv")).set_index("dteday")
    frames = []
    for col, stem, fill in (("cnt", "c", 0), ("casual", "ca", 0), ("registered", "r", 0), ("temp", "tp", None),
                            ("hum", "hu", None), ("windspeed", "ws", None), ("weathersit", "wc", None)):
        wide = hours.pivot_table(index="dteday", columns="hr", values=col, aggfunc="first").reindex(columns=range(24))
        if fill is not None:
            wide = wide.fillna(fill)
        wide.columns = [f"{stem}{h:02d}" for h in range(24)]
        frames.append(wide)
    table = pd.concat(frames, axis=1)
    assert (table[[f"c{h}" for h in H24]].sum(axis=1).to_numpy() == days.loc[table.index, "cnt"].to_numpy()).all()
    for col in ("weekday", "workingday", "holiday"):
        table[col] = days.loc[table.index, col].to_numpy()
    table.index.name = "unit"
    return table


def distinct_weather_sites(df):
    """Sites whose hourly temperature series is not a copy of an alphabetically earlier site's series.

    The data set matches the meteorological readings of each air-quality site with the nearest weather
    station, so several sites share one series; one site is kept per distinct series.
    """
    temps = df.pivot_table(index=["year", "month", "day", "hour"], columns="station", values="TEMP", aggfunc="first")
    keep = []
    for site in sorted(temps.columns):
        both = [temps[[site, k]].dropna() for k in keep]
        if not any(len(b) and (b[site] == b[k]).mean() > 0.99 for b, k in zip(both, keep)):
            keep.append(site)
    return keep


def weather_table(zip_path):
    """Hourly readings of the Beijing sites with distinct weather series, for every station-day with all
    24 temperatures recorded."""
    with zipfile.ZipFile(zip_path) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read("PRSA2017_Data_20130301-20170228.zip")))
        df = pd.concat([pd.read_csv(inner.open(n)) for n in sorted(inner.namelist()) if n.endswith(".csv")],
                       ignore_index=True)
    df = df[df["station"].isin(distinct_weather_sites(df))].copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]]).dt.strftime("%Y-%m-%d")
    frames = []
    for col, stem in (("TEMP", "T"), ("DEWP", "D"), ("PRES", "P"), ("WSPM", "W"), ("RAIN", "R"), ("wd", "wd")):
        wide = df.pivot_table(index=["station", "date"], columns="hour", values=col, aggfunc="first",
                              dropna=False).reindex(columns=range(24))
        wide.columns = [f"{stem}{h:02d}" for h in range(24)]
        frames.append(wide)
    table = pd.concat(frames, axis=1)
    table = table[table[[f"T{h}" for h in H24]].notna().all(axis=1)].reset_index()
    table.index = table["station"] + "_" + table["date"]
    table.index.name = "unit"
    return table


BUILDERS = {"electricity": electricity_table, "bike": bike_table, "weather": weather_table}


def write_table(table, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, float_format="%.10g", lineterminator="\n")


def read_table(dataset):
    return prepare(dataset, pd.read_csv(TABLES[dataset], index_col="unit"))


# ---------------------------------------------------------------- public values, targets, and features
def prepare(dataset, table):
    """Adds the date, the values public at 06:00, 12:00 and 18:00, and the daily targets."""
    table = table.copy()
    if dataset == "weather":
        table["date"] = pd.to_datetime(table["date"])
        temps = table[[f"T{h}" for h in H24]].to_numpy(dtype=float)
        hi, lo = np.maximum.accumulate(temps, axis=1), np.minimum.accumulate(temps, axis=1)
        for h in HOURS.values():
            table[f"tmax_by{h:02d}"], table[f"tmin_by{h:02d}"] = hi[:, h - 1], lo[:, h - 1]
        table["daily_tmax"], table["daily_tmin"] = hi[:, -1], lo[:, -1]
        return table.sort_values(["station", "date"])
    table["date"] = pd.to_datetime(table.index)
    stem, name = ("e", "energy") if dataset == "electricity" else ("c", "rentals")
    cum = np.cumsum(table[[f"{stem}{h}" for h in H24]].to_numpy(dtype=float), axis=1)
    for h in HOURS.values():
        table[f"{name}_by{h:02d}"] = cum[:, h - 1]
    table[f"daily_{name}"] = cum[:, -1]
    return table.sort_values("date")


def _lag(table, col, days, by=None):
    """Value of `col` on the same unit `days` earlier, or NaN when that day is not in the table."""
    key = table["date"] - pd.Timedelta(days=days)
    if by is None:
        ref = pd.Series(table[col].to_numpy(), index=table["date"])
        return ref.reindex(key).to_numpy()
    ref = pd.Series(table[col].to_numpy(), index=pd.MultiIndex.from_arrays([table[by], table["date"]]))
    return ref.reindex(pd.MultiIndex.from_arrays([table[by], key])).to_numpy()


def _season(table):
    angle = 2 * np.pi * (table["date"].dt.month.to_numpy() - 1) / 12
    return np.sin(angle), np.cos(angle)


def features(dataset, table, t):
    """Features public at fraction t of the day; returns the feature table and its categorical columns."""
    k = HOURS[t]
    X = pd.DataFrame(index=table.index)
    if dataset in ("electricity", "bike"):
        stem, name = ("e", "energy") if dataset == "electricity" else ("c", "rentals")
        hourly = table[[f"{stem}{h}" for h in H24]].to_numpy(dtype=float)
        so_far = table[f"{name}_by{k:02d}"].to_numpy()
        X["log_so_far"] = np.log1p(so_far)
        X["log_last_hour"] = np.log1p(hourly[:, k - 1])
        X["log_last_3h"] = np.log1p(hourly[:, k - 3:k].sum(axis=1))
        for days, tag in ((1, "prev"), (7, "week")):
            X[f"log_{tag}_total"] = np.log1p(_lag(table, f"daily_{name}", days))
            X[f"log_{tag}_same_time"] = np.log1p(_lag(table, f"{name}_by{k:02d}", days))
        if dataset == "electricity":
            for i in (1, 2, 3):
                sub = table[[f"s{i}{h}" for h in H24[:k]]].to_numpy(dtype=float).sum(axis=1)
                X[f"sub{i}_share"] = sub / so_far
            weekday = table["date"].dt.dayofweek.to_numpy()
        else:
            for stem2, tag in (("ca", "casual"), ("r", "registered")):
                X[f"log_{tag}_so_far"] = np.log1p(table[[f"{stem2}{h}" for h in H24[:k]]].to_numpy(dtype=float).sum(axis=1))
            obs = {s: table[[f"{s}{h}" for h in H24[:k]]] for s in ("tp", "hu", "ws", "wc")}
            X["temp_last"] = obs["tp"].ffill(axis=1).iloc[:, -1].to_numpy()
            X["temp_mean"] = obs["tp"].mean(axis=1).to_numpy()
            X["humidity_mean"] = obs["hu"].mean(axis=1).to_numpy()
            X["wind_mean"] = obs["ws"].mean(axis=1).to_numpy()
            X["weather_worst"] = obs["wc"].max(axis=1).to_numpy()
            X["workingday"] = table["workingday"].to_numpy()
            X["holiday"] = table["holiday"].to_numpy()
            weekday = table["weekday"].to_numpy()
        X["weekday"] = pd.Series(weekday, index=table.index).astype(str)
        X["month_sin"], X["month_cos"] = _season(table)
        return X, ["weekday"]
    temps = table[[f"T{h}" for h in H24]].to_numpy(dtype=float)
    X["max_so_far"] = table[f"tmax_by{k:02d}"].to_numpy()
    X["min_so_far"] = table[f"tmin_by{k:02d}"].to_numpy()
    X["temp_last"] = temps[:, k - 1]
    X["temp_change_3h"] = temps[:, k - 1] - temps[:, k - 4]
    X["dewpoint_last"] = table[f"D{k - 1:02d}"].to_numpy()
    X["pressure_last"] = table[f"P{k - 1:02d}"].to_numpy()
    X["pressure_change_3h"] = table[f"P{k - 1:02d}"].to_numpy() - table[f"P{k - 4:02d}"].to_numpy()
    X["wind_mean"] = table[[f"W{h}" for h in H24[:k]]].mean(axis=1).to_numpy()
    X["rain_so_far"] = table[[f"R{h}" for h in H24[:k]]].sum(axis=1, min_count=1).to_numpy()
    X["prev_tmax"] = _lag(table, "daily_tmax", 1, by="station")
    X["prev_tmin"] = _lag(table, "daily_tmin", 1, by="station")
    X["month_sin"], X["month_cos"] = _season(table)
    X["wind_direction"] = table[f"wd{k - 1:02d}"].fillna("missing").astype(str).to_numpy()
    X["station"] = table["station"].astype(str).to_numpy()
    return X, ["wind_direction", "station"]
