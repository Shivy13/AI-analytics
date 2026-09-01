"""CSV loading, type inference and the in-memory dataset store."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

MAX_ROWS = 250_000

BOOL_TRUE = {"true", "t", "yes", "y", "1", "oui", "si", "haan"}
BOOL_FALSE = {"false", "f", "no", "n", "0", "non", "nahi"}
DATE_SAMPLE = 3000


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def sniff_delimiter(text: str) -> str:
    head = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except Exception:
        counts = {d: head.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else ","


def read_csv_bytes(raw: bytes, name: str = "upload.csv") -> pd.DataFrame:
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    delim = sniff_delimiter(text)
    df = pd.read_csv(
        io.StringIO(text),
        sep=delim,
        skipinitialspace=True,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "n/a", "null", "NULL", "None", "-", "--", "nan"],
        dtype=str,
        low_memory=False,
    )
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    # de-duplicate column names
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed: \d+$")]
    if len(df) > MAX_ROWS:
        df = df.iloc[:MAX_ROWS]
    return df


# --------------------------------------------------------------------------
# type inference
# --------------------------------------------------------------------------
def infer_semantic_types(df: pd.DataFrame) -> dict[str, str]:
    n = len(df)
    types: dict[str, str] = {}
    for col in df.columns:
        s = df[col]
        non_null = s.notna().sum()
        if non_null == 0:
            types[col] = "empty"
            continue

        # boolean
        low = s.dropna().astype(str).str.strip().str.lower()
        uniq = set(low.unique())
        if uniq and uniq <= (BOOL_TRUE | BOOL_FALSE):
            types[col] = "boolean"
            continue

        # numeric
        num = pd.to_numeric(s, errors="coerce")
        num_non_null = num.notna().sum()
        if num_non_null >= max(1, int(0.97 * non_null)):
            types[col] = "numeric"
            continue

        # datetime (only for string-ish columns)
        if s.dtype == object or str(s.dtype).startswith("string"):
            sample = s.dropna().astype(str).head(DATE_SAMPLE)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
            ok = parsed.notna().sum()
            if len(sample) and ok >= 0.9 * len(sample):
                types[col] = "datetime"
                continue

        vals = low.unique()
        avg_len = float(low.astype(str).str.len().mean())
        avg_words = float(low.astype(str).str.split().str.len().mean())
        distinct_ratio = len(vals) / max(1, non_null)

        if avg_words >= 5 or avg_len > 40:
            types[col] = "text"
        elif re.search(r"(^|[_\s\-])(id|uid|guid|code|no|number|num)$", _norm(col)) and distinct_ratio >= 0.3:
            types[col] = "id"
        elif distinct_ratio >= 0.95 and n >= 20:
            types[col] = "id"
        elif len(vals) <= max(50, int(0.05 * n)) or distinct_ratio < 0.5:
            types[col] = "categorical"
        else:
            types[col] = "text"
    return types


def coerce_series(df: pd.DataFrame, col: str, sem: str) -> pd.Series:
    s = df[col]
    if sem == "numeric":
        return pd.to_numeric(s, errors="coerce")
    if sem == "datetime":
        return pd.to_datetime(s.astype("string"), errors="coerce", format="mixed", dayfirst=False)
    if sem == "boolean":
        low = s.astype("string").str.strip().str.lower()
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out[low.isin(BOOL_TRUE)] = 1.0
        out[low.isin(BOOL_FALSE)] = 0.0
        return out
    return s.astype("string").str.strip()


class Dataset:
    def __init__(self, dataset_id: str, name: str, df: pd.DataFrame):
        self.id = dataset_id
        self.name = name
        self.df = df
        self.created = datetime.now(timezone.utc).isoformat()
        self.types = infer_semantic_types(df)
        self._series: dict[str, pd.Series] = {}
        self.profile: dict | None = None
        self.insights: dict | None = None

    def series(self, col: str) -> pd.Series:
        if col not in self._series:
            sem = self.types.get(col, "text")
            self._series[col] = coerce_series(self.df, col, sem)
        return self._series[col]

    @property
    def rows(self) -> int:
        return len(self.df)

    @property
    def numeric_cols(self) -> list[str]:
        return [c for c in self.df.columns if self.types.get(c) == "numeric"]

    @property
    def categorical_cols(self) -> list[str]:
        return [c for c in self.df.columns if self.types.get(c) == "categorical"]

    @property
    def datetime_cols(self) -> list[str]:
        return [c for c in self.df.columns if self.types.get(c) == "datetime"]

    def meta(self) -> dict:
        counts: dict[str, int] = {}
        for t in self.types.values():
            counts[t] = counts.get(t, 0) + 1
        return {
            "id": self.id,
            "name": self.name,
            "rows": self.rows,
            "columns": len(self.df.columns),
            "created": self.created,
            "type_counts": counts,
            "column_names": list(self.df.columns),
        }


# --------------------------------------------------------------------------
# store (survives restarts: CSVs are kept on disk)
# --------------------------------------------------------------------------
class DatasetStore:
    def __init__(self):
        self._items: dict[str, Dataset] = {}
        self.index_path = os.path.join(DATA_DIR, "index.json")
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                meta = json.load(open(self.index_path))
            except Exception:
                meta = {}
            for did, info in meta.items():
                path = os.path.join(DATA_DIR, f"{did}.csv")
                if os.path.exists(path):
                    try:
                        self._items[did] = self._open(did, info.get("name", did + ".csv"), path)
                    except Exception:
                        pass

    def _save_index(self):
        json.dump({d: {"name": v.name, "created": v.created} for d, v in self._items.items()},
                  open(self.index_path, "w"), indent=2)

    @staticmethod
    def _open(did: str, name: str, path: str) -> Dataset:
        df = pd.read_csv(path, dtype=str, keep_default_na=True,
                         na_values=["", "NA", "N/A", "n/a", "null", "None", "-", "--"],
                         low_memory=False)
        return Dataset(did, name, df)

    def add(self, name: str, df: pd.DataFrame) -> Dataset:
        did = uuid.uuid4().hex[:12]
        df.to_csv(os.path.join(DATA_DIR, f"{did}.csv"), index=False)
        ds = Dataset(did, name, df)
        self._items[did] = ds
        self._save_index()
        return ds

    def get(self, did: str) -> Dataset | None:
        return self._items.get(did)

    def remove(self, did: str) -> bool:
        if did in self._items:
            del self._items[did]
            p = os.path.join(DATA_DIR, f"{did}.csv")
            if os.path.exists(p):
                os.remove(p)
            self._save_index()
            return True
        return False

    def list(self) -> list[dict]:
        return [d.meta() for d in sorted(self._items.values(), key=lambda x: x.created, reverse=True)]


STORE = DatasetStore()


def fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update((",".join(df.columns)).encode())
    h.update(str(len(df)).encode())
    return h.hexdigest()[:10]
