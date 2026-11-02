"""Dataset loading, inspection, and cleaning helpers."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


def sample_directory() -> Path:
    """Resolve the project samples folder both in source and frozen builds."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / "samples"


def list_sample_files() -> list[dict[str, Any]]:
    """List supported data files in the bundled/user-maintained samples folder."""
    folder = sample_directory()
    if not folder.is_dir():
        return []
    supported = {".csv", ".tsv", ".xlsx"}
    items = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.lower() in supported:
            items.append({"name": path.name, "path": path,
                          "extension": path.suffix.lower(),
                          "size_bytes": path.stat().st_size,
                          "category": sample_category(path.name)})
    return items


def sample_category(filename: str) -> str:
    name = filename.casefold()
    if "nlp" in name or "npl" in name or "imdb" in name or "review" in name:
        return "nlp"
    if "classification" in name or "churn" in name or "behavior" in name:
        return "classification"
    if "regression" in name or "gold" in name or "car_data" in name:
        return "regression"
    if "forecast" in name or "taxi" in name:
        return "forecasting"
    return "general"


def preview_dataset(source: Any, filename: str, rows: int = 8) -> pd.DataFrame:
    """Read only a few rows for a lightweight sample preview."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(source, nrows=rows)
    separator = "\t" if suffix == ".tsv" else ","
    try:
        return pd.read_csv(source, sep=separator, nrows=rows, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, sep=separator, nrows=rows, encoding="latin-1")


def load_dataset(source: Any, filename: str | None = None) -> pd.DataFrame:
    name = filename or getattr(source, "name", None) or str(source)
    suffix = Path(name).suffix.lower()
    if hasattr(source, "seek"):
        source.seek(0)
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        try:
            return pd.read_csv(source, sep=separator, encoding="utf-8-sig")
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, sep=separator, encoding="latin-1")
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(source)
        except ImportError as exc:
            raise RuntimeError("XLSX_READER_MISSING") from exc
    raise ValueError("Choose a CSV or Excel (.xlsx) file.")


def churn_sample() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    n = 260
    tenure = rng.integers(1, 73, n)
    monthly = rng.normal(74, 21, n).clip(18, 160).round(2)
    support = rng.poisson(1.3, n)
    plan = rng.choice(["Basic", "Plus", "Premium"], n, p=[.46, .34, .20])
    score = 1.5 - .04 * tenure + .024 * monthly + .48 * support + (plan == "Basic") * .55
    churn = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    return pd.DataFrame({"customer_tenure_months": tenure, "monthly_spend": monthly,
        "support_tickets": support, "plan": plan, "churned": churn})


def review_sample() -> pd.DataFrame:
    positives = [
        "Excellent quality, works perfectly and arrived quickly",
        "Me encantó, funciona perfectamente y llegó muy rápido",
        "Great product, easy to use and worth the price",
        "Muy buena compra, excelente calidad y buen precio",
        "I love this, very reliable and beautifully made",
        "Me gusta mucho, es fácil de usar y muy confiable",
        "Fantastic value, exceeded my expectations",
        "Excelente producto, superó todas mis expectativas",
    ]
    negatives = [
        "Poor quality, stopped working after a few days",
        "Mala calidad, dejó de funcionar después de unos días",
        "Disappointing product, difficult to use and overpriced",
        "Muy decepcionante, llegó roto y no funciona",
        "Not worth the money, unreliable and badly made",
        "No lo recomiendo, pésima calidad y mal servicio",
        "Terrible experience, the item broke immediately",
        "No vale el precio, se rompió casi de inmediato",
    ]
    rng = np.random.default_rng(17)
    texts, labels, stars = [], [], []
    for label, bank in [(1, positives), (0, negatives)]:
        for _ in range(100):
            base = bank[int(rng.integers(0, len(bank)))]
            texts.append(base if rng.random() > .35 else f"{base}. {bank[int(rng.integers(0, len(bank)))]}")
            labels.append(label)
            stars.append((5 if label else 1) if rng.random() > .18 else (4 if label else 2))
    order = rng.permutation(len(texts))
    return pd.DataFrame({"review_text": np.array(texts)[order], "stars": np.array(stars)[order],
                         "sentiment": np.array(labels)[order]})


def clean_column_names(df: pd.DataFrame, style: str) -> pd.DataFrame:
    result = df.copy()
    if style == "lowercase":
        result.columns = [str(c).lower() for c in result.columns]
    elif style == "UPPERCASE":
        result.columns = [str(c).upper() for c in result.columns]
    elif style == "Title Case":
        result.columns = [str(c).title() for c in result.columns]
    return result


def normalize_text(df: pd.DataFrame, columns: list[str], style: str) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        values = result[col].astype("string").str.strip()
        if style == "lowercase":
            values = values.str.lower()
        elif style == "UPPERCASE":
            values = values.str.upper()
        elif style == "Title Case":
            values = values.str.title()
        result[col] = values
    return result


def add_calculated_column(df: pd.DataFrame, name: str, expression: str) -> pd.DataFrame:
    if not name.strip() or name in df.columns:
        raise ValueError("Enter a unique, non-empty column name.")
    result = df.copy()
    try:
        result[name] = result.eval(expression, engine="numexpr")
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return result


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1]),
            "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 2)}


def data_quality_findings(df: pd.DataFrame, targets: list[str],
                          features: list[str], task: str) -> list[dict[str, Any]]:
    """Return conservative, actionable checks before model training."""
    findings: list[dict[str, Any]] = []
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        findings.append({"severity": "warning", "key": "quality_duplicates",
                         "count": duplicate_count})
    if len(df) < 50:
        findings.append({"severity": "info", "key": "quality_small_dataset",
                         "rows": len(df)})

    missing_alerts = []
    constants = []
    identifiers = []
    for col in features:
        series = df[col]
        missing_pct = float(series.isna().mean() * 100)
        if missing_pct >= 20:
            missing_alerts.append({"column": str(col), "percentage": round(missing_pct, 1)})
        non_null = series.dropna()
        if len(non_null) and non_null.nunique() <= 1:
            constants.append(str(col))
        unique_ratio = float(non_null.nunique() / len(non_null)) if len(non_null) else 0.0
        normalized_name = "".join(ch for ch in str(col).casefold() if ch.isalnum())
        id_name = (normalized_name in {"id", "uuid", "key", "identifier"} or
                   normalized_name.endswith(("recordid", "userid", "customerid", "rowid")))
        if (id_name or (len(non_null) >= 50 and unique_ratio >= 0.98)) and str(col) not in identifiers:
            identifiers.append(str(col))
        if pd.api.types.is_object_dtype(series) and series.nunique(dropna=True) > 50:
            findings.append({"severity": "info", "key": "quality_high_cardinality",
                             "column": str(col), "count": int(series.nunique(dropna=True))})

    for item in missing_alerts:
        findings.append({"severity": "warning", "key": "quality_missing",
                         **item})
    if constants:
        findings.append({"severity": "warning", "key": "quality_constant",
                         "columns": ", ".join(constants)})
    if identifiers:
        findings.append({"severity": "warning", "key": "quality_identifier",
                         "columns": ", ".join(identifiers)})

    for target in targets:
        if target not in df:
            continue
        target_key = "".join(ch for ch in str(target).casefold() if ch.isalnum())
        for col in features:
            if col == target or col not in df:
                continue
            feature_key = "".join(ch for ch in str(col).casefold() if ch.isalnum())
            if len(target_key) >= 4 and target_key in feature_key:
                findings.append({"severity": "warning", "key": "quality_target_name",
                                 "feature": str(col), "target": str(target)})
            try:
                aligned = df[col].notna() & df[target].notna()
                if aligned.any() and df.loc[aligned, col].equals(df.loc[aligned, target]):
                    findings.append({"severity": "warning", "key": "quality_target_copy",
                                     "feature": str(col), "target": str(target)})
            except (TypeError, ValueError):
                pass
        if task == "Classification":
            counts = df[target].value_counts(dropna=True)
            if len(counts) >= 2:
                ratio = float(counts.max() / max(counts.min(), 1))
                if ratio >= 5:
                    findings.append({"severity": "warning", "key": "quality_imbalance",
                                     "target": str(target), "ratio": round(ratio, 1)})
    return findings
