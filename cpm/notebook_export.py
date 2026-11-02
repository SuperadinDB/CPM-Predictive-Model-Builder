"""Build a Jupyter notebook that reproduces an app experiment using CPM source modules."""
from __future__ import annotations

import json
from typing import Any


def _cell(cell_type: str, source: str) -> dict[str, Any]:
    return {"cell_type": cell_type, "metadata": {},
            "source": source.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if cell_type == "code" else {})}


def build_notebook(experiment: dict[str, Any], *, forecast: bool = False) -> bytes:
    """Create an .ipynb with exact settings and reproducible calls used by the app."""
    dataset = experiment.get("dataset", {}).get("name", "your_dataset.csv")
    common = [
        _cell("markdown", "# CPM Model Builder · Reproducible experiment\n\n"
              "This notebook repeats the settings recorded by CPM Model Builder. "
              "Keep the project folder (including `cpm/`) beside this notebook, install `requirements.txt`, "
              "and update `DATA_PATH` if your source file is elsewhere. CPM records configuration and transformations, "
              "not a copy of your private dataset.\n"),
        _cell("code", "from pathlib import Path\nimport json\nimport pandas as pd\n\n"
              f"DATA_PATH = Path({dataset!r})\n"
              "if DATA_PATH.suffix.lower() in {'.xlsx', '.xls'}:\n    df = pd.read_excel(DATA_PATH)\n"
              "elif DATA_PATH.suffix.lower() == '.tsv':\n    df = pd.read_csv(DATA_PATH, sep='\\t')\n"
              "else:\n    df = pd.read_csv(DATA_PATH)\n"
              "print(f'{df.shape[0]:,} rows × {df.shape[1]:,} columns')\ndf.head()\n"),
    ]
    if forecast:
        params = experiment["forecast_config"]
        common.append(_cell("code", "from cpm.data import clean_column_names, normalize_text, add_calculated_column\n\n"
            f"cleaning_steps = {experiment.get('cleaning_steps', [])!r}\n"
            "for step in cleaning_steps:\n"
            "    action, details = step['action'], step['details']\n"
            "    if action == 'standardize_column_names':\n        df = clean_column_names(df, details['style'])\n"
            "    elif action == 'normalize_text':\n        df = normalize_text(df, details['columns'], details['style'])\n"
            "    elif action == 'remove_columns':\n        df = df.drop(columns=details['columns'])\n"
            "    elif action == 'add_calculated_column':\n        df = add_calculated_column(df, details['name'], details['expression'])\n"
            "    elif action == 'remove_duplicate_rows':\n        df = df.drop_duplicates().reset_index(drop=True)\n"
            "print(df.shape)\n"))
        code = (
            "from cpm.forecasting import train_forecast\n\n"
            f"config = {params!r}\n"
            "result = train_forecast(\n"
            "    df,\n"
            "    time_column=config['time_column'],\n"
            "    target_column=config['target_column'],\n"
            "    frequency=config['frequency'],\n"
            "    aggregation=config['aggregation'],\n"
            "    missing_policy=config['missing_policy'],\n"
            "    horizon=config['horizon'], lags=config['lags'], windows=config['windows'],\n"
            "    test_size=config['test_size'], model_name=config['model_name'], cfg=config['model_config'],\n"
            ")\n"
            "print('Overall metrics:', result['metrics'])\n"
            "print('Last-value baseline:', result['baseline_metrics'])\n"
            "display(result['horizon_metrics'])\n"
            "display(result['future'])\n"
            "display(result['holdout'].head())\n"
        )
        common.extend([
            _cell("markdown", "## Forecasting\n\nThe app regularizes the selected series, creates past-only lag and rolling features, "
                  "evaluates forecasts on a chronological holdout, compares a last-observed-value baseline, and fits one direct estimator per lead.\n"),
            _cell("code", code),
        ])
    else:
        common.extend([
            _cell("markdown", "## Replay cleaning and train the selected model\n\n"
                  "The notebook calls the same modular preprocessing/training functions as the desktop app. "
                  "Review `cleaning_steps`, feature/target selection, model settings, and holdout split below.\n"),
            _cell("code", "from cpm.data import clean_column_names, normalize_text, add_calculated_column\n"
                  "from cpm.models import train_model, set_classification_threshold\n\n"
                  f"cleaning_steps = {experiment.get('cleaning_steps', [])!r}\n"
                  "for step in cleaning_steps:\n"
                  "    action, details = step['action'], step['details']\n"
                  "    if action == 'standardize_column_names':\n"
                  "        df = clean_column_names(df, details['style'])\n"
                  "    elif action == 'normalize_text':\n"
                  "        df = normalize_text(df, details['columns'], details['style'])\n"
                  "    elif action == 'remove_columns':\n"
                  "        df = df.drop(columns=details['columns'])\n"
                  "    elif action == 'add_calculated_column':\n"
                  "        df = add_calculated_column(df, details['name'], details['expression'])\n"
                  "    elif action == 'remove_duplicate_rows':\n"
                  "        df = df.drop_duplicates().reset_index(drop=True)\n"
                  f"experiment = {experiment!r}\n"
                  "results = []\n"
                  "for target in experiment['targets']:\n"
                  "    for model_name in experiment['models_requested']:\n"
                  "        result = train_model(df, experiment['features'], target, model_name,\n"
                  "            experiment['task'], experiment['training_settings']['parameters'],\n"
                  "            experiment.get('text_columns', []))\n"
                  "        threshold = experiment.get('classification_threshold')\n"
                  "        if threshold is not None and result.get('probabilities') is not None:\n"
                  "            set_classification_threshold(result, threshold)\n"
                  "        results.append(result)\n"
                  "        print(target, model_name, result['metrics'])\n"),
        ])
    common.append(_cell("markdown", "## Notes\n\n"
                        "The holdout score is an estimate, not a guarantee of future performance. "
                        "For NLP, use clean labels, remove duplicate or near-duplicate text across the split, "
                        "inspect class-level errors, tune n-grams and vocabulary size, and use a split that reflects the real deployment setting.\n"))
    notebook = {"cells": common, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(notebook, ensure_ascii=False, indent=2, default=str).encode("utf-8")
