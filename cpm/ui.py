"""Bilingual Streamlit user interface for CPM Model Builder."""
from __future__ import annotations

import io
import json
import joblib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from cpm.data import (add_calculated_column, clean_column_names, data_quality_findings, dataset_summary,
                      list_sample_files, load_dataset, normalize_text, preview_dataset)
from cpm.i18n import translator
from cpm.forecasting import FREQUENCIES, predict_forecast_bundle, train_forecast, parse_positive_ints
from cpm.notebook_export import build_notebook
from cpm.models import (available_models, feature_impact, predict_bundle,
                        quality_feedback, set_classification_threshold, train_model)
from cpm.trace import record_event

ACCENT = "#5ce1e6"


def _theme() -> None:
    st.markdown(f"""
    <style>
    :root {{ --ink:#e8f1f8; --muted:#91a6b7; --panel:#101d2c; --line:#203548; --accent:{ACCENT}; }}
    .stApp {{ background:radial-gradient(ellipse at 12% 0%,#153b4c 0%,#0a1420 38%,#08111a 100%); color:var(--ink); }}
    [data-testid="stHeader"] {{ background:rgba(8,17,26,.82); }}
    [data-testid="stSidebar"] {{ background:#0c1824; border-right:1px solid #203548; }}
    .block-container {{ max-width:1280px; padding-top:2rem; padding-bottom:2.5rem; }}
    html,body,[class*="css"] {{ font-family:'Segoe UI',Inter,Arial,sans-serif; }}
    h1,h2,h3 {{ font-family:'Segoe UI',Inter,Arial,sans-serif; letter-spacing:-.035em; color:#f3f8fc; }}
    .hero {{ position:relative; overflow:hidden; padding:2rem 2.2rem; border:1px solid #25485a; border-radius:22px;
      background:linear-gradient(115deg,rgba(16,39,54,.97),rgba(16,29,45,.88) 58%,rgba(41,38,73,.8));
      box-shadow:0 18px 55px rgba(0,0,0,.22); margin-bottom:1.35rem; }}
    .hero:after {{ content:""; position:absolute; right:-55px; top:-130px; width:360px; height:360px; border:1px solid rgba(92,225,230,.13); border-radius:50%; box-shadow:0 0 0 32px rgba(92,225,230,.025),0 0 0 70px rgba(92,225,230,.02); }}
    .eyebrow {{ color:var(--accent); font-size:.74rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }}
    .hero p {{ color:#aabac8; max-width:760px; font-size:1.03rem; margin-bottom:0; }}
    .step-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin:1rem 0 1.6rem; }}
    .step-chip {{ background:rgba(16,29,44,.78); border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; color:#a7bac9; font-size:.78rem; font-weight:700; letter-spacing:.04em; }}
    .step-chip span {{ color:var(--accent); margin-right:.5rem; }}
    .step-chip.active {{ border-color:var(--accent); background:rgba(31,157,172,.17); color:#f4fbff; box-shadow:0 0 18px rgba(92,225,230,.08); }}
    .step-chip.done {{ border-color:#2e6670; color:#d1e9ed; }}
    .guide-card {{ border:1px solid #244858; background:linear-gradient(120deg,rgba(16,44,58,.76),rgba(16,29,44,.6)); border-radius:14px; padding:1rem 1.15rem; margin:.65rem 0 1rem; color:#c2d3df; }}
    .guide-card strong {{ color:var(--accent); }}
    div[data-testid="stMetric"] {{ background:rgba(16,29,44,.82); border:1px solid #203548; padding:1rem; border-radius:14px; }}
    div[data-testid="stMetricLabel"] p {{ color:#a7bac9; }}
    .stButton>button {{ border-radius:10px; border-color:#315166; min-height:2.7rem; font-weight:600; }}
    .stButton>button[kind="primary"] {{ background:linear-gradient(100deg,#20b9c8,#5ce1e6); color:#07151d; border:0; }}
    [data-testid="stExpander"] {{ background:rgba(16,29,44,.55); border:1px solid #203548; border-radius:14px; }}
    .hint {{ color:#91a6b7; font-size:.9rem; }}
    .footer {{ text-align:center; color:#8298aa; font-size:.85rem; padding:1.2rem 0 .2rem; }}
    a {{ color:var(--accent)!important; }}
    </style>
    """, unsafe_allow_html=True)


def _header(t, current_step: int) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{t("brand")} &nbsp; / &nbsp; {t("tagline")}</div>'
        f'<h1>{t("hero_title")}</h1><p>{t("hero_subtitle")}</p></div>',
        unsafe_allow_html=True)
    labels = [t("step_data"), t("step_prepare"), t("step_model"), t("step_results")]
    chips = []
    for index, label in enumerate(labels, start=1):
        state = "active" if index == current_step else "done" if index < current_step else ""
        chips.append(f'<div class="step-chip {state}">{label}</div>')
    st.markdown('<div class="step-row">' + "".join(chips) + "</div>", unsafe_allow_html=True)


def _profile(df: pd.DataFrame, t) -> None:
    summary = dataset_summary(df)
    cols = st.columns(5)
    labels = [("rows", f'{summary["rows"]:,}'), ("columns", summary["columns"]),
              ("missing", f'{summary["missing"]:,}'), ("duplicates", summary["duplicates"]),
              ("memory", f'{summary["memory_mb"]} MB')]
    for col, (key, val) in zip(cols, labels):
        col.metric(t(key), val)
    with st.expander(t("dataset_details"), expanded=False):
        details = pd.DataFrame({
            t("column"): df.columns,
            t("dtype"): [str(dtype) for dtype in df.dtypes],
            t("nonnull"): df.notna().sum().values,
            t("missing"): df.isna().sum().values,
            t("unique"): df.nunique(dropna=True).values,
        })
        st.dataframe(details, use_container_width=True, hide_index=True)
        left, right = st.columns([1.3, 1])
        with left:
            st.caption(t("first_rows"))
            st.dataframe(df.head(12), use_container_width=True, hide_index=True)
        with right:
            st.caption(t("numeric_summary"))
            st.dataframe(df.describe(include="all").transpose().fillna("—"),
                         use_container_width=True)


def _record_cleaning_step(action: str, details: dict) -> None:
    history = st.session_state.setdefault("cleaning_steps", [])
    history.append({"action": action, "details": details})
    st.session_state.pop("forecast_result", None)
    code_calls = {
        "standardize_column_names": f"df = clean_column_names(df, {details.get('style')!r})",
        "remove_columns": f"df = df.drop(columns={details.get('columns', [])!r})",
        "normalize_text": f"df = normalize_text(df, {details.get('columns', [])!r}, {details.get('style')!r})",
        "add_calculated_column": f"df = add_calculated_column(df, {details.get('name')!r}, {details.get('expression')!r})",
        "remove_duplicate_rows": "df = df.drop_duplicates().reset_index(drop=True)",
    }
    record_event(st.session_state, "data cleaning",
        code_calls.get(action, f"df = apply_cleaning_step(df, {action!r}, {details!r})"), details,
        "Cleaning operation applied to the in-memory working copy")


def _cleaning(df: pd.DataFrame, t) -> pd.DataFrame:
    st.markdown(f'<div class="guide-card"><strong>{t("current_step")}:</strong> {t("guide_prepare")}</div>',
                unsafe_allow_html=True)
    st.subheader(t("clean_title"))
    st.caption(t("clean_desc"))
    with st.expander("✦  " + t("clean_title"), expanded=False):
        style_labels = [t("unchanged"), t("lower"), t("upper"), t("title_case")]
        style_map = dict(zip(style_labels, ["keep", "lowercase", "UPPERCASE", "Title Case"]))
        style = st.selectbox(t("column_case"), style_labels)
        if st.button(t("apply_names"), key="apply_names"):
            _record_cleaning_step("standardize_column_names", {"style": style_map[style]})
            st.session_state.data = clean_column_names(df, style_map[style])
            st.session_state.results = {}
            st.rerun()

        df = st.session_state.data
        c1, c2 = st.columns([2, 1])
        with c1:
            drops = st.multiselect(t("drop_columns"), list(df.columns), key="drop_cols")
        with c2:
            st.write("")
            st.write("")
            if st.button(t("remove_columns"), disabled=not drops):
                _record_cleaning_step("remove_columns", {"columns": [str(col) for col in drops]})
                st.session_state.data = df.drop(columns=drops)
                st.session_state.results = {}
                st.rerun()

        text_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)
        if text_cols:
            st.markdown(f"**{t('text_cleanup')}**")
            c1, c2 = st.columns(2)
            with c1:
                selected = st.multiselect(t("text_columns"), text_cols, key="normalize_cols")
            with c2:
                format_labels = [t("lower"), t("upper"), t("title_case")]
                selected_format = st.selectbox(t("format"), format_labels, key="normalize_format")
            if st.button(t("apply_text"), disabled=not selected):
                style = dict(zip(format_labels, ["lowercase", "UPPERCASE", "Title Case"]))[selected_format]
                _record_cleaning_step("normalize_text", {"columns": [str(col) for col in selected],
                                                          "style": style})
                st.session_state.data = normalize_text(df, selected, style)
                st.session_state.results = {}
                st.rerun()
        st.markdown(f"**{t('calculated')}**")
        c1, c2 = st.columns(2)
        name = c1.text_input(t("new_name"), key="calculated_name")
        expression = c2.text_input(t("expression"), key="calculated_expression")
        st.caption(t("expression_help"))
        if st.button(t("add_calculated"), disabled=not name or not expression):
            try:
                st.session_state.data = add_calculated_column(df, name, expression)
                _record_cleaning_step("add_calculated_column", {"name": name, "expression": expression})
                st.session_state.results = {}
                st.rerun()
            except ValueError as exc:
                st.error(t("error_expr", error=str(exc)))
        if st.button(t("dedupe")):
            removed = int(df.duplicated().sum())
            _record_cleaning_step("remove_duplicate_rows", {"rows_removed": removed})
            st.session_state.data = df.drop_duplicates().reset_index(drop=True)
            st.session_state.results = {}
            st.rerun()
    return st.session_state.data


def _data_input(t) -> None:
    st.markdown(f'<div class="eyebrow">{t("step_data")}</div>', unsafe_allow_html=True)
    st.subheader(t("data_title"))
    st.caption(t("data_desc"))
    st.markdown(f'<div class="guide-card"><strong>{t("current_step")}:</strong> {t("guide_data")}</div>',
                unsafe_allow_html=True)
    source_choices = [t("upload"), t("samples_title")]
    source_choice = st.radio(t("data_source"), source_choices, horizontal=True,
                             label_visibility="collapsed", key="data_source")
    if source_choice == source_choices[0]:
        uploaded = st.file_uploader(t("upload"), type=["csv", "tsv", "xlsx"], key="dataset_upload")
        if uploaded:
            if st.session_state.get("source_name") != uploaded.name:
                try:
                    st.session_state.data = load_dataset(uploaded)
                    st.session_state.source_name = uploaded.name
                    st.session_state.results = {}
                    st.session_state.pop("forecast_result", None)
                    st.session_state.cleaning_steps = []
                    record_event(st.session_state, "data loading", "df = load_dataset(uploaded_file)",
                        {"file": uploaded.name}, f"Loaded {len(st.session_state.data):,} rows × {len(st.session_state.data.columns):,} columns")
                except RuntimeError as exc:
                    if str(exc) == "XLSX_READER_MISSING":
                        st.error(t("xlsx_error"))
                    else:
                        st.error(t("error_read", error=str(exc)))
                except Exception as exc:
                    st.error(t("error_read", error=str(exc)))
        return

    st.markdown(f"**{t('samples_title')}**")
    st.caption(t("samples_desc"))
    samples = list_sample_files()
    if not samples:
        st.info(t("no_samples"))
        return
    names = [item["name"] for item in samples]
    selected_name = st.selectbox(t("samples_select"), names, key="sample_file_select")
    selected = next(item for item in samples if item["name"] == selected_name)
    category_key = f'category_{selected["category"]}'
    size_mb = selected["size_bytes"] / 1024**2
    left, right = st.columns([1, 3])
    left.metric(t("sample_category"), t(category_key))
    right.metric(t("sample_size"), f"{size_mb:.2f} MB")
    try:
        with st.expander(t("sample_preview"), expanded=True):
            st.dataframe(preview_dataset(selected["path"], selected["name"]),
                         use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(t("error_read", error=str(exc)))
    if st.button(t("load_sample"), type="primary", use_container_width=True):
        try:
            with st.spinner(t("sample_loading")):
                st.session_state.data = load_dataset(selected["path"], selected["name"])
                st.session_state.source_name = f'sample:{selected["name"]}'
                st.session_state.results = {}
                st.session_state.pop("forecast_result", None)
                st.session_state.cleaning_steps = []
                record_event(st.session_state, "sample loading", "df = load_dataset(sample_path)",
                    {"file": selected["name"], "category": selected["category"]},
                    f"Loaded {len(st.session_state.data):,} rows × {len(st.session_state.data.columns):,} columns")
            st.rerun()
        except Exception as exc:
            st.error(t("error_read", error=str(exc)))


def _model_upload(t) -> None:
    with st.expander("↗  " + t("saved_model"), expanded=False):
        st.caption(t("saved_model_desc"))
        st.caption(t("security_note"))
        model_file = st.file_uploader(t("model_file"), type=["joblib"], key="saved_model")
        if model_file:
            try:
                bundle = joblib.load(model_file)
                compatible_tabular = isinstance(bundle, dict) and "pipeline" in bundle and "feature_names" in bundle
                compatible_forecast = isinstance(bundle, dict) and bundle.get("format") == "cpm-forecast-v1"
                if not (compatible_tabular or compatible_forecast):
                    st.error(t("no_model"))
                else:
                    st.session_state.loaded_bundle = bundle
                    record_event(st.session_state, "model bundle loading", "bundle = joblib.load(model_file)",
                        {"model_type": "forecasting" if compatible_forecast else bundle.get("task"),
                         "model": bundle.get("model_name")}, "Compatible model bundle loaded")
            except Exception as exc:
                st.error(t("error_read", error=str(exc)))
        bundle = st.session_state.get("loaded_bundle")
        if not bundle:
            return
        if bundle.get("format") == "cpm-forecast-v1":
            st.caption(f'{t("forecast_time")}: {bundle["time_column"]} · {t("forecast_target")}: {bundle["target"]}')
        else:
            st.caption(f'{t("expected_features")}: {", ".join(bundle["feature_names"])}')
        new_file = st.file_uploader(t("prediction_file"), type=["csv", "tsv", "xlsx"], key="prediction_data")
        if not new_file:
            return
        try:
            frame = load_dataset(new_file)
            if bundle.get("format") == "cpm-forecast-v1":
                output = predict_forecast_bundle(bundle, frame)
            else:
                preds = predict_bundle(bundle, frame)
                output = frame.copy()
                output[f'{bundle.get("target", "target")}_prediction'] = preds
            st.markdown(f"**{t('run_predictions')}**")
            st.dataframe(output.head(30), use_container_width=True, hide_index=True)
            record_event(st.session_state, "loaded model prediction",
                "predictions = predict_forecast_bundle(bundle, history)" if bundle.get("format") == "cpm-forecast-v1" else
                    "predictions = predict_bundle(bundle, frame)",
                {"file": new_file.name, "rows": len(frame), "model": bundle.get("model_name")},
                f"Generated {len(output):,} prediction row(s)")
            st.download_button(t("download_new_preds"),
                output.to_csv(index=False).encode("utf-8"),
                "cpm_predictions.csv", "text/csv")
        except ValueError as exc:
            message = str(exc)
            if message.startswith("MISSING_FEATURES:"):
                st.error(t("missing_features", columns=message.split(":", 1)[1]))
            else:
                st.error(t("error_read", error=message))
        except RuntimeError as exc:
            st.error(t("xlsx_error") if str(exc) == "XLSX_READER_MISSING" else t("error_read", error=str(exc)))
        except Exception as exc:
            st.error(t("error_read", error=str(exc)))


def _training(df: pd.DataFrame, t) -> None:
    st.markdown(f'<div class="eyebrow">{t("step_model")}</div>', unsafe_allow_html=True)
    st.subheader(t("configure_title"))
    st.markdown(f'<div class="guide-card"><strong>{t("current_step")}:</strong> {t("guide_model")}</div>',
                unsafe_allow_html=True)
    if len(df.columns) < 2:
        st.info(t("need_columns"))
        return

    task_labels = {"Classification": t("classification"), "Regression": t("regression"),
                   "Forecasting": t("forecasting")}
    task = st.radio(t("task"), list(task_labels), format_func=lambda option: task_labels[option],
                    horizontal=True, key="task_type")
    if task == "Forecasting":
        _forecasting_panel(df, t)
        return
    targets = st.multiselect(t("targets"), list(df.columns), help=t("targets_help"), key="target_cols")
    candidates = [col for col in df.columns if col not in targets]
    features = st.multiselect(t("features"), candidates, default=candidates,
                              help=t("features_help"), key="feature_cols")

    text_candidates = list(df.select_dtypes(include=["object", "string", "category"]).columns)
    use_tfidf = st.checkbox(t("nlp"), help=t("nlp_help"), key="enable_tfidf")
    text_columns: list[str] = []
    if use_tfidf:
        st.caption(t("nlp_model_note"))
        with st.expander(t("nlp_guidance"), expanded=False):
            st.info(t("nlp_guidance_text"))
        possible_text = [col for col in text_candidates if col not in targets]
        text_columns = st.multiselect(t("nlp_columns"), possible_text,
            default=possible_text[:1], help=t("nlp_help"), key="tfidf_columns")
        if not text_columns:
            st.info(t("need_features"))
    model_options = available_models(task, use_tfidf)
    model = st.selectbox(t("algorithm"), model_options, key="algorithm")
    compare = False
    if not use_tfidf:
        compare = st.checkbox(t("compare_models"), help=t("compare_help"), key="compare_models")
        if compare:
            st.caption(t("long_compare_help"))
    description_key = {
        "Logistic Regression": "algorithm_logistic", "Linear Regression": "algorithm_linear",
        "Random Forest": "algorithm_forest", "Gradient Boosting": "algorithm_boosting",
        "CatBoost": "algorithm_catboost", "XGBoost": "algorithm_xgboost",
        "LightGBM": "algorithm_lightgbm",
    }.get(model)
    if description_key:
        st.caption(t(description_key))

    with st.expander(t("preprocess"), expanded=False):
        c1, c2 = st.columns(2)
        ohe = c1.checkbox(t("onehot"), value=True, help=t("onehot_help"), key="ohe")
        scaler_labels = [t("no_scaling"), t("standard"), t("robust")]
        scaler_label = c2.selectbox(t("scaler"), scaler_labels, help=t("scaler_help"))
        scaler = dict(zip(scaler_labels, [None, "StandardScaler", "RobustScaler"]))[scaler_label]

    ngram_range, max_features, stop_words = (1, 1), 20000, None
    if use_tfidf:
        with st.expander(t("tfidf_options"), expanded=True):
            c1, c2, c3 = st.columns(3)
            max_features = int(c1.number_input(t("max_features"), 100, 100000, 20000, step=1000))
            ngram_label = c2.selectbox(t("ngram"), [t("ngram_uni"), t("ngram_bi")])
            ngram_range = (1, 1) if ngram_label == t("ngram_uni") else (1, 2)
            stop_label = c3.selectbox(t("stop_words"),
                [t("no_stop"), t("english_stop"), t("spanish_stop")])
            stop_words = dict(zip([t("no_stop"), t("english_stop"), t("spanish_stop")],
                                  [None, "english", "spanish"]))[stop_label]

    with st.expander(t("parameters"), expanded=False):
        c1, c2, c3 = st.columns(3)
        tree_model = model in ["Random Forest", "Gradient Boosting", "CatBoost", "XGBoost", "LightGBM"]
        estimators = c1.number_input(t("estimators"), 25, 2000, 250, step=25, disabled=not tree_model)
        max_depth = c2.number_input(t("depth"), 0, 100, 0, disabled=model not in ["Random Forest", "Gradient Boosting", "CatBoost", "XGBoost", "LightGBM"])
        leaf = c3.number_input(t("leaf"), 1, 100, 1, disabled=model != "Random Forest")
        c1, c2, c3 = st.columns(3)
        learning_rate = c1.number_input(t("learning"), .001, 1.0, .1, step=.01,
            format="%.3f", disabled=model not in ["Gradient Boosting", "CatBoost", "XGBoost", "LightGBM"])
        supports_balance = model in ["Logistic Regression", "Random Forest", "CatBoost", "XGBoost", "LightGBM"]
        if model == "XGBoost" and targets:
            supports_balance = all(df[target].nunique(dropna=True) == 2 for target in targets)
        balance = c2.checkbox(t("balance"), disabled=task != "Classification" or not supports_balance,
                              help=t("balance_help"))
        test_percent = c3.slider(t("test_split"), 10, 40, 20)
        c1, c2, c3 = st.columns(3)
        max_iter = c1.number_input(t("max_iter"), 100, 5000, 1000, step=100,
                                   disabled=model != "Logistic Regression")
        C = c2.number_input(t("regularization"), .001, 100.0, 1.0, format="%.3f",
                            disabled=model != "Logistic Regression")
        seed = c3.number_input(t("seed"), 0, 999999, 42)

    if task == "Classification" and targets and any(df[target].nunique(dropna=True) > 100 for target in targets):
        st.warning(t("bad_target"))
    findings = data_quality_findings(df, targets, features, task) if targets else []
    with st.expander(t("quality_review"), expanded=bool(findings)):
        st.caption(t("quality_review_help"))
        if findings:
            for finding in findings:
                message = t(finding["key"], **{key: value for key, value in finding.items()
                    if key not in {"severity", "key"}})
                if finding["severity"] == "warning":
                    st.warning(message)
                else:
                    st.info(message)
        else:
            st.success(t("quality_clean"))
    disabled = not targets or (not features and not text_columns) or (use_tfidf and not text_columns)
    train_label = t("train_and_compare") if compare else t("train_selected")
    if st.button(train_label, type="primary", use_container_width=True, disabled=disabled):
        cfg = {"ohe": ohe, "scaler": scaler, "estimators": int(estimators),
            "max_depth": int(max_depth), "min_samples_leaf": int(leaf),
            "learning_rate": float(learning_rate), "balance": balance,
            "test_size": test_percent / 100, "seed": int(seed), "max_iter": int(max_iter),
            "C": float(C), "max_features": max_features, "ngram_range": ngram_range,
            "stop_words": stop_words, "class_weight_ratio": 1.0}
        selected_models = model_options if compare else [model]
        pairs = [(target, candidate_model) for target in targets for candidate_model in selected_models]
        results, skipped, failed = [], [], []
        progress = st.progress(0, text=t("training", target="…"))
        for index, (target, candidate_model) in enumerate(pairs):
            progress.progress(index / len(pairs),
                text=t("training_model", model=candidate_model, target=target))
            try:
                result = train_model(df, features, target, candidate_model, task, cfg, text_columns)
                result["target"] = target
                result["run_config"] = dict(cfg)
                results.append(result)
                record_event(st.session_state, "model training",
                    "result = train_model(df, features, target, model_name, task, cfg, text_columns)",
                    {"target": str(target), "model": candidate_model, "task": task,
                     "features": [str(col) for col in features], "text_columns": [str(col) for col in text_columns],
                     "config": cfg}, f"Training finished; metrics={result['metrics']}")
            except ImportError as exc:
                error = str(exc)
                if error.startswith("OPTIONAL_MODEL:"):
                    _, skipped_model, package = error.split(":")
                    skipped.append(f"{skipped_model} ({package})")
                else:
                    failed.append(f"{candidate_model}: {error}")
            except Exception as exc:
                error = str(exc)
                messages = {"SMALL_DATA": t("small_data"), "SINGLE_CLASS": t("single_class"),
                            "BAD_TARGET": t("bad_target")}
                failed.append(f'{candidate_model}: {messages.get(error, error)}')
                record_event(st.session_state, "model training failed",
                    "result = train_model(df, features, target, model_name, task, cfg, text_columns)",
                    {"target": str(target), "model": candidate_model, "task": task, "config": cfg}, error)
        progress.progress(1.0, text=t("training_finished"))
        if not results:
            st.error(t("error_train", error="; ".join(failed or skipped)))
            return
        st.session_state.results = results
        st.session_state.model_errors = {
            "skipped": list(dict.fromkeys(skipped)),
            "failed": list(dict.fromkeys(failed)),
        }
        st.session_state.experiment = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {"name": st.session_state.get("source_name", "uploaded dataset"),
                        "rows": int(len(df)), "columns": [str(col) for col in df.columns]},
            "task": task, "targets": [str(item) for item in targets],
            "features": [str(item) for item in features],
            "text_columns": [str(item) for item in text_columns],
            "models_requested": selected_models, "compare_models": compare,
            "preprocessing": {"one_hot_encoding": bool(ohe), "numeric_scaler": scaler,
                              "tfidf": {"enabled": bool(use_tfidf),
                                  "max_features": max_features, "ngram_range": ngram_range,
                                  "stop_words": stop_words}},
            "training_settings": {"test_size": cfg["test_size"], "seed": cfg["seed"],
                                  "parameters": cfg},
            "data_quality_findings": findings,
            "cleaning_steps": list(st.session_state.get("cleaning_steps", [])),
        }
        record_event(st.session_state, "experiment export metadata",
            "experiment = {'task': task, 'targets': targets, 'features': features, 'parameters': cfg}",
            {"task": task, "targets": [str(item) for item in targets], "models": selected_models,
             "test_size": cfg["test_size"], "seed": cfg["seed"]},
            f"Completed {len(results)} model run(s)")
        st.rerun()

    results = st.session_state.get("results", [])
    if not results:
        return
    st.markdown(f'<div class="eyebrow">{t("step_results")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="guide-card"><strong>{t("current_step")}:</strong> {t("guide_results")}</div>',
                unsafe_allow_html=True)
    st.caption(t("results_run_note"))
    run_errors = st.session_state.get("model_errors", {})
    if run_errors.get("skipped"):
        st.warning(t("models_skipped", models=", ".join(run_errors["skipped"])))
    if run_errors.get("failed"):
        st.warning(t("models_failed", models="; ".join(run_errors["failed"])))

    target_order = list(dict.fromkeys(result["target"] for result in results))
    for target in target_order:
        target_results = [result for result in results if result["target"] == target]
        st.divider()
        st.subheader(str(target))
        localized_metric = {"Accuracy": t("metric_accuracy_label"),
            "Precision (weighted)": t("metric_precision_label"),
            "Recall (weighted)": t("metric_recall_label"), "F1 (weighted)": t("metric_f1_label"),
            "ROC AUC": t("metric_auc_label"), "MAE (EAM)": t("metric_mae_label"),
            "RMSE": t("metric_rmse_label"), "R²": t("metric_r2_label")}
        if len(target_results) > 1:
            table = pd.DataFrame([{t("model_column"): result["model_name"],
                **{localized_metric[key]: value for key, value in result["metrics"].items()}}
                for result in target_results]).set_index(t("model_column"))
            st.markdown(f"**{t('comparison_title')}**")
            styled = table.style.format("{:.3f}", na_rep="—")
            for metric, column in localized_metric.items():
                if column in table.columns:
                    styled = styled.highlight_min(subset=[column], color="#185b55") if metric in {"MAE (EAM)", "RMSE"} else styled.highlight_max(subset=[column], color="#185b55")
            st.dataframe(styled, use_container_width=True)
            rank_metric = "F1 (weighted)" if target_results[0]["task"] == "Classification" else "R²"
            available_ranked = [r for r in target_results if pd.notna(r["metrics"].get(rank_metric))]
            if available_ranked:
                best = max(available_ranked, key=lambda item: item["metrics"][rank_metric])
                st.metric(t("best_model"), f'{best["model_name"]} · {best["metrics"][rank_metric]:.3f}',
                          help=t("best_model_help"))

        if target_results[0]["task"] == "Classification" and all(
                len(result.get("class_labels", [])) == 2 for result in target_results):
            threshold = st.slider(t("threshold"), .05, .95, .50, .05,
                key=f'threshold_{str(target)}', help=t("threshold_help"))
            positive = target_results[0]["class_labels"][-1]
            st.caption(f'{t("positive_class")}: {positive}')
            for result in target_results:
                set_classification_threshold(result, threshold)

        metric_help = {"Accuracy": "metric_accuracy", "Precision (weighted)": "metric_precision",
            "Recall (weighted)": "metric_recall", "F1 (weighted)": "metric_f1", "ROC AUC": "metric_auc",
            "MAE (EAM)": "metric_mae", "RMSE": "metric_rmse", "R²": "metric_r2"}
        for result in target_results:
            st.markdown(f'#### {result["model_name"]}')
            st.caption(f'{t("trained_on")}: {result["train_rows"]:,}  ·  {t("tested_on")}: {result["test_rows"]:,}')
            metrics = [(key, value) for key, value in result["metrics"].items() if value is not None]
            cols = st.columns(len(metrics))
            for col, (key, value) in zip(cols, metrics):
                col.metric(localized_metric[key], f"{value:.3f}", help=t(metric_help[key]))

            heading, message = quality_feedback(result)
            notify = st.success if heading == "quality_strong" else st.info if heading == "quality_promising" else st.warning
            notify(f'**{t(heading)}.** {t(message)}')

            with st.expander(t("feature_impact"), expanded=False):
                st.caption(t("feature_impact_help"))
                impact = feature_impact(result)
                if impact.empty:
                    st.info(t("no_feature_impact"))
                else:
                    directions = {"Positive": t("positive_direction"), "Negative": t("negative_direction"),
                                  "Neutral": t("neutral_direction"), "—": "—"}
                    impact["Direction"] = impact["Direction"].map(directions).fillna(impact["Direction"])
                    impact.rename(columns={"Feature": t("feature"), "Importance": t("importance"),
                                           "Direction": t("direction")}, inplace=True)
                    st.dataframe(impact, use_container_width=True, hide_index=True)
                    chart = impact.set_index(t("feature"))[[t("importance")]]
                    st.bar_chart(chart)

            with st.expander(t("prediction_sample"), expanded=False):
                if result["task"] == "Classification":
                    report = classification_report(result["y_test"], result["predictions"],
                        output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose().round(3)
                    report_df.rename(columns={"precision": t("report_precision"),
                        "recall": t("report_recall"), "f1-score": t("report_f1"),
                        "support": t("report_support")}, inplace=True)
                    report_df.rename(index={"accuracy": t("report_accuracy"),
                        "macro avg": t("report_macro_avg"),
                        "weighted avg": t("report_weighted_avg")}, inplace=True)
                    st.dataframe(report_df, use_container_width=True)
                    labels = sorted(pd.unique(pd.concat([result["y_test"],
                        pd.Series(result["predictions"])])), key=str)
                    cm = confusion_matrix(result["y_test"], result["predictions"], labels=labels)
                    st.caption(t("confusion_help"))
                    st.dataframe(pd.DataFrame(cm, index=[f'{t("actual")} · {x}' for x in labels],
                        columns=[f'{t("predicted")} · {x}' for x in labels]), use_container_width=True)
                else:
                    actual = result["y_test"].to_numpy()
                    pred = result["predictions"]
                    actual_label, predicted_label = t("actual"), t("predicted")
                    st.dataframe(pd.DataFrame({actual_label: actual, predicted_label: pred,
                        t("residual"): actual - pred}).head(40), use_container_width=True, hide_index=True)
                    st.scatter_chart(pd.DataFrame({actual_label: actual, predicted_label: pred}),
                                     x=actual_label, y=predicted_label)

            pred_frame = pd.DataFrame({t("actual"): result["y_test"].to_numpy(),
                                       t("predicted"): result["predictions"]})
            if result["task"] == "Classification":
                pred_frame.columns = [t("actual_class"), t("predicted_class")]
            else:
                pred_frame.columns = [t("actual_value"), t("predicted_value")]
            b1, b2 = st.columns(2)
            safe_model_name = result["model_name"].lower().replace(" ", "_")
            safe_target = str(target).lower().replace(" ", "_")
            b1.download_button(t("download_preds"), pred_frame.to_csv(index=False).encode("utf-8"),
                f"{safe_target}_{safe_model_name}_predictions.csv", "text/csv",
                key=f'pred_{safe_target}_{safe_model_name}', on_click=_record_download,
                args=("evaluation_predictions_csv", {"target": str(target), "model": result["model_name"]}))
            bundle = {key: result[key] for key in ["pipeline", "feature_names", "target", "task",
                "model_name", "text_columns", "label_encoder", "threshold", "run_config"]}
            payload = io.BytesIO()
            joblib.dump(bundle, payload)
            b2.download_button(t("download_model"), payload.getvalue(),
                f"cpm_{safe_target}_{safe_model_name}.joblib", "application/octet-stream",
                key=f'bundle_{safe_target}_{safe_model_name}', on_click=_record_download,
                args=("tabular_model_joblib", {"target": str(target), "model": result["model_name"]}))

    experiment = st.session_state.get("experiment")
    if experiment:
        st.divider()
        st.markdown(f"**{t('report_title')}**")
        st.caption(t("report_help"))
        report = dict(experiment)
        report["model_results"] = [{
            "target": str(result["target"]), "model": result["model_name"],
            "metrics": {key: (float(value) if pd.notna(value) else None)
                        for key, value in result["metrics"].items()},
            "classification_threshold": result.get("threshold"),
            "training_rows": result["train_rows"], "evaluation_rows": result["test_rows"],
        } for result in results]
        report["models_skipped"] = run_errors.get("skipped", [])
        report["models_failed"] = run_errors.get("failed", [])
        report["package_versions"] = {}
        for package in ["streamlit", "pandas", "numpy", "scikit-learn",
                        "openpyxl", "catboost", "xgboost", "lightgbm"]:
            try:
                report["package_versions"][package] = version(package)
            except PackageNotFoundError:
                report["package_versions"][package] = "unknown"
        report["application_version"] = "1.0.0"
        st.download_button(t("download_report"), json.dumps(report, ensure_ascii=False,
            indent=2, default=str).encode("utf-8"), "cpm_experiment_report.json",
            "application/json", use_container_width=True, on_click=_record_download,
            args=("experiment_report_json", {"task": experiment.get("task"), "models": experiment.get("models_requested")}))
        notebook = build_notebook({**experiment,
            "classification_threshold": next((r.get("threshold") for r in results
                if r.get("threshold") is not None), None)})
        st.caption(t("notebook_help"))
        st.download_button(t("download_notebook"), notebook,
            "cpm_reproducible_experiment.ipynb", "application/x-ipynb+json",
            use_container_width=True, on_click=_record_download,
            args=("reproducible_jupyter_notebook", {"task": experiment.get("task"),
                "targets": experiment.get("targets"), "models": experiment.get("models_requested")}))


def _forecasting_panel(df: pd.DataFrame, t) -> None:
    """Guided single-series forecasting workflow and reproducibility exports."""
    st.markdown(f'<div class="guide-card"><strong>{t("current_step")}:</strong> {t("forecast_guide")}</div>',
                unsafe_allow_html=True)
    st.caption(t("forecast_guide"))
    left, right = st.columns(2)
    time_column = left.selectbox(t("forecast_time"), list(df.columns), key="forecast_time_col")
    numeric = [str(col) for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric:
        numeric = [str(col) for col in df.columns]
    default_target = numeric[0] if numeric else str(df.columns[-1])
    target_column = right.selectbox(t("forecast_target"), numeric, index=numeric.index(default_target),
                                    key="forecast_target_col")
    freq_labels = [t("auto_frequency")] + list(FREQUENCIES)
    freq_choice = st.selectbox(t("forecast_frequency"), freq_labels, key="forecast_freq")
    frequency = None if freq_choice == t("auto_frequency") else FREQUENCIES[freq_choice]
    agg_labels = [t("agg_sum"), t("agg_mean"), t("agg_last")]
    aggregation = dict(zip(agg_labels, ["sum", "mean", "last"]))[
        st.selectbox(t("forecast_aggregation"), agg_labels, key="forecast_agg")]
    missing_labels = [t("missing_keep"), t("missing_zero"), t("missing_ffill")]
    missing_policy = dict(zip(missing_labels, ["skip", "zero", "forward_fill"]))[
        st.selectbox(t("forecast_missing"), missing_labels, key="forecast_missing_policy")]
    st.markdown(f"**{t('forecast_lags')}**")
    lag_text = st.text_input(t("forecast_lags"), "1,2,3,7", help=t("forecast_lags_help"), key="forecast_lag_text")
    window_text = st.text_input(t("forecast_windows"), "3,7", help=t("forecast_windows_help"), key="forecast_window_text")
    st.caption(t("forecast_lags_help") + " " + t("forecast_windows_help"))
    horizon = st.number_input(t("forecast_horizon"), min_value=1, max_value=365, value=7, key="forecast_horizon")
    model_options = available_models("Regression")
    model_name = st.selectbox(t("forecast_model"), model_options, key="forecast_model_name")
    with st.expander(t("parameters"), expanded=False):
        col1, col2, col3 = st.columns(3)
        estimators = col1.number_input(t("estimators"), 25, 2000, 250, step=25, key="forecast_estimators")
        depth = col2.number_input(t("depth"), 0, 100, 0, key="forecast_depth")
        learning = col3.number_input(t("learning"), .001, 1.0, .1, step=.01, format="%.3f", key="forecast_learning")
        col1, col2, col3 = st.columns(3)
        test_percent = col1.slider(t("forecast_test"), 10, 40, 20, key="forecast_test_pct")
        scaler_labels = [t("no_scaling"), t("standard"), t("robust")]
        scaler = dict(zip(scaler_labels, [None, "StandardScaler", "RobustScaler"]))[
            col2.selectbox(t("forecast_scaler"), scaler_labels, key="forecast_scaler")]
        seed = col3.number_input(t("seed"), 0, 999999, 42, key="forecast_seed")
        leaf = st.number_input(t("leaf"), 1, 100, 1, key="forecast_leaf",
                               disabled=model_name != "Random Forest")
    run = st.button(t("train_forecast"), type="primary", use_container_width=True,
                    key="run_forecasting", disabled=time_column == target_column)
    if run:
        try:
            lags = parse_positive_ints(lag_text)
            windows = parse_positive_ints(window_text)
            cfg = {"ohe": True, "scaler": scaler, "estimators": int(estimators),
                "max_depth": int(depth), "min_samples_leaf": int(leaf),
                "learning_rate": float(learning), "balance": False,
                "test_size": test_percent / 100, "seed": int(seed), "max_iter": 1000,
                "C": 1.0, "max_features": 20000, "ngram_range": (1, 1),
                "stop_words": None, "class_weight_ratio": 1.0}
            with st.spinner(t("training", target=target_column)):
                result = train_forecast(df, time_column=time_column, target_column=target_column,
                    frequency=frequency, aggregation=aggregation, missing_policy=missing_policy,
                    horizon=int(horizon), lags=lags, windows=windows, test_size=test_percent / 100,
                    model_name=model_name, cfg=cfg)
            experiment = {"created_at": datetime.now(timezone.utc).isoformat(),
                "dataset": {"name": st.session_state.get("source_name", "uploaded dataset"),
                            "rows": int(len(df)), "columns": [str(c) for c in df.columns]},
                "forecast_config": {"time_column": time_column, "target_column": target_column,
                    "frequency": result["frequency"], "aggregation": aggregation,
                    "missing_policy": missing_policy, "horizon": int(horizon), "lags": lags,
                    "windows": windows, "test_size": test_percent / 100,
                    "model_name": model_name, "model_config": cfg}}
            experiment["cleaning_steps"] = list(st.session_state.get("cleaning_steps", []))
            result["experiment"] = experiment
            st.session_state.forecast_result = result
            record_event(st.session_state, "forecast training",
                "fc = experiment['forecast_config']\n"
                "result = train_forecast(df, time_column=fc['time_column'], target_column=fc['target_column'],\n"
                "    frequency=fc['frequency'], aggregation=fc['aggregation'], missing_policy=fc['missing_policy'],\n"
                "    horizon=fc['horizon'], lags=fc['lags'], windows=fc['windows'],\n"
                "    test_size=fc['test_size'], model_name=fc['model_name'], cfg=fc['model_config'])",
                experiment["forecast_config"],
                f"Forecast trained; holdout metrics={result['metrics']}; baseline={result['baseline_metrics']}")
        except ValueError as exc:
            message = str(exc)
            record_event(st.session_state, "forecast training failed",
                "result = train_forecast(df, **forecast_config)",
                {"time_column": time_column, "target_column": target_column,
                 "horizon": int(horizon), "model": model_name}, message)
            if message == "TOO_LITTLE_HISTORY":
                st.error(t("forecast_history_error"))
            elif message == "FREQUENCY_REQUIRED":
                st.error(t("forecast_frequency_error"))
            elif message == "LATEST_PERIOD_HAS_GAPS":
                st.error(t("forecast_latest_gap_error"))
            else:
                st.error(t("forecast_error", error=message))
        except Exception as exc:
            record_event(st.session_state, "forecast training failed",
                "result = train_forecast(df, **forecast_config)",
                {"time_column": time_column, "target_column": target_column,
                 "horizon": int(horizon), "model": model_name}, str(exc))
            st.error(t("forecast_error", error=str(exc)))

    result = st.session_state.get("forecast_result")
    if not result:
        return
    st.divider()
    st.subheader(t("forecast_results"))
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", f"{result['metrics']['MAE']:.3f}", help=t("forecast_metric_mae"))
    metric_columns[1].metric("RMSE", f"{result['metrics']['RMSE']:.3f}", help=t("forecast_metric_rmse"))
    metric_columns[2].metric("sMAPE", f"{result['metrics']['sMAPE (%)']:.2f}%", help=t("forecast_metric_smape"))
    metric_columns[3].metric(t("forecast_baseline"), f"MAE {result['baseline_metrics']['MAE']:.3f}",
                             help=t("forecast_baseline_help"))
    if result["metrics"]["MAE"] < result["baseline_metrics"]["MAE"]:
        st.success(t("forecast_above_baseline"))
    else:
        st.warning(t("forecast_below_baseline"))
    st.markdown(f"**{t('forecast_future')}**")
    st.dataframe(result["future"], use_container_width=True, hide_index=True)
    plot = pd.concat([result["series"].rename("history"),
                      result["future"].set_index("target_time")["prediction"].rename("forecast")], axis=1)
    st.line_chart(plot)
    with st.expander(t("forecast_horizon_metrics"), expanded=False):
        st.dataframe(result["horizon_metrics"], use_container_width=True, hide_index=True)
        st.caption(t("forecast_metric_mae") + " " + t("forecast_metric_rmse") + " " + t("forecast_metric_smape"))
    with st.expander(t("forecast_holdout"), expanded=False):
        st.dataframe(result["holdout"], use_container_width=True, hide_index=True)
    st.caption(t("forecast_bundle_note"))
    safe_target = str(result["target"]).lower().replace(" ", "_")
    st.download_button(t("forecast_download"), result["future"].to_csv(index=False).encode("utf-8"),
        f"{safe_target}_forecast.csv", "text/csv", use_container_width=True,
        on_click=_record_download, args=("forecast_csv", {"target": str(result["target"]), "horizon": result["horizon"]}))
    bundle = {"model_type": "forecasting", "format": "cpm-forecast-v1",
        "models_by_lead": result["models_by_lead"], "time_column": result["time_column"],
        "target": result["target"], "frequency": result["frequency"],
        "aggregation": result["aggregation"], "missing_policy": result["missing_policy"],
        "horizon": result["horizon"], "lags": result["lags"], "windows": result["windows"],
        "model_name": result["model_name"], "config": result["config"]}
    model_buffer = io.BytesIO()
    joblib.dump(bundle, model_buffer)
    st.download_button(t("forecast_model_download"), model_buffer.getvalue(),
        f"cpm_{safe_target}_forecast.joblib", "application/octet-stream", use_container_width=True,
        on_click=_record_download, args=("forecast_model_joblib", {"model": result["model_name"], "horizon": result["horizon"]}))
    notebook = build_notebook(result["experiment"], forecast=True)
    st.caption(t("notebook_help"))
    st.download_button(t("download_notebook"), notebook, "cpm_forecasting_experiment.ipynb",
        "application/x-ipynb+json", use_container_width=True, on_click=_record_download,
        args=("forecast_jupyter_notebook", result["experiment"]["forecast_config"]))
    report = {"experiment": result["experiment"], "metrics": result["metrics"],
        "baseline_metrics": result["baseline_metrics"], "horizon_metrics": result["horizon_metrics"].to_dict(orient="records"),
        "cleaning_steps": st.session_state.get("cleaning_steps", [])}
    st.download_button(t("download_report"), json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        "cpm_forecast_report.json", "application/json", use_container_width=True,
        on_click=_record_download, args=("forecast_report_json", result["experiment"]["forecast_config"]))
def run() -> None:
    st.set_page_config(page_title="CPM Model Builder", page_icon="◈", layout="wide")
    with st.sidebar:
        st.markdown("## ◈ CPM Model Builder")
        language_label = st.selectbox("Language / Idioma", ["English", "Español"], index=0)
        language = "en" if language_label == "English" else "es"
        t = translator(language)
        st.caption(t("sidebar_tagline"))
        st.divider()
        st.markdown(f'#### {t("workflow")}')
        st.markdown(t("workflow_steps_forecast") if st.session_state.get("task_type") == "Forecasting"
                    else t("workflow_steps"))
        st.caption(t("local_processing"))
    _theme()
    if "data" not in st.session_state:
        active_step = 1
    elif st.session_state.get("results") or st.session_state.get("forecast_result"):
        active_step = 4
    elif st.session_state.get("target_cols") and (
            st.session_state.get("feature_cols") or st.session_state.get("tfidf_columns")):
        active_step = 3
    else:
        active_step = 2
    _header(t, active_step)
    rail_left, rail_right = st.columns([5, 1])
    with rail_right:
        with st.popover("⌘  " + t("developer_log"), use_container_width=True):
            st.caption(t("developer_log_help"))
            events = st.session_state.get("developer_events", [])
            if st.button(t("clear_log"), key="clear_developer_log", disabled=not events):
                st.session_state.developer_events = []
                st.rerun()
            if not events:
                st.info(t("empty_log"))
            for event in reversed(events):
                with st.expander(f"In [{event['number']}] · {event['stage']} · {event['timestamp']}", expanded=False):
                    st.code(event["code"], language="python")
                    st.caption(t("log_parameters"))
                    st.json(event["parameters"])
                    st.caption(t("log_output"))
                    st.write(event["output"])
    _data_input(t)
    if "data" not in st.session_state:
        st.info(t("no_dataset"))
    else:
        df = _cleaning(st.session_state.data, t)
        st.markdown(f'<div class="eyebrow">{t("step_prepare")}</div>', unsafe_allow_html=True)
        st.subheader(t("preview_title"))
        st.caption(t("preview_desc"))
        _profile(df, t)
        st.success(f'**{t("ready")}.** {t("ready_msg")}')
        _training(df, t)
    _model_upload(t)
    st.markdown(
        f'<div class="footer">{t("developer")} &nbsp;·&nbsp; {t("footer")}<br>'
        f'<a href="https://www.linkedin.com/in/nahuelsoliz" target="_blank" rel="noopener noreferrer">{t("linkedin")}</a></div>',
        unsafe_allow_html=True)
