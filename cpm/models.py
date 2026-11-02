"""Preprocessing, model training, metrics, and inference helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, RobustScaler,
                                   StandardScaler, LabelEncoder)


TEXT_FEATURE = "__cpm_text_for_tfidf__"
SPANISH_STOP_WORDS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante", "e",
    "el", "ella", "ellas", "ellos", "en", "entre", "era", "erais", "eran",
    "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estabais", "estaban", "estabas", "estad", "estada", "estadas",
    "estado", "estados", "estamos", "estan", "estando", "estar", "estaremos",
    "estara", "estaran", "estaras", "estare", "estareis", "estaria", "estariais",
    "estariamos", "estarian", "estarias", "estas", "este", "estemos", "esto",
    "estos", "estoy", "estuve", "estuviera", "estuvierais", "estuvieran",
    "estuvieras", "estuvieron", "estuviese", "estuvieseis", "estuviesen",
    "estuvieses", "estuvimos", "estuviste", "estuvisteis", "estuviéramos",
    "estuviésemos", "estuvo", "está", "estábamos", "estáis", "están", "estás",
    "esté", "estéis", "estén", "estés", "fue", "fuera", "fuerais", "fueran",
    "fueras", "fueron", "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos",
    "fuiste", "fuisteis", "fuéramos", "fuésemos", "ha", "habida", "habidas",
    "habido", "habidos", "habiendo", "habremos", "habra", "habran", "habras",
    "habre", "habreis", "habria", "habriais", "habriamos", "habrian", "habrias",
    "habéis", "había", "habíais", "habíamos", "habían", "habías", "han", "has",
    "hasta", "hay", "haya", "hayamos", "hayan", "hayas", "he", "hemos", "hube",
    "hubiera", "hubierais", "hubieran", "hubieras", "hubieron", "hubiese",
    "hubieseis", "hubiesen", "hubieses", "hubimos", "hubiste", "hubisteis",
    "hubiéramos", "hubiésemos", "hubo", "la", "las", "le", "les", "lo", "los",
    "mas", "me", "mi", "mis", "mucho", "muchos", "muy", "más", "mí", "nada",
    "ni", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
    "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para", "pero",
    "poco", "por", "porque", "que", "quien", "quienes", "qué", "se", "sea",
    "seamos", "sean", "seas", "ser", "seremos", "sera", "seran", "seras", "sere",
    "sereis", "seria", "seriais", "seriamos", "serian", "serias", "seáis",
    "sentid", "sentida", "sentidas", "sentido", "sentidos", "será", "serán",
    "serás", "seré", "seréis", "sería", "seríais", "seríamos", "serían",
    "serías", "si", "sido", "siendo", "sin", "sobre", "sois", "somos", "son",
    "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "sí", "tambien",
    "también", "tanto", "te", "tendremos", "tendra", "tendran", "tendras",
    "tendre", "tendreis", "tendria", "tendriais", "tendriamos", "tendrian",
    "tendrias", "tened", "tenemos", "tenga", "tengamos", "tengan", "tengas",
    "tengo", "tenida", "tenidas", "tenido", "tenidos", "teniendo", "tenéis",
    "tenía", "teníais", "teníamos", "tenían", "tenías", "ti", "tiene", "tienen",
    "tienes", "todo", "todos", "tu", "tus", "tuve", "tuviera", "tuvierais",
    "tuvieran", "tuvieras", "tuvieron", "tuviese", "tuvieseis", "tuviesen",
    "tuvieses", "tuvimos", "tuviste", "tuvisteis", "tuviéramos", "tuviésemos",
    "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "un", "una", "uno", "unos",
    "unas", "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
    "y", "ya", "yo",
]


class FlattenText(BaseEstimator, TransformerMixin):
    """Convert the single-column frame from ColumnTransformer to a 1D text array."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X).reshape(-1).astype(str)


class Densify(BaseEstimator, TransformerMixin):
    """Convert sparse features only for estimators that require dense input."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X


def available_models(task: str, use_tfidf: bool = False) -> list[str]:
    if use_tfidf:
        return ["Logistic Regression"] if task == "Classification" else ["Linear Regression"]
    return (["Logistic Regression", "Random Forest", "Gradient Boosting", "CatBoost", "XGBoost", "LightGBM"]
            if task == "Classification" else
            ["Linear Regression", "Random Forest", "Gradient Boosting", "CatBoost", "XGBoost", "LightGBM"])


def prepare_features(frame: pd.DataFrame, text_columns: list[str] | None = None) -> pd.DataFrame:
    prepared = frame.copy()
    if text_columns:
        prepared[TEXT_FEATURE] = prepared[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    return prepared


def _estimator(name: str, task: str, cfg: dict[str, Any]):
    classification = task == "Classification"
    if name == "Logistic Regression":
        return LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"],
            class_weight="balanced" if cfg.get("balance") else None)
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        kwargs = {"n_estimators": cfg["estimators"], "max_depth": cfg["max_depth"] or None,
            "min_samples_leaf": cfg["min_samples_leaf"], "random_state": cfg["seed"], "n_jobs": -1}
        if classification:
            kwargs["class_weight"] = "balanced" if cfg.get("balance") else None
        return cls(**kwargs)
    if name == "Gradient Boosting":
        cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        return cls(n_estimators=cfg["estimators"], learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"] or 3, random_state=cfg["seed"])
    package_error = {
        "CatBoost": ("catboost", "catboost"),
        "XGBoost": ("xgboost", "xgboost"),
        "LightGBM": ("lightgbm", "lightgbm"),
    }
    try:
        if name == "CatBoost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            cls = CatBoostClassifier if classification else CatBoostRegressor
            kwargs = {"iterations": cfg["estimators"], "learning_rate": cfg["learning_rate"],
                "depth": cfg["max_depth"] or 6, "verbose": False,
                "random_seed": cfg["seed"], "allow_writing_files": False}
            if classification:
                kwargs["auto_class_weights"] = "Balanced" if cfg.get("balance") else None
            return cls(**kwargs)
        if name == "XGBoost":
            from xgboost import XGBClassifier, XGBRegressor
            cls = XGBClassifier if classification else XGBRegressor
            return cls(n_estimators=cfg["estimators"], learning_rate=cfg["learning_rate"],
                max_depth=cfg["max_depth"] or 6, random_state=cfg["seed"], n_jobs=-1,
                eval_metric="logloss" if classification else "rmse",
                **({"scale_pos_weight": cfg.get("class_weight_ratio", 1) if cfg.get("balance") else 1}
                   if classification else {}))
        if name == "LightGBM":
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMClassifier if classification else LGBMRegressor
            kwargs = {"n_estimators": cfg["estimators"], "learning_rate": cfg["learning_rate"],
                "max_depth": cfg["max_depth"] or -1, "random_state": cfg["seed"],
                "verbosity": -1, "n_jobs": -1}
            if classification:
                kwargs["class_weight"] = "balanced" if cfg.get("balance") else None
            return cls(**kwargs)
    except ImportError as exc:
        package = package_error.get(name, (name, name))[1]
        raise ImportError(f"OPTIONAL_MODEL:{name}:{package}") from exc
    raise ValueError(f"Unsupported model: {name}")


def build_pipeline(X: pd.DataFrame, model_name: str, task: str,
                   cfg: dict[str, Any], text_columns: list[str] | None = None) -> Pipeline:
    text_columns = text_columns or []
    categorical = list(X.select_dtypes(include=["object", "category", "string", "bool"]).columns)
    categorical = [c for c in categorical if c not in text_columns and c != TEXT_FEATURE]
    numeric = [c for c in X.columns if c not in categorical and c not in text_columns and c != TEXT_FEATURE]
    scaler = {"StandardScaler": StandardScaler, "RobustScaler": RobustScaler}.get(cfg["scaler"])
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler:
        numeric_steps.append(("scaler", scaler()))
    encoder = (OneHotEncoder(handle_unknown="ignore") if cfg["ohe"] else
               OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    categorical_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]
    transformers: list[tuple[str, Any, Any]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical))
    if text_columns:
        stop = cfg.get("stop_words") or None
        if stop == "spanish":
            stop = SPANISH_STOP_WORDS
        text_pipe = Pipeline([
            ("flatten", FlattenText()),
            ("tfidf", TfidfVectorizer(max_features=cfg["max_features"],
                ngram_range=cfg["ngram_range"], stop_words=stop, sublinear_tf=True,
                strip_accents="unicode")),
        ])
        transformers.append(("text", text_pipe, [TEXT_FEATURE]))
    if not transformers:
        raise ValueError("Select at least one feature.")
    preprocessing = ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)
    steps = [("preprocess", preprocessing)]
    if model_name == "Gradient Boosting":
        steps.append(("dense", Densify()))
    steps.append(("model", _estimator(model_name, task, cfg)))
    return Pipeline(steps)


def _classification_predictions(probabilities: np.ndarray, classes: list[Any],
                                threshold: float = 0.5) -> np.ndarray:
    """Use a configurable positive-class threshold for binary predictions."""
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        negative, positive = classes[0], classes[-1]
        return np.where(probabilities[:, 1] >= threshold, positive, negative)
    return np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]


def _classification_metrics(y_true, predictions, probabilities=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision (weighted)": float(precision_score(y_true, predictions, average="weighted", zero_division=0)),
        "Recall (weighted)": float(recall_score(y_true, predictions, average="weighted", zero_division=0)),
        "F1 (weighted)": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "ROC AUC": None,
    }
    if probabilities is not None:
        try:
            metrics["ROC AUC"] = float(roc_auc_score(
                y_true, probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities,
                multi_class="ovr", average="weighted"))
        except (AttributeError, ValueError):
            pass
    return metrics


def set_classification_threshold(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Update binary predictions and threshold-sensitive metrics in place."""
    probabilities = result.get("probabilities")
    classes = result.get("class_labels", [])
    if probabilities is None or len(classes) != 2:
        return result
    result["threshold"] = float(threshold)
    result["predictions"] = _classification_predictions(probabilities, classes, threshold)
    result["metrics"] = _classification_metrics(
        result["y_test"], result["predictions"], probabilities)
    return result


def feature_impact(result: dict[str, Any], top_n: int = 20) -> pd.DataFrame:
    """Return model-native importance or coefficient magnitudes with feature labels."""
    pipeline = result["pipeline"]
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        columns = [str(c) for c in (columns if isinstance(columns, (list, tuple, np.ndarray, pd.Index)) else [columns])]
        try:
            if name == "text":
                transform_names = transformer.named_steps["tfidf"].get_feature_names_out()
            elif isinstance(transformer, Pipeline):
                last_step = transformer.steps[-1][1]
                transform_names = last_step.get_feature_names_out(columns) if hasattr(last_step, "get_feature_names_out") else columns
            else:
                transform_names = transformer.get_feature_names_out(columns) if hasattr(transformer, "get_feature_names_out") else columns
            names.extend(str(item) for item in transform_names)
        except (AttributeError, ValueError):
            names.extend(columns)

    direction = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim > 1:
            coefficients = coefficients.mean(axis=0)
        direction = np.sign(coefficients)
        values = np.abs(coefficients)
    else:
        return pd.DataFrame(columns=["Feature", "Importance", "Direction"])
    if len(names) != len(values):
        names = [f"feature_{index + 1}" for index in range(len(values))]
    frame = pd.DataFrame({"Feature": names, "Importance": values})
    if direction is not None:
        frame["Direction"] = np.where(direction > 0, "Positive", np.where(direction < 0, "Negative", "Neutral"))
    else:
        frame["Direction"] = "—"
    return frame.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def train_model(df: pd.DataFrame, features: list[str], target: str, model_name: str,
                task: str, cfg: dict[str, Any], text_columns: list[str] | None = None) -> dict[str, Any]:
    text_columns = text_columns or []
    selected = list(dict.fromkeys(features + text_columns))
    data = df[selected + [target]].dropna(subset=[target])
    X = prepare_features(data[selected], text_columns)
    y = data[target]
    if len(data) < 8:
        raise ValueError("SMALL_DATA")
    if task == "Classification" and y.nunique() < 2:
        raise ValueError("SINGLE_CLASS")
    if task == "Classification" and y.nunique() > 100:
        raise ValueError("BAD_TARGET")
    label_encoder = None
    if task == "Classification" and model_name == "XGBoost":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target)
    stratify = y if task == "Classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["test_size"],
        random_state=cfg["seed"], stratify=stratify)
    model_cfg = dict(cfg)
    if label_encoder is not None and model_cfg.get("balance") and y_train.nunique() == 2:
        class_counts = y_train.value_counts()
        model_cfg["class_weight_ratio"] = float(class_counts.get(0, 1) / class_counts.get(1, 1))
    pipeline = build_pipeline(X_train, model_name, task, model_cfg, text_columns)
    pipeline.fit(X_train, y_train)
    probabilities = None
    class_labels: list[Any] = []
    if task == "Classification":
        model_classes = np.asarray(pipeline.named_steps["model"].classes_)
        class_labels = label_encoder.inverse_transform(model_classes.astype(int)).tolist() if label_encoder else model_classes.tolist()
        try:
            probabilities = pipeline.predict_proba(X_test)
            predictions = _classification_predictions(probabilities, class_labels)
        except AttributeError:
            predictions = pipeline.predict(X_test)
    else:
        predictions = pipeline.predict(X_test)
    if label_encoder is not None:
        y_test = pd.Series(label_encoder.inverse_transform(y_test.to_numpy()), index=y_test.index, name=target)
    if task == "Classification":
        metrics = _classification_metrics(y_test, predictions, probabilities)
    else:
        metrics = {"MAE (EAM)": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R²": float(r2_score(y_test, predictions))}
    return {"pipeline": pipeline, "metrics": metrics, "y_test": y_test,
        "predictions": predictions, "feature_names": selected, "target": target,
        "task": task, "model_name": model_name, "text_columns": text_columns,
        "label_encoder": label_encoder,
        "probabilities": probabilities, "class_labels": class_labels,
        "threshold": 0.5 if task == "Classification" else None,
        "train_rows": len(X_train), "test_rows": len(X_test), "tfidf_config": {
            "max_features": cfg["max_features"], "ngram_range": cfg["ngram_range"],
            "stop_words": cfg.get("stop_words")}}


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    required = bundle["feature_names"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError("MISSING_FEATURES:" + ",".join(map(str, missing)))
    X = prepare_features(frame[required], bundle.get("text_columns", []))
    encoder = bundle.get("label_encoder")
    if bundle.get("task") == "Classification" and bundle.get("threshold") is not None:
        try:
            probs = bundle["pipeline"].predict_proba(X)
            model_classes = np.asarray(bundle["pipeline"].named_steps["model"].classes_)
            original_classes = encoder.inverse_transform(model_classes.astype(int)) if encoder else model_classes
            predictions = _classification_predictions(probs, original_classes.tolist(), bundle["threshold"])
            return predictions
        except AttributeError:
            pass
    predictions = bundle["pipeline"].predict(X)
    return encoder.inverse_transform(np.asarray(predictions, dtype=int)) if encoder else predictions


def quality_feedback(result: dict[str, Any]) -> tuple[str, str]:
    m = result["metrics"]
    if result["task"] == "Classification":
        score = m["F1 (weighted)"]
        if score >= .8:
            return "quality_strong", "classification_good"
        if score >= .6:
            return "quality_promising", "classification_mid"
        return "quality_low", "classification_low"
    score = m["R²"]
    if score >= .75:
        return "quality_strong", "regression_good"
    if score >= .4:
        return "quality_promising", "regression_mid"
    return "quality_low", "regression_low"
