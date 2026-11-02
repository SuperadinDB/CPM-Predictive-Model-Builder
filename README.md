# CPM Model Builder

**A bilingual, local-first workspace for preparing datasets, training machine-learning models, and understanding their results.**

## Start the app

1. Install Python 3.12 or newer.
2. Open this folder in VS Code.
3. In the integrated terminal, create and activate a virtual environment:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

4. Install the core requirements:

   python -m pip install -r requirements.txt

5. Start the app:

   python -m streamlit run app.py

You can also run run_app.bat from Windows Explorer.

## What you can do

- Work in English or Spanish. The language selector is in the sidebar.
- Upload CSV, TSV, and XLSX files, or browse the included sample library. The app previews a selected sample before loading the full file.
- Add your own supported data files to the root-level samples folder. The folder is included in the Windows package.
- Review row and column counts, missing data, duplicates, data types, statistics, and sample rows.
- Standardize column names and text casing, remove columns and duplicates, or add an arithmetic calculated column.
- Train classification or regression models for one or more target columns.
- Forecast one regularly spaced numeric time series, including a simple two-column date/value dataset. Select date frequency, duplicate-date aggregation, missing-period policy, lag values, rolling windows, prediction horizon, and regression algorithm. Evaluation follows time order and compares against a last-observed-value benchmark.
- Compare available algorithms on the same holdout rows, features, preprocessing, and random seed.
- Review data-quality and potential leakage signals, including duplicate rows, high missingness, constant fields, identifier-like columns, target copies, and class imbalance.
- Explore tree-based feature importance or linear-model coefficient direction, and adjust the decision threshold for binary classification.
- Export an experiment report with the data reference, selected features, settings, evaluation scores, data checks, and package versions.
- Select input features, one-hot encoding or ordinal category encoding, numeric scaling, class balancing where supported, and a holdout split.
- Use Logistic Regression, Linear Regression, Random Forest, Gradient Boosting, CatBoost, XGBoost, or LightGBM.
- Turn on TF-IDF to train text models from reviews, comments, tickets, or other text columns. TF-IDF is available with Logistic Regression for classification and Linear Regression for regression. Configure vocabulary size, one- or two-word phrases, and common English or Spanish stop words.
- For NLP, review the in-app guidance on label quality, duplicate text leakage, vocabulary tuning, class-level errors, and the limits of bag-of-words models. Transformer language models are not included.
- Review classification scores, a class report and confusion matrix, or regression errors and an actual-versus-predicted plot.
- Download evaluation predictions and a reusable .joblib model bundle. Load that bundle later to predict on a matching dataset.
- Download a Jupyter `.ipynb` notebook that replays the selected cleaning operations and settings through the project's modular Python code. Keep the notebook beside the project, install the requirements, and update `DATA_PATH` to the original CSV, TSV, or Excel file; the private dataset is not embedded.
- Open the developer execution log from the upper-right menu to inspect session-only `In[n]` records of executed data, cleaning, training, forecast, prediction, and export actions, including their parameters and outcomes. It is a trace viewer, not a live Python kernel.

The cleaning controls create a working copy inside the app session. They do not edit your original source file.

## Optional gradient-boosting libraries

The core app uses scikit-learn. CatBoost, XGBoost, and LightGBM are additional packages. Install them if you want to use those algorithms:

    python -m pip install -r requirements-models.txt

## Windows installer

Install Python 3.12 and Inno Setup 6 or 7, then run `build_windows.bat`. The script creates a separate `.build-venv`, installs the core and optional model requirements plus PyInstaller, packages the app, and creates:

    installer\CPM_Model_Builder_Setup.exe

The installed program runs a local Streamlit service and opens CPM Model Builder in your default browser. The app does not upload your data.

## Project structure

- app.py — development entry point.
- launch.py — packaged application launcher.
- cpm/ui.py — guided interface and result displays.
- cpm/i18n.py — English and Spanish interface text.
- cpm/data.py — file loading, example data, profiling, and cleaning.
- cpm/models.py — preprocessing, model training, metrics, TF-IDF, and prediction.
- cpm/forecasting.py — chronological, lag-based forecasting and prediction bundles.
- cpm/notebook_export.py — reproducible Jupyter notebook generation.
- cpm/trace.py — session-only developer execution log.
- requirements*.txt — core and optional dependencies.
- CPM_Model_Builder.spec and CPM_Model_Builder.iss — Windows packaging files.

## Model file safety

Only load .joblib model bundles from sources you trust. Joblib uses Python pickle serialization, which can execute code while loading a file. Evaluation scores are useful baselines; validate models with fresh, representative data before using predictions in a real process.
