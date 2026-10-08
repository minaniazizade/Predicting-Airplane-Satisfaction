import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix


st.set_page_config(page_title="Predicting Airline Satisfaction", page_icon="✈️", layout="wide")

st.title("✈️ Predicting Airline Satisfaction")
st.caption("AI + Business")

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def prepare_features(train_df, test_df=None):
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None

    target = "satisfaction"
    id_col = "id"

    if target not in train_df.columns:
        raise ValueError("error!")

    y = train_df[target].copy()
    X = train_df.drop(columns=[target, id_col], errors="ignore")
    X_test = None
    test_ids = None

    if test_df is not None:
        test_ids = test_df[id_col].copy() if id_col in test_df.columns else pd.Series(np.arange(len(test_df)), name="id")
        X_test = test_df.drop(columns=[id_col], errors="ignore")

    for frame in [X, X_test]:
        if frame is None:
            continue
        for col in frame.columns:
            if col not in X.columns:
                frame.drop(columns=[col], inplace=True)
        if "Customer Type" in frame.columns:
            frame["Customer Type"] = frame["Customer Type"].astype("string").str.replace("Customer", "", regex=False).astype(object)
        if "Type of Travel" in frame.columns:
            frame["Type of Travel"] = frame["Type of Travel"].astype("string").str.replace("Travel", "", regex=False).astype(object)

    if X_test is not None:
        missing_cols = [c for c in X.columns if c not in X_test.columns]
        for c in missing_cols:
            X_test[c] = np.nan
        X_test = X_test[X.columns]

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for frame in [X, X_test]:
        if frame is None:
            continue
        for col in cat_cols:
            frame[col] = frame[col].astype("string").fillna("Unknown").astype(str)
        for col in frame.columns:
            if col not in cat_cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return X, y, X_test, test_ids, cat_cols

def fill_missing(X_train, X_other=None):
    X_train = X_train.copy()
    X_other = X_other.copy() if X_other is not None else None
    medians = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            med = X_train[col].median()
            if pd.isna(med):
                med = 0
            medians[col] = med
            X_train[col] = X_train[col].fillna(med)
            if X_other is not None and col in X_other.columns:
                X_other[col] = X_other[col].fillna(med)
        else:
            X_train[col] = X_train[col].fillna("Unknown")
            if X_other is not None and col in X_other.columns:
                X_other[col] = X_other[col].fillna("Unknown")
    return X_train, X_other

with st.sidebar:
    st.header("Model Settings")
    iterations = st.slider("Iterations", 200, 2000, 1000, 100)
    learning_rate = st.slider("Learning rate", 0.01, 0.20, 0.03, 0.01)
    test_size = st.slider("val_test", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=99999, value=2)
    use_gpu = st.checkbox("GPU", value=False)

tab1, tab2, tab3 = st.tabs(["📁Dataset", "🧠Learn and Validation", "🔮test_pred"])

with tab1:
    st.subheader("Upload")
    c1, c2 = st.columns(2)
    with c1:
        train_file = st.file_uploader("train.csv", type=["csv"], key="train")
    with c2:
        test_file = st.file_uploader("test.csv", type=["csv"], key="test")

    if train_file is not None:
        try:
            train_df = load_csv(train_file)
            st.success(f"train: {train_df.shape[0]:,} rows and {train_df.shape[1]} columns")
            st.dataframe(train_df.head(20), use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("rows", f"{len(train_df):,}")
            b.metric("columns", len(train_df.columns))
            c.metric("Missing values", f"{int(train_df.isna().sum().sum()):,}")
            with st.expander("Summary"):
                st.dataframe(pd.DataFrame({
                    "Data type": train_df.dtypes.astype(str),
                    "Missing Values": train_df.isna().sum(),
                    "uniques": train_df.nunique(dropna=False)
                }), use_container_width=True)
            if "satisfaction" in train_df.columns:
                st.write("target")
                counts = train_df["satisfaction"].astype(str).value_counts().rename_axis("Class").reset_index(name="num")
                st.bar_chart(counts.set_index("Class"))
        except Exception as e:
            st.error(f"error: {e}")
    else:
        st.info("Optional")

with tab2:
    st.subheader("Learn Model and Validation")
    if train_file is None:
        st.info("Upload")
    else:
        if st.button("Model", type="primary", key="train_model"):
            try:
                train_df = load_csv(train_file)
                X, y, _, _, cat_cols = prepare_features(train_df)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=float(test_size), random_state=int(random_state),
                    stratify=y if y.nunique() > 1 else None
                )
                X_train, X_val = fill_missing(X_train, X_val)
                params = dict(
                    iterations=int(iterations),
                    learning_rate=float(learning_rate),
                    loss_function="Logloss" if y.nunique() == 2 else "MultiClass",
                    random_seed=int(random_state),
                    verbose=False,
                    cat_features=cat_cols,
                    allow_writing_files=False
                )
                if use_gpu:
                    params["task_type"] = "GPU"
                model = CatBoostClassifier(**params)
                with st.spinner("learning..."):
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
                preds = model.predict(X_val).ravel()
                probs = model.predict_proba(X_val)
                st.session_state["trained_model"] = model
                st.session_state["feature_columns"] = X.columns.tolist()
                st.session_state["cat_cols"] = cat_cols
                st.session_state["medians_source"] = X_train
                st.session_state["classes"] = list(model.classes_)
                st.session_state["model_params"] = params
                st.success("Success")
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy", f"{accuracy_score(y_val, preds):.4f}")
                m2.metric("F1 (weighted)", f"{f1_score(y_val, preds, average='weighted'):.4f}")
                try:
                    if y.nunique() == 2:
                        positive_label = model.classes_[1]
                        y_binary = (y_val == positive_label).astype(int)
                        auc = roc_auc_score(y_binary, probs[:, 1])
                        m3.metric("ROC-AUC", f"{auc:.4f}")
                    else:
                        m3.metric("ROC-AUC", "MultiClass")
                except Exception:
                    m3.metric("ROC-AUC", "uncountable")
                st.write("Confusion Matrix")
                labels = list(model.classes_)
                cm = confusion_matrix(y_val, preds, labels=labels)
                importance = pd.DataFrame({"feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(20)
                st.write("20")
                st.bar_chart(importance.set_index("Feature"))
                model_bytes = model.save_model
                st.info("...")
            except Exception as e:
                st.error(f"failed: {e}")
                st.exception(e)

