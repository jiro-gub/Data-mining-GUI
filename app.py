import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

st.set_page_config(page_title="SVM Classifier", layout="wide")
st.title("🧠 Support Vector Machine Classification GUI")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df, height=400)

    total_rows = df.shape[0]
    total_columns = df.shape[1]
    numeric_columns = df.select_dtypes(include=np.number).shape[1]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Total Rows")
        st.markdown(f"<h2 style='text-align: center;'>{total_rows}</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown("#### Total Columns")
        st.markdown(f"<h2 style='text-align: center;'>{total_columns}</h2>", unsafe_allow_html=True)
    with col3:
        st.markdown("#### Numeric Columns")
        st.markdown(f"<h2 style='text-align: center;'>{numeric_columns}</h2>", unsafe_allow_html=True)

    target = st.selectbox("🎯 Select the target column", df.columns)
    features = st.multiselect(
        "🧬 Select feature columns",
        [col for col in df.columns if col != target],
        default=[col for col in df.select_dtypes(include=np.number).columns if col != target]
    )

    kernel = st.sidebar.selectbox("Select SVM Kernel", ["linear", "rbf", "poly"], index=0)
    scale_features = st.sidebar.checkbox("🔄 Scale Features", value=True)

    if st.button("🚀 Run SVM"):
        if not target or len(features) == 0:
            st.error("Please select the target and at least one feature column.")
        else:
            X = df[features]
            y = df[target]

            if y.dtype == 'object' or y.dtype.name == 'category':
                y, label_mapping = pd.factorize(y)
                st.markdown("🔍 **Label Mapping:**")
                st.json(dict(enumerate(label_mapping)))

            num_classes = np.unique(y).size

            class_dist = pd.Series(y).value_counts(normalize=True)
            if class_dist.min() < 0.2:
                st.warning("⚠️ Warning: Potential class imbalance detected.")

            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = X.select_dtypes(include=np.number).columns.tolist()

            preprocessor = ColumnTransformer([
                ("num", StandardScaler() if scale_features else "passthrough", numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', SVC(
                    kernel=kernel,
                    probability=True,
                    random_state=42
                ))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = None
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                pass

            accuracy = accuracy_score(y_test, y_pred)
            average_type = 'binary' if num_classes == 2 else 'weighted'
            precision = precision_score(y_test, y_pred, average=average_type, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average_type, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average_type, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            specificity = None
            if num_classes == 2 and cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            auc = None
            if y_proba is not None:
                try:
                    if num_classes == 2:
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except Exception:
                    pass

            st.subheader("📊 Classification Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔍 Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='magma', cbar=True, ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

            with col2:
                if num_classes == 2 and y_proba is not None:
                    st.markdown("### 📉 ROC Curve")
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
                    ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})", color='orange')
                    ax_roc.plot([0, 1], [0, 1], 'b--')
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.set_title("ROC Curve")
                    ax_roc.legend(loc="lower right")
                    st.pyplot(fig_roc)

            st.markdown("### 📝 Classification Report & Metrics")
            report = classification_report(y_test, y_pred, digits=2)
            st.text(report)

            st.markdown("**Additional Metrics:**")
            if specificity is not None:
                st.markdown(f"- **Specificity:** `{specificity:.4f}`")
            if auc is not None:
                st.markdown(f"- **AUC:** `{auc:.4f}`")

            # --- Decision Boundary / Hyperplane Plot ---
            if len(numerical_features) == 2 and len(categorical_features) == 0:
                st.subheader("🧭 SVM Decision Boundary Visualization")

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X[numerical_features])
                X_train_scaled, _, y_train_scaled, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

                clf = SVC(kernel=kernel, probability=True, random_state=42)
                clf.fit(X_train_scaled, y_train_scaled)

                h = 0.02
                x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
                y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')

                for class_value in np.unique(y_train_scaled):
                    idx = np.where(y_train_scaled == class_value)
                    ax.scatter(X_train_scaled[idx, 0], X_train_scaled[idx, 1],
                               label=f"Class {class_value}",
                               edgecolor='k', s=100, alpha=0.8,
                               c=('lightgreen' if class_value == 0 else 'salmon'))

                ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                           facecolors='none', edgecolors='black', linewidths=1.5,
                           s=200, label='Support Vectors')

                if kernel == 'linear':
                    w = clf.coef_[0]
                    b = clf.intercept_[0]
                    margin = 2 / np.linalg.norm(w)

                    xx_vals = np.linspace(x_min, x_max, 100)
                    yy_vals = -(w[0] * xx_vals + b) / w[1]
                    margin_up = yy_vals + margin / 2
                    margin_down = yy_vals - margin / 2

                    ax.plot(xx_vals, yy_vals, 'k-', label='Decision Boundary')
                    ax.plot(xx_vals, margin_up, 'k--')
                    ax.plot(xx_vals, margin_down, 'k--')

                    ax.text(x_min, y_max, f"Margin Width: {margin:.3f}", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

                ax.set_xlabel(f"Scaled {numerical_features[0]}")
                ax.set_ylabel(f"Scaled {numerical_features[1]}")
                ax.set_title(f"SVM Decision Boundary\nKernel: {kernel}\nFeatures: {numerical_features[0]} vs {numerical_features[1]}\nTarget: {target}")
                ax.legend(loc="best")

                st.pyplot(fig)

            else:
                st.info("ℹ️ Decision boundary visualization requires exactly 2 numerical features and no categorical features.")

else:
    st.info("📥 Please upload a CSV or Excel file to begin.")
