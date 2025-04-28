import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype

# App Configuration
st.set_page_config(page_title="Advanced CSV Data Analyzer", layout="wide")
st.title("📊 Advanced CSV Data Analyzer")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

if uploaded_file:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
    df = st.session_state.df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # Tabs
    tabs = st.tabs(["🔍 Overview", "🧼 Clean Data", "📈 Analysis", "📊 Visuals", "🤖 PCA/ML"])

    # --- Overview Tab ---
    with tabs[0]:
        st.subheader("🔍 Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)

        with st.expander("🧾 Data Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Rows:**", df.shape[0])
                st.write("**Columns:**", df.shape[1])
                st.write("**Column Names:**", list(df.columns))
            with col2:
                st.write("**Data Types:**")
                st.dataframe(df.dtypes.reset_index().rename(
                    columns={"index": "Column", 0: "Type"}), height=300)

        with st.expander("📈 Descriptive Statistics"):
            selected = st.multiselect("Select columns for statistics", all_cols, key="desc_cols")
            if selected:
                st.dataframe(df[selected].describe(include='all'), height=400)

        with st.expander("⚠️ Missing Values"):
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                st.dataframe(missing)
            else:
                st.success("No missing values detected!")

        st.download_button("⬇️ Download Current Dataset", df.to_csv(index=False),
                           "cleaned_data.csv", "text/csv")

    # --- Clean Data Tab ---
    with tabs[1]:
        st.subheader("🧼 Clean & Transform Data")

        with st.expander("📍 Drop Columns"):
            cols_to_drop = st.multiselect("Select columns to drop", all_cols, key="drop_cols")
            if st.button("Drop Selected Columns"):
                if cols_to_drop:
                    df.drop(columns=cols_to_drop, inplace=True)
                    st.success(f"Dropped columns: {cols_to_drop}")
                    st.session_state.df = df

        with st.expander("💡 Fill Missing Values"):
            col = st.selectbox("Select a column", all_cols, key="fill_col")
            method = st.selectbox("Fill method", ["Mean", "Median", "Mode"], key="fill_method")
            if st.button("Fill Missing"):
                if method == "Mean" and is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median" and is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    if not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        st.warning(f"No mode available to fill {col}.")
                st.success(f"Filled missing values in {col} using {method}.")
                st.session_state.df = df

        with st.expander("🔢 Convert Data Type"):
            col = st.selectbox("Column to convert", all_cols, key="convert_col")
            dtype = st.selectbox("New type", ["int", "float", "str"], key="convert_type")
            if st.button("Convert Type"):
                try:
                    df[col] = df[col].astype(dtype)
                    st.success(f"Converted {col} to {dtype}")
                    st.session_state.df = df
                except Exception as e:
                    st.error(f"Error: {e}")

    # --- Analysis Tab ---
    with tabs[2]:
        st.subheader("📈 Exploratory Data Analysis")

        with st.expander("📉 Value Counts"):
            col = st.selectbox("Select column", all_cols, key="value_counts_col")
            st.write(df[col].value_counts())

        with st.expander("📊 Correlation Matrix"):
            if numeric_cols:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No numeric columns to compute correlation.")

    # --- Visuals Tab ---
    with tabs[3]:
        st.subheader("📊 Visualization")

        with st.expander("📍 Plot Settings"):
            plot_type = st.selectbox("Select Plot Type", ["Histogram", "Box Plot", "Scatter Plot"], key="plot_type")
            
            if plot_type == "Histogram":
                col = st.selectbox("Column", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 5, 100, 20, key="hist_bins")
                fig = px.histogram(df, x=col, nbins=bins)
                st.plotly_chart(fig)

            elif plot_type == "Box Plot":
                col = st.selectbox("Column", numeric_cols, key="box_col")
                fig = px.box(df, y=col)
                st.plotly_chart(fig)

            elif plot_type == "Scatter Plot":
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                fig = px.scatter(df, x=x_col, y=y_col)
                st.plotly_chart(fig)

    # --- PCA & ML Tab ---
    with tabs[4]:
        st.subheader("🤖 PCA - Dimensionality Reduction")

        if len(numeric_cols) >= 2:
            scale = st.checkbox("🔵 Standardize Data (Recommended)", value=True, key="pca_scale")
            if scale:
                st.caption("Standardization centers and scales numeric columns for better PCA performance.")

            n_components = st.slider("Number of Components", 2, min(len(numeric_cols), 10), 2, key="pca_ncomp")

            data = df[numeric_cols].dropna()
            if scale:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
            else:
                data_scaled = data

            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(data_scaled)
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])

            st.write("Explained Variance Ratio:")
            st.bar_chart(pca.explained_variance_ratio_)

            st.subheader("PCA Scatter Plot")
            if n_components >= 2:
                fig = px.scatter(pca_df, x='PC1', y='PC2')
                st.plotly_chart(fig)
        else:
            st.warning("Need at least 2 numeric columns for PCA.")
