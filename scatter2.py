import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

# --- Helper Functions ---
def find_cross_table(df):
    rows, cols = df.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            if not isinstance(df.iat[r, c], str):
                continue
            col_end, row_end = c + 1, r + 1
            while col_end < cols and isinstance(df.iat[r, col_end], str):
                col_end += 1
            while row_end < rows and isinstance(df.iat[row_end, c], str):
                row_end += 1
            body = df.iloc[r+1:row_end, c+1:col_end]
            try:
                numeric_body = pd.to_numeric(body.stack(), errors='coerce')
                if numeric_body.notnull().all():
                    return df.iloc[r:row_end, c:col_end].reset_index(drop=True), r, c, row_end - 1, col_end - 1
            except Exception:
                continue
    return None, None, None, None, None

def has_totals(df):
    if df.empty:
        return False
    try:
        row_totals = df.iloc[:-1, -1]
        col_totals = df.iloc[-1, :-1]
        expected_row_totals = df.iloc[:-1, :-1].sum(axis=1)
        expected_col_totals = df.iloc[:-1, :-1].sum(axis=0)
        return row_totals.equals(expected_row_totals) and col_totals.equals(expected_col_totals)
    except:
        return False

def is_flat_table(df):
    try:
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        first_col_values = df.iloc[1:, 0]
        return pd.to_numeric(first_col_values, errors='coerce').notnull().all()
    except:
        return False

def summarize_statistics(label, data):
    if len(data) == 0:
        return
    mode_val = stats.mode(data, keepdims=True).mode[0]
    mean_val = np.mean(data)
    median_val = np.median(data)
    skew_val = stats.skew(data) if len(data) > 2 else None
    skewness_type = (
        "Positively skewed" if skew_val > 0 else
        "Negatively skewed" if skew_val < 0 else
        "Zero skew (symmetric)" if skew_val == 0 else
        "Not enough data"
    )
    st.markdown(f"""
    ### 📊 Statistics for **{label}**
    - **Mode**: {mode_val:.2f}
    - **Mean**: {mean_val:.2f}
    - **Median**: {median_val:.2f}
    - **Skewness**: {skew_val:.2f} → *{skewness_type}*
    """)

def plot_2d(x, y, color=None, xlabel="", ylabel="", clabel=""):
    fig, ax = plt.subplots()
    if color is not None:
        sc = ax.scatter(x, y, c=color, cmap="viridis")
        plt.colorbar(sc, label=clabel)
    else:
        ax.scatter(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("2D Scatter Plot")
    st.pyplot(fig)

def plot_3d(x, y, z, xlabel="", ylabel="", zlabel=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap="plasma")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(sc, shrink=0.5, aspect=10, label=zlabel)
    ax.set_title("3D Scatter Plot")
    st.pyplot(fig)

# --- Streamlit App ---
st.set_page_config(page_title="Table Type Visualizer", layout="wide")
st.title("📊 Table Type Detector & Visualizer")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None)
    sheet_name = list(all_sheets.keys())[0] if len(all_sheets) == 1 else st.selectbox("Select Sheet", list(all_sheets.keys()))
    df = all_sheets[sheet_name].fillna("")

    # Check flat table
    if is_flat_table(df):
        st.success("✅ Flat table detected")
        data = df.copy()
        data.columns = data.iloc[0]
        data = data[1:].apply(pd.to_numeric, errors='coerce').dropna()
        st.subheader("📈 Flat Table Visualization")
        num_cols = data.shape[1]

        if num_cols >= 3:
            x, y, z = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
            plot_2d(x, y, color=z, xlabel=data.columns[0], ylabel=data.columns[1], clabel=data.columns[2])
            plot_3d(x, y, z, xlabel=data.columns[0], ylabel=data.columns[1], zlabel=data.columns[2])
        elif num_cols >= 2:
            x, y = data.iloc[:, 0], data.iloc[:, 1]
            plot_2d(x, y, xlabel=data.columns[0], ylabel=data.columns[1])
        else:
            st.warning("❗ Not enough columns to plot.")

    else:
        cross_df, r_start, c_start, r_end, c_end = find_cross_table(df)
        if cross_df is not None:
            st.success(f"✅ Cross table detected from **{chr(65 + c_start)}{r_start + 1}** to **{chr(65 + c_end)}{r_end + 1}**")

            detected = cross_df.copy()
            detected.columns = detected.iloc[0]
            detected = detected[1:]
            detected.index = detected.iloc[:, 0]
            detected = detected.iloc[:, 1:]
            detected = detected.apply(pd.to_numeric, errors='coerce')

            if not has_totals(detected):
                detected["Total"] = detected.sum(axis=1)
                total_row = detected.sum(axis=0)
                total_row.name = "Total"
                detected = pd.concat([detected, total_row.to_frame().T])

            # Clean any remaining totals
            detected = detected.drop(columns=["Total"], errors='ignore')
            detected = detected.drop(index="Total", errors='ignore')

            st.subheader("📈 Cross Table Visualization")
            num_cols = detected.shape[1]
            try:
                if num_cols >= 3:
                    x = detected.iloc[:, 0].values
                    y = detected.iloc[:, 1].values
                    z = detected.iloc[:, 2].values
                    plot_2d(x, y, color=z, xlabel=detected.columns[0], ylabel=detected.columns[1], clabel=detected.columns[2])
                    plot_3d(x, y, z, xlabel=detected.columns[0], ylabel=detected.columns[1], zlabel=detected.columns[2])
                elif num_cols >= 2:
                    x = detected.iloc[:, 0].values
                    y = detected.iloc[:, 1].values
                    plot_2d(x, y, xlabel=detected.columns[0], ylabel=detected.columns[1])
                else:
                    st.warning("❗ Not enough columns to plot.")
            except Exception as e:
                st.error(f"Plotting error: {e}")

            st.subheader("📋 Detected Cross Table")
            st.dataframe(detected)

    # Show statistics
    if "x" in locals() and "y" in locals():
        summarize_statistics("X Axis", x)
        summarize_statistics("Y Axis", y)
    if "z" in locals():
        summarize_statistics("Z Axis", z)
