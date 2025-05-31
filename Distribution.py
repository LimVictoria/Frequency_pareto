import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from io import BytesIO
import matplotlib.ticker as ticker

# --- Helper: Excel column name from index ---
def get_excel_column_name(n: int) -> str:
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

# --- Sidebar: Upload/Input & Settings ---
with st.sidebar:
    st.header("Input Data")
    input_method = st.radio("Provide data via:", ["Upload Excel file", "Enter manually"])

    data = []

    if input_method == "Upload Excel file":
        uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df_cleaned = df.dropna(axis=1, how='all')
                if df_cleaned.shape[1] == 0:
                    st.error("❌ Uploaded file contains no usable columns.")
                else:
                    column_mapping = {get_excel_column_name(i): col for i, col in enumerate(df_cleaned.columns)}
                    selected_key = st.selectbox("Select column with numeric data:", list(column_mapping.keys()))
                    selected_column = column_mapping[selected_key]
                    col_data = df_cleaned[selected_column].dropna()
                    data = [float(x) for x in col_data if isinstance(x, (int, float, np.integer, np.floating))]
            except Exception as e:
                st.error(f"❌ Error reading Excel file: {str(e)}")

    else:  # Manual input
        user_input = st.text_area("Enter numbers (comma-separated):", height=100)
        if user_input:
            try:
                data = [float(x.strip()) for x in user_input.split(",") if x.strip() != '']
            except ValueError:
                st.error("❌ Invalid input. Ensure values are comma-separated numbers.")

    # Normalization options
    st.markdown("---")
    st.header("Normalization Settings")
    transformation = st.selectbox("Normalization method:", 
                                  ["Sample to Standard Normal", "Population to Standard Normal"])

    if transformation == "Population to Standard Normal":
        pop_mean = st.number_input("Population Mean (μ):", value=0.0)
        pop_std = st.number_input("Population Standard Deviation (σ):", value=1.0, min_value=1e-6)

    show_hist = st.checkbox("Show Histogram Overlay", value=True)

    if show_hist:
        num_bins = st.slider("Number of intervals for histogram:", min_value=5, max_value=100, value=30)
    else:
        num_bins = 30  # Default fallback when histogram is hidden

# --- Main Panel ---
st.title("Distribution")

if data:
    # --- Compute Basic Stats ---
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0
    sem = std / np.sqrt(n) if n > 0 else 0

    st.subheader("Sample Distribution - Gaussian KDE applied")
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        density = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 500)
        ax.plot(x_vals, density(x_vals), color='red', lw=2, label='Sample dist.')
    except Exception:
        ax.hist(data, bins=num_bins, color='red', alpha=0.5, density=True)
    if show_hist:
        ax.hist(data, bins=num_bins, color='gray', alpha=0.2, density=True, label='Histogram')
    ax.set_xlabel("Data points")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend()
    st.pyplot(fig)

    # --- Show Stats ---
    st.markdown("#### Statistics")
    st.markdown(f"""
    - **Count (n)**: {n}  
    - **Sample Mean (𝑥̄)**: {mean:.1f}  
    - **Sample Standard Deviation (s)**: {std:.1f}  
    - **Standard Error of Mean (SEM)**: {sem:.1f}
    """)
    
    st.write("")  
    st.write("") 
    st.write("") 
    st.subheader("Normal Distribution - CLT Approximation applied")
    if n >= 30:
        sample_std = std / np.sqrt(n)
        x_clt = np.linspace(mean - 4 * sample_std, mean + 4 * sample_std, 300)
        y_clt = norm.pdf(x_clt, mean, sample_std)

        fig_clt, ax_clt = plt.subplots(figsize=(6, 4))
        ax_clt.plot(x_clt, y_clt, 'r-', label='CLT Normal Curve')
        ax_clt.axvline(mean, color='blue', linestyle='--', label=f'Mean = {mean:.1f}')
        ax_clt.axvline(mean - sample_std, color='green', linestyle='--', label=f'Mean - Std err mean = {mean - sample_std:.1f}')
        ax_clt.axvline(mean + sample_std, color='green', linestyle='--', label=f'Mean + Std err mean = {mean + sample_std:.1f}')
        ax_clt.set_xlabel("Data points")
        ax_clt.set_ylabel("Frequency")
        ax_clt.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax_clt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax_clt.legend(loc='upper right', fontsize='small')
        st.pyplot(fig_clt)
    else:
        st.warning("CLT not recommended (sample size < 30).")

    st.markdown("#### Z-score")
    norm_data = []
    if transformation == "Sample to Standard Normal":
        if std == 0:
            st.error("❌ Sample standard deviation is 0, normalization not possible.")
        else:
            norm_data = [(x - mean) / std for x in data]
            st.markdown(f"**Formula:** z = (x - 𝑥̄) / s")
    else:
        if pop_std == 0:
            st.error("❌ Population std deviation cannot be zero.")
        else:
            norm_data = [(x - pop_mean) / pop_std for x in data]
            st.markdown(f"**Formula:** z = (x - μ) / σ")

    st.write("")
    st.write("")
    st.write("")
    if norm_data:
        st.subheader("Standard Normal Distribution , Z ~ N(0, 1)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        try:
            density_norm = gaussian_kde(norm_data)
            x_norm = np.linspace(-4, 4, 300)
            ax2.plot(x_norm, density_norm(x_norm), color='red', lw=2, label='Sample dist.')
        except Exception:
            ax2.hist(norm_data, bins=num_bins, color='red', alpha=0.5, density=True)

        y_norm = norm.pdf(x_norm, 0, 1)
        ax2.plot(x_norm, y_norm, 'green', linestyle='--', lw=2, label='Normal dist.')
        if show_hist:
            ax2.hist(norm_data, bins=num_bins, alpha=0.2, color='gray', density=True, label='Histogram')

        ax2.set_xlabel("Z-score")
        ax2.set_ylabel("Frequency")
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.legend(loc='upper right', fontsize='small')
        st.pyplot(fig2)

        # --- Downloadable CSV of Normalized Data ---
        df_norm = pd.DataFrame({"Original": data, "Normalized": norm_data})
        csv = df_norm.to_csv(index=False).encode()
        st.download_button("⬇️ Download Z-scores as CSV", csv, "Z-scores.csv", "text/csv")

else:
    st.info("Please upload or enter data using the sidebar.")
