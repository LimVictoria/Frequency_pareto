import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# --- Helper function to convert Excel column number to Excel-style column name ---
def get_excel_column_name(n: int) -> str:
    """Convert zero-based column index to Excel column letter(s)"""
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

# --- Sidebar: Input Data and Settings ---
with st.sidebar:
    st.header("📥 Input Data")
    input_method = st.radio("How do you want to provide your data?", ["Upload Excel file", "Enter manually"])

    data = []

    if input_method == "Upload Excel file":
        uploaded_file = st.file_uploader("Upload Excel file (.xlsx):", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df_cleaned = df.dropna(axis=1, how='all')

                if df_cleaned.shape[1] == 0:
                    st.error("❌ No usable data found in the uploaded Excel file.")
                else:
                    column_mapping = {
                        get_excel_column_name(i): col for i, col in enumerate(df.columns) if not df[col].dropna().empty
                    }
                    if len(column_mapping) > 1:
                        selected_key = st.selectbox("Select column with numeric data:", list(column_mapping.keys()))
                    else:
                        selected_key = list(column_mapping.keys())[0]

                    selected_column = column_mapping[selected_key]
                    col_data = df[selected_column].dropna()
                    data = [float(x) for x in col_data if isinstance(x, (int, float, np.integer, np.floating))]

            except Exception as e:
                st.error(f"❌ Failed to read file: {str(e)}")

    elif input_method == "Enter manually":
        user_input = st.text_area("Enter numbers (comma-separated):", height=100)
        if user_input:
            try:
                data = [float(x.strip()) for x in user_input.split(",") if x.strip() != '']
            except ValueError:
                st.error("❌ Please enter valid comma-separated numbers.")

    st.markdown("---")
    st.header("⚙️ Settings")

    transformation = st.selectbox("Select normalization method:", 
                                  ["Sample to Standard Normal", "Population to Standard Normal"])

    if transformation == "Population to Standard Normal":
        pop_mean = st.number_input("Population Mean:", value=0.0)
        pop_std = st.number_input("Population Standard Deviation:", value=1.0, min_value=1e-6)

# --- Main Panel ---
st.title("🔄 Normalization Visualizer")

if data:
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0
    sem = std / np.sqrt(n) if n > 0 else 0

    st.subheader("Original Data Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        density = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 300)
        ax.plot(x_vals, density(x_vals), color='skyblue', lw=2)
        ax.fill_between(x_vals, 0, density(x_vals), color='skyblue', alpha=0.5)
    except Exception:
        ax.hist(data, bins='auto', color='skyblue', alpha=0.5, density=True)

    ax.set_title("Original Data Density (KDE)")
    ax.set_xlabel("Data values")
    ax.set_ylabel("Density")
    st.pyplot(fig)

    st.markdown("""
    #### Calculation Formulas:
    - **Sample Mean**: 𝑥̄ = (Σxᵢ) / n  
    - **Sample Standard Deviation (s)**: s = √[ Σ(xᵢ - 𝑥̄)² / (n - 1) ]  
    - **Standard Error of the Mean (SEM)**: σₓ̄ = s / √n

    #### Computed Statistics:
    """)
    st.markdown(f"""
    - **Count (n)**: {n}  
    - **Sample Mean (𝑥̄)**: {mean:.4f}  
    - **Sample Standard Deviation (s)**: {std:.4f}  
    - **Standard Error of Mean (SEM, σₓ̄)**: {sem:.4f}
    """)

    st.subheader("Central Limit Theorem (CLT) Approximation")
    if n >= 30:
        st.markdown("Since sample size ≥ 30, CLT can be applied showing normal approximation of the sample mean.")
        sample_std = std / np.sqrt(n)
        x_clt = np.linspace(mean - 4 * sample_std, mean + 4 * sample_std, 300)
        y_clt = norm.pdf(x_clt, mean, sample_std)

        fig_clt, ax_clt = plt.subplots(figsize=(6, 4))
        ax_clt.plot(x_clt, y_clt, 'red', label='Normal Distribution Curve (CLT)')
        ax_clt.axvline(mean, color='blue', linestyle='--', label=f'Mean = {mean:.2f}')
        ax_clt.axvline(mean - sample_std, color='green', linestyle='--', label=f'Mean - Std Dev = {mean - sample_std:.2f}')
        ax_clt.axvline(mean + sample_std, color='green', linestyle='--', label=f'Mean + Std Dev = {mean + sample_std:.2f}')
        ax_clt.set_title(f"CLT Approximation Using Full Data (n={n})")
        ax_clt.legend(title=f"Sample Std Dev of Mean = {sample_std:.4f}", loc='upper right', fontsize='small')
        ax_clt.set_xlabel("Sample Mean Values")
        ax_clt.set_ylabel("Probability Density")
        st.pyplot(fig_clt)
    else:
        st.warning("⚠️ Sample size less than 30. CLT approximation may not be reliable.")

    st.subheader("Normalization")
    norm_data = []
    if transformation == "Sample to Standard Normal":
        if std == 0:
            st.error("❌ Sample standard deviation is zero, cannot perform sample normalization.")
        else:
            norm_data = [(x - mean) / std for x in data]
            st.markdown(f"""
            #### Sample Normalization Formula:
            - **z = (x - 𝑥̄) / s**

            #### Calculation Values:
            - Sample Mean (𝑥̄): {mean:.4f}  
            - Sample Standard Deviation (s): {std:.4f}
            """)
            st.success("✅ Normalized using Sample Mean and Std Dev")
    else:
        if pop_std == 0:
            st.error("❌ Population standard deviation cannot be zero.")
        else:
            norm_data = [(x - pop_mean) / pop_std for x in data]
            st.markdown(f"""
            #### Population Normalization Formula:
            - **z = (x - μ) / σ**

            #### Calculation Values:
            - Population Mean (μ): {pop_mean:.4f}  
            - Population Standard Deviation (σ): {pop_std:.4f}
            """)
            st.success("✅ Normalized using Population Mean and Std Dev")

    if norm_data:
        st.subheader("Standard Normalized Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        try:
            density_norm = gaussian_kde(norm_data)
            max_abs = max(abs(min(norm_data)), abs(max(norm_data)), 4)
            x_norm = np.linspace(-max_abs, max_abs, 300)
            ax2.plot(x_norm, density_norm(x_norm), color='lightgreen', lw=2, label='KDE of Normalized Data')
        except Exception:
            ax2.hist(norm_data, bins='auto', color='lightgreen', alpha=0.5, density=True)

        y_norm = norm.pdf(x_norm, 0, 1)
        ax2.plot(x_norm, y_norm, 'purple', linestyle='--', lw=2, label='Standard Normal PDF')

        ax2.set_title("Standard Normalized Data Density")
        ax2.set_xlabel("Normalized Values")
        ax2.set_ylabel("Density")
        ax2.legend(loc='upper right', fontsize='small')
        st.pyplot(fig2)
else:
    st.warning("⚠️ Please provide input data to continue.")
