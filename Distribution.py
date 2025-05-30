import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Helper to convert Excel column number to name
def get_excel_column_name(n):
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

# Sidebar
with st.sidebar:
    st.header("📥 Input Data")
    input_method = st.radio("How do you want to provide your data?", ["Upload Excel file", "Enter manually"])
    data = []

    if input_method == "Upload Excel file":
        uploaded_file = st.file_uploader("Upload Excel file:", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df_cleaned = df.dropna(axis=1, how='all')
                if df_cleaned.shape[1] == 0:
                    st.error("❌ No usable data found in the uploaded Excel file.")
                else:
                    column_mapping = {}
                    for i, col in enumerate(df.columns):
                        if not df[col].dropna().empty:
                            column_mapping[get_excel_column_name(i)] = col
                    selected_key = list(column_mapping.keys())[0] if len(column_mapping) == 1 else st.selectbox("Select column:", list(column_mapping.keys()))
                    selected_column = column_mapping[selected_key]
                    data = df[selected_column].dropna().tolist()
                    data = [float(x) for x in data if isinstance(x, (int, float))]
            except Exception as e:
                st.error(f"❌ Failed to read file: {str(e)}")

    elif input_method == "Enter manually":
        user_input = st.text_input("Enter numbers (comma-separated):", "")
        if user_input:
            try:
                data = list(map(float, user_input.split(",")))
            except:
                st.error("❌ Please enter valid comma-separated numbers.")

    st.markdown("---")
    st.header("⚙️ Settings")
    transformation = st.selectbox("Select transformation:", ["Sample to Standard Normal", "Population to Standard Normal"])
    if transformation == "Population to Standard Normal":
        pop_mean = st.number_input("Population Mean:", value=0.0)
        pop_std = st.number_input("Population Standard Deviation:", value=1.0)

# Main Panel
st.title("🔄 Normalization Visualizer")

if data:
    # Sample mean and std
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    st.subheader("📊 Original Data Distribution")
    fig, ax = plt.subplots()
    ax.hist(data, bins='auto', color='skyblue', edgecolor='black', density=True)
    ax.set_title("Original Data Histogram")
    st.pyplot(fig)

    # Central Limit Theorem plot if data size >= 30
    if len(data) >= 30:
        st.subheader("📉 Central Limit Theorem (CLT) Approximation")
        st.markdown("Since data ≥ 30, CLT can be applied. This shows the normal approximation of the sample mean.")

        sample_mean = mean
        sample_std = std / np.sqrt(len(data))  # Std dev of sample mean

        x_vals = np.linspace(sample_mean - 4 * sample_std, sample_mean + 4 * sample_std, 300)
        y_vals = norm.pdf(x_vals, sample_mean, sample_std)

        fig_clt, ax_clt = plt.subplots()
        ax_clt.plot(x_vals, y_vals, 'red', label='Normal Curve (CLT)')
        ax_clt.axvline(sample_mean, color='blue', linestyle='--', label=f'Mean = {sample_mean:.2f}')
        ax_clt.axvline(sample_mean - sample_std, color='green', linestyle='--', label=f'Mean - Std Dev = {sample_mean - sample_std:.2f}')
        ax_clt.axvline(sample_mean + sample_std, color='green', linestyle='--', label=f'Mean + Std Dev = {sample_mean + sample_std:.2f}')
        ax_clt.set_title(f"CLT Approximation Using Full Data (n={len(data)})")
        ax_clt.legend(loc='upper right', fontsize='small')
        st.pyplot(fig_clt)
    else:
        st.subheader("📉 Central Limit Theorem (CLT) Approximation")
        st.warning("❌ Data size less than 30. CLT cannot be applied reliably.")

    # Perform normalization
    if transformation == "Sample to Standard Normal":
        norm_data = [(x - mean) / std for x in data]
        st.success(f"Sample Mean: {mean:.2f}, Sample Std Dev: {std:.2f}")
    else:
        norm_data = [(x - pop_mean) / pop_std for x in data]
        st.success(f"Population Mean: {pop_mean:.2f}, Population Std Dev: {pop_std:.2f}")

    # Normalized data plot
    st.subheader("📈 Standard Normalized Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(norm_data, bins='auto', color='lightgreen', edgecolor='black', density=True)

    x_norm = np.linspace(min(norm_data), max(norm_data), 300)
    y_norm = norm.pdf(x_norm, 0, 1)
    ax2.plot(x_norm, y_norm, 'purple', linestyle='--', label='Standard Normal Curve')
    ax2.set_title("Standard Normal Histogram with Curve")
    ax2.legend(loc='upper right', fontsize='small')
    st.pyplot(fig2)

else:
    st.warning("⚠️ Please provide input data to continue.")
