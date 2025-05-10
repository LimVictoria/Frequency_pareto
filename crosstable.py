import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_cross_table(df):
    rows, cols = df.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            if not isinstance(df.iat[r, c], str):
                continue

            col_end = c + 1
            while col_end < cols and isinstance(df.iat[r, col_end], str):
                col_end += 1

            row_end = r + 1
            while row_end < rows and isinstance(df.iat[row_end, c], str):
                row_end += 1

            body = df.iloc[r+1:row_end, c+1:col_end]
            try:
                numeric_body = pd.to_numeric(body.stack(), errors='coerce')
                if numeric_body.notnull().all():
                    sub_df = df.iloc[r:row_end, c:col_end].reset_index(drop=True)
                    return sub_df, r, c, row_end - 1, col_end - 1
            except Exception:
                continue

    return None, None, None, None, None

def has_totals(df):
    if df.empty:
        return False

    last_col = df.iloc[:, -1]
    last_row = df.iloc[-1, :]

    if last_col.isna().any() or last_row.isna().any():
        return False

    row_totals = df.iloc[:-1, -1]
    col_totals = df.iloc[-1, :-1]

    expected_row_totals = df.iloc[:-1, :-1].sum(axis=1)
    expected_col_totals = df.iloc[:-1, :-1].sum(axis=0)

    return row_totals.equals(expected_row_totals) and col_totals.equals(expected_col_totals)

st.title("Cross Table & Grouped Bar Chart Generator")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None)
    sheet_name = list(all_sheets.keys())[0] if len(all_sheets) == 1 else st.selectbox("Select Sheet", list(all_sheets.keys()))
    df = all_sheets[sheet_name].fillna("")

    result, r_start, c_start, r_end, c_end = find_cross_table(df)

    if result is not None:
        start_cell = f"{chr(65 + c_start)}{r_start + 1}"
        end_cell = f"{chr(65 + c_end)}{r_end + 1}"
        st.success(f"✅ Cross table detected from **{start_cell}** to **{end_cell}**")

        detected = result.copy()
        detected.columns = detected.iloc[0]
        detected = detected[1:]
        detected.index = detected.iloc[:, 0]
        detected = detected.iloc[:, 1:]
        detected = detected.apply(pd.to_numeric, errors='coerce')

        st.subheader("Detected Cross Table (Original)")
        st.dataframe(detected)

        if not has_totals(detected):
            with_totals = detected.copy()
            with_totals["Total"] = with_totals.sum(axis=1)
            total_row = with_totals.sum(axis=0)
            total_row.name = "Total"
            with_totals = pd.concat([with_totals, total_row.to_frame().T])
            st.subheader("Cross Table with Totals")
            st.dataframe(with_totals)
        else:
            with_totals = detected.copy()

        # Remove totals for plotting
        if "Total" in with_totals.columns:
            with_totals = with_totals.drop(columns=["Total"])
        if "Total" in with_totals.index:
            with_totals = with_totals.drop(index="Total")

        # Grouped Bar Chart (Row-wise)
        st.subheader("Grouped Bar Chart (Row-wise)")
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        x = np.arange(len(with_totals.index))
        bar_width = 0.8 / len(with_totals.columns)

        for i, col in enumerate(with_totals.columns):
            ax1.bar(x + i * bar_width, with_totals[col], width=bar_width, label=col)

        ax1.set_ylabel('Value')
        ax1.set_title('Grouped Bar Chart (by Rows)')
        ax1.set_xticks(x + bar_width * (len(with_totals.columns) / 2 - 0.5))
        ax1.set_xticklabels(with_totals.index, rotation=45)
        ax1.legend(title="Columns")
        st.pyplot(fig1)

        # Grouped Bar Chart (Column-wise)
        st.subheader("Grouped Bar Chart (Column-wise)")
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        x_col = np.arange(len(with_totals.columns))
        bar_width_col = 0.8 / len(with_totals.index)

        for i, (row_label, row_data) in enumerate(with_totals.iterrows()):
            x_positions = x_col + i * bar_width_col
            heights = row_data.values.astype(float)
            ax2.bar(x_positions, heights, width=bar_width_col, label=row_label)


        ax2.set_ylabel('Value')
        ax2.set_title('Grouped Bar Chart (by Columns)')
        ax2.set_xticks(x_col + bar_width_col * (len(with_totals.index) / 2 - 0.5))
        ax2.set_xticklabels(with_totals.columns, rotation=45)
        ax2.legend(title="Rows")
        st.pyplot(fig2)

    else:
        st.error("❌ No cross table detected with required pattern.")
