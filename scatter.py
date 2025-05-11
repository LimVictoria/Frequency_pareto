import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def find_cross_table(df):
    rows, cols = df.shape

    for r in range(rows - 1):
        for c in range(cols - 1):
            if not isinstance(df.iat[r, c], str):
                continue  # top-left must be a string

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

st.title("Cross Table Detector & Scatter Plot Visualizer")

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

        if not has_totals(detected):
            with_totals = detected.copy()
            with_totals["Total"] = with_totals.sum(axis=1)
            total_row = with_totals.sum(axis=0)
            total_row.name = "Total"
            with_totals = pd.concat([with_totals, total_row.to_frame().T])
        else:
            with_totals = detected.copy()

        if "Total" in with_totals.columns:
            with_totals = with_totals.drop(columns=["Total"])
        if "Total" in with_totals.index:
            with_totals = with_totals.drop(index="Total")

        st.subheader("Scatter Plot (Auto-detected Columns)")
        fig, ax = plt.subplots(figsize=(10, 6))

        num_columns = with_totals.shape[1]
        
        # Check if the first column contains strings
        first_col_is_string = all(isinstance(val, str) for val in with_totals.iloc[:, 0])

        try:
            if first_col_is_string:
                if num_columns >= 4:
                    x = with_totals.iloc[:, 1].values  # 2nd column
                    y = with_totals.iloc[:, 2].values  # 3rd column
                    ax.set_xlabel(with_totals.columns[1])
                    ax.set_ylabel(with_totals.columns[2])
                    title = "XY Scatter Plot (Cols 2 & 3)"
                    if num_columns >= 5:
                        z = with_totals.iloc[:, 3].values  # 4th column
                        scatter = ax.scatter(x, y, c=z, cmap='viridis')
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label(with_totals.columns[3])
                        title = "XYZ Scatter Plot (Cols 2, 3, 4)"
                    else:
                        ax.scatter(x, y)
                    ax.set_title(title)
                else:
                    st.warning("❗ Not enough columns to plot XY or XYZ (need ≥ 4 cols when 1st col is string).")
            else:
                if num_columns >= 3:
                    x = with_totals.iloc[:, 1].values  # 2nd column
                    y = with_totals.iloc[:, 2].values  # 3rd column
                    ax.set_xlabel(with_totals.columns[1])
                    ax.set_ylabel(with_totals.columns[2])
                    title = "XY Scatter Plot (Cols 2 & 3)"
                    if num_columns >= 4:
                        z = with_totals.iloc[:, 3].values  # 4th column
                        scatter = ax.scatter(x, y, c=z, cmap='plasma')
                        cbar = fig.colorbar(scatter, ax=ax)
                        cbar.set_label(with_totals.columns[3])
                        title = "XYZ Scatter Plot (Cols 2, 3, 4)"
                    else:
                        ax.scatter(x, y)
                    ax.set_title(title)
                else:
                    st.warning("❗ Not enough columns to plot XY or XYZ (need ≥ 3 cols when 1st col is not string).")

            if 'x' in locals() and 'y' in locals():
                ax.set_xlim(left=0, right=max(x) * 1.1 if len(x) > 0 else 1)
                ax.set_ylim(bottom=0, top=max(y) * 1.1 if len(y) > 0 else 1)
                ax.grid(True)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error generating plot: {e}")

    else:
        st.error("❌ No cross table detected with required pattern.")
