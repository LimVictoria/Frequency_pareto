import streamlit as st
import pandas as pd

def find_cross_table(df):
    rows, cols = df.shape

    for r in range(rows - 1):
        for c in range(cols - 1):
            if not isinstance(df.iat[r, c], str):
                continue  # top-left must be a string

            # Detect column headers (rightwards from top-left)
            col_end = c + 1
            while col_end < cols and isinstance(df.iat[r, col_end], str):
                col_end += 1

            # Detect row labels (downwards from top-left)
            row_end = r + 1
            while row_end < rows and isinstance(df.iat[row_end, c], str):
                row_end += 1

            # Extract numeric body
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
    # Check if the last column contains totals
    last_col = df.iloc[:, -1]
    if last_col.isna().all():  # If the last column has any NaN, it doesn't contain totals
        return False

    # Check if the last row contains totals
    last_row = df.iloc[-1, :]
    if last_row.isna().all():  # If the last row has any NaN, it doesn't contain totals
        return False

    # Validate if the row totals and column totals are correct
    row_totals = df.iloc[:-1, -1]
    col_totals = df.iloc[-1, :-1]

    expected_row_totals = df.iloc[:-1, :-1].sum(axis=1)
    expected_col_totals = df.iloc[:-1, :-1].sum(axis=0)

    if not row_totals.equals(expected_row_totals) or not col_totals.equals(expected_col_totals):
        return False

    return True


st.title("Cross Table Detector (Position Independent)")

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

        # --- Clean the detected cross table ---
        detected = result.copy()
        detected.columns = detected.iloc[0]
        detected = detected[1:]
        detected.index = detected.iloc[:, 0]
        detected = detected.iloc[:, 1:]
        detected = detected.apply(pd.to_numeric, errors='coerce')

        st.subheader("Detected Cross Table (Original)")
        st.dataframe(detected)

        # --- Check if totals are already present ---
        if not has_totals(detected):
            # Create copy with totals if not present
            with_totals = detected.copy()
            with_totals["Total"] = with_totals.sum(axis=1)
            total_row = with_totals.sum(axis=0)
            total_row.name = "Total"
            with_totals = pd.concat([with_totals, total_row.to_frame().T])

            st.subheader("Cross Table with Totals")
            st.dataframe(with_totals)

    else:
        st.error("❌ No cross table detected with required pattern.")
