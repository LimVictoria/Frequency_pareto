import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew


# Function to detect cross tables in a DataFrame
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


# Function to check if the table has totals
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


# Streamlit app setup
st.title("Flat table with index & Scatter Plot Generator")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Process uploaded file
if uploaded_file:
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None, header=None)
    sheet_name = list(all_sheets.keys())[0] if len(all_sheets) == 1 else st.selectbox("Select Sheet", list(all_sheets.keys()))
    df = all_sheets[sheet_name].fillna("")

    # Detect cross table
    result, r_start, c_start, r_end, c_end = find_cross_table(df)

    if result is not None:
        # Display results of detected cross table
        start_cell = f"{chr(65 + c_start)}{r_start + 1}"
        end_cell = f"{chr(65 + c_end)}{r_end + 1}"
        st.success(f"✅ Flat table detected")

        # Prepare the detected table for plotting
        detected = result.copy()
        detected.columns = detected.iloc[0]
        detected = detected[1:]
        detected.index = detected.iloc[:, 0]
        detected = detected.iloc[:, 1:]
        detected = detected.apply(pd.to_numeric, errors='coerce')

        # Add totals if necessary
        if not has_totals(detected):
            with_totals = detected.copy()
            with_totals["Total"] = with_totals.sum(axis=1)
            total_row = with_totals.sum(axis=0)
            total_row.name = "Total"
            with_totals = pd.concat([with_totals, total_row.to_frame().T])
        else:
            with_totals = detected.copy()

        # Remove totals for plotting
        if "Total" in with_totals.columns:
            with_totals = with_totals.drop(columns=["Total"])
        if "Total" in with_totals.index:
            with_totals = with_totals.drop(index="Total")

        # 2D Scatter Plot
        st.subheader("2D Scatter Plot")
        fig2d, ax2d = plt.subplots(figsize=(10, 6))

        num_columns = with_totals.shape[1]
        first_col_is_string = all(isinstance(val, str) for val in with_totals.iloc[:, 0])

        x = y = z = None  # Initialize x, y, z to None

        try:
            if first_col_is_string and num_columns >= 4:
                x = with_totals.iloc[:, 1].values
                y = with_totals.iloc[:, 2].values
                z = with_totals.iloc[:, 3].values
                scatter = ax2d.scatter(x, y, c=z, cmap='cividis')
                ax2d.set_xlabel(with_totals.columns[1])
                ax2d.set_ylabel(with_totals.columns[2])
                ax2d.set_title("XYZ Scatter Plot (Cols 2, 3, 4)")
                cbar = fig2d.colorbar(scatter, ax=ax2d)
                cbar.set_label(with_totals.columns[3])
            elif not first_col_is_string and num_columns >= 4:
                x = with_totals.iloc[:, 1].values
                y = with_totals.iloc[:, 2].values
                z = with_totals.iloc[:, 3].values
                scatter = ax2d.scatter(x, y, c=z, cmap='plasma')
                ax2d.set_xlabel(with_totals.columns[1])
                ax2d.set_ylabel(with_totals.columns[2])
                ax2d.set_title("XYZ Scatter Plot (Cols 2, 3, 4)")
                cbar = fig2d.colorbar(scatter, ax=ax2d)
                cbar.set_label(with_totals.columns[3])
            elif num_columns >= 3:
                x = with_totals.iloc[:, 1].values
                y = with_totals.iloc[:, 2].values
                ax2d.scatter(x, y)
                ax2d.set_xlabel(with_totals.columns[1])
                ax2d.set_ylabel(with_totals.columns[2])
                ax2d.set_title("XY Scatter Plot (Cols 2 & 3)")
            else:
                st.warning("❗ Not Flat table")

            if x is not None and y is not None:
                ax2d.set_xlim(left=0, right=max(x) * 1.1 if len(x) > 0 else 1)
                ax2d.set_ylim(bottom=0, top=max(y) * 1.1 if len(y) > 0 else 1)
                ax2d.grid(True)
                st.pyplot(fig2d)

            # 3D scatter plot
            if x is not None and y is not None and z is not None:
                st.subheader("3D Scatter Plot")
                fig3d = plt.figure(figsize=(10, 7))
                ax3d = fig3d.add_subplot(111, projection='3d')
                sc = ax3d.scatter(x, y, z, c=z, cmap='plasma', depthshade=True, alpha=1.0)
                ax3d.set_xlabel(with_totals.columns[1])
                ax3d.set_ylabel(with_totals.columns[2])
                ax3d.set_zlabel(with_totals.columns[3])
                ax3d.set_title("3D Scatter Plot (Cols 2, 3, 4)")
                fig3d.colorbar(sc, ax=ax3d, shrink=0.5, aspect=10)
                st.pyplot(fig3d)

            # Show the detected cross table as a table
            st.subheader("Detected Flat Table")
            st.dataframe(detected)

            # Display statistics and skewness
            st.subheader("Statistics and Skewness")

            def display_stats(label, data):
                s = pd.Series(data)
                mode_val = s.mode()
                mode_val = mode_val.iloc[0] if not mode_val.empty else "N/A"
                skew_val = skew(s.dropna())

                if skew_val > 0:
                    skew_desc = "Positive"
                elif skew_val < 0:
                    skew_desc = "Negative"
                else:
                    skew_desc = "Zero"

                stats_df = pd.DataFrame({
                    "Mean": [s.mean()],
                    "Median": [s.median()],
                    "Mode": [mode_val],
                    "Skewness": [round(skew_val, 3)],
                    "Skewness Type": [skew_desc]
                }, index=[label])
                st.table(stats_df)

            if x is not None:
                display_stats("X axis", x)
            if y is not None:
                display_stats("Y axis", y)
            if z is not None:
                display_stats("Z axis", z)

        except Exception as e:
            st.error(f"❌ Error generating plot: {e}")
    else:
        st.error("❌ No cross table detected with the required pattern.")
