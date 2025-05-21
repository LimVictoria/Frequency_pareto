import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

st.title("Frequency, Pareto Chart & Histogram Generator")

def get_excel_column_name(n):
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

def is_categorical_column(series):
    # Drop NaN
    clean = series.dropna()
    # Check all values are strings (including pandas object dtype)
    # Treat datetime as string too
    if clean.empty:
        return False
    return all(isinstance(x, str) or pd.api.types.is_datetime64_any_dtype(type(x)) for x in clean)

def is_numerical_column(series):
    # Drop NaN
    clean = series.dropna()
    if clean.empty:
        return False

    # Check if all numeric
    if pd.api.types.is_numeric_dtype(clean):
        return True

    # If dtype is object, check if first row is string and rest are numeric
    if len(clean) < 2:
        # If only one row and it's numeric, treat as numeric col
        try:
            float(clean.iloc[0])
            return True
        except:
            return False

    first = clean.iloc[0]
    rest = clean.iloc[1:]

    # First row string and rest numeric?
    if isinstance(first, str):
        # Check rest are numeric
        try:
            # Convert rest to float to check numeric
            rest_floats = rest.astype(float)
            return True
        except:
            return False

    # Otherwise check if all numeric (try converting all)
    try:
        clean.astype(float)
        return True
    except:
        return False

    return False

# Sidebar for data input
with st.sidebar:
    st.header("Input Data")
    input_method = st.radio("How do you want to provide your data?", ["Upload Excel file", "Enter manually"])
    data = []
    selected_type = None
    column_mapping = {}

    if input_method == "Upload Excel file":
        uploaded_file = st.file_uploader("Upload Excel file:", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)

                if df.shape[1] == 0:
                    st.error("❌ No columns found in the uploaded Excel file.")
                else:
                    # Find first non-empty column index in original df
                    first_non_empty_col_index = None
                    for idx, col in enumerate(df.columns):
                        if df[col].dropna().shape[0] > 0:
                            first_non_empty_col_index = idx
                            break

                    if first_non_empty_col_index is None:
                        st.error("❌ No non-empty column found in the Excel file.")
                    else:
                        selected_type = st.radio("Select data type:", ["Categorical", "Numerical"])

                        # Build column mapping from first non-empty column onwards based on your rules
                        column_mapping = {}
                        for idx, col in enumerate(df.columns):
                            if idx < first_non_empty_col_index:
                                continue  # skip empty leading columns
                            col_data = df[col]
                            if selected_type == "Categorical":
                                if is_categorical_column(col_data):
                                    column_mapping[get_excel_column_name(idx)] = col
                            elif selected_type == "Numerical":
                                if is_numerical_column(col_data):
                                    column_mapping[get_excel_column_name(idx)] = col

                        if len(column_mapping) == 0:
                            st.error(f"❌ No {selected_type.lower()} columns found starting from column {get_excel_column_name(first_non_empty_col_index)}.")
                        else:
                            if len(column_mapping) == 1:
                                selected_key = list(column_mapping.keys())[0]
                            else:
                                selected_key = st.selectbox("Select column:", list(column_mapping.keys()))
                            selected_column = column_mapping[selected_key]
                            # For numerical with first row string: drop that first row
                            if selected_type == "Numerical":
                                col_data = df[selected_column].dropna()
                                if isinstance(col_data.iloc[0], str):
                                    data = col_data.iloc[1:].astype(float).tolist()
                                else:
                                    data = col_data.astype(float).tolist()
                            else:
                                data = df[selected_column].dropna().astype(str).tolist()

            except Exception as e:
                st.error(f"❌ Failed to read file: {str(e)}")

    elif input_method == "Enter manually":
        selected_type = st.radio("Select data type:", ["Categorical", "Numerical"])
        user_input = st.text_input("Enter values (comma-separated):", "")
        if user_input:
            try:
                if selected_type == "Numerical":
                    data = list(map(float, user_input.split(",")))
                else:
                    data = list(map(str, user_input.split(",")))
            except:
                st.error("❌ Please enter valid comma-separated values.")

    num_intervals = st.text_input("Enter number of intervals (for numerical only):", "")
    st.button("Finish")

# Main app section
col1, col2 = st.columns([1, 2])

with col2:
    if data:
        if selected_type == "Categorical":
            data = [str(x) for x in data]

            frequency_dict = {}
            for item in data:
                frequency_dict[item] = frequency_dict.get(item, 0) + 1

            freq_df = pd.DataFrame({
                "Category": list(frequency_dict.keys()),
                "Frequency": list(frequency_dict.values())
            }).sort_values("Frequency", ascending=False).reset_index(drop=True)
            freq_df.index += 1

            st.subheader("Categorical Frequency Table")
            st.table(freq_df)

            cumulative_freq = np.cumsum(freq_df["Frequency"])
            cumulative_percentages = (cumulative_freq / cumulative_freq.iloc[-1]) * 100

            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.bar(freq_df["Category"], freq_df["Frequency"], color='skyblue', alpha=0.7)
            ax1.set_xlabel("Category")
            ax1.set_ylabel("Frequency", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

            ax2 = ax1.twinx()
            ax2.plot(freq_df["Category"], cumulative_percentages, color='red', marker='o', linestyle='-')
            ax2.set_ylabel("Cumulative Percentage (%)", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 110)

            plt.title("Pareto Chart (Categorical Data)")
            plt.grid(True, linestyle="--", alpha=0.7)
            st.subheader("Pareto Chart")
            st.pyplot(fig)

            fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
            ax_hist.bar(freq_df["Category"], freq_df["Frequency"], color='lightgreen', edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel("Category")
            ax_hist.set_ylabel("Frequency")
            ax_hist.set_title("Histogram")
            plt.setp(ax_hist.get_xticklabels(), rotation=45, ha='right')
            st.subheader("Histogram")
            st.pyplot(fig_hist)

        elif selected_type == "Numerical" and num_intervals.isdigit():
            try:
                data = [float(x) for x in data if isinstance(x, (int, float))]
                interval = int(num_intervals)
                largest = max(data)
                smallest = min(data)
                diff = largest - smallest
                interval_length = math.ceil(diff / interval)

                frequency_dict = {}
                for num in data:
                    frequency_dict[num] = frequency_dict.get(num, 0) + 1

                freq_df = pd.DataFrame({
                    "Number": [round(k, 2) for k in frequency_dict.keys()],
                    "Frequency": list(frequency_dict.values())
                }).sort_values("Number").reset_index(drop=True)
                freq_df.index += 1
                st.subheader("Numerical Frequency Table")
                st.table(freq_df)

                raw_intervals = []
                raw_frequencies = []
                raw_relative_frequencies = []
                interval_edges = []
                first = True
                start = smallest

                while start < largest:
                    end = start + interval_length
                    if first:
                        label = f"{round(start, 2)} ≤ x ≤ {round(end, 2)}"
                        count = sum(start <= x <= end for x in data)
                        first = False
                    else:
                        label = f"{round(start, 2)} < x ≤ {round(end, 2)}"
                        count = sum(start < x <= end for x in data)

                    raw_intervals.append(label)
                    raw_frequencies.append(count)
                    raw_relative_frequencies.append(round(count / len(data), 4))
                    interval_edges.append((start, end))
                    start = end

                zipped = list(zip(raw_intervals, raw_frequencies, raw_relative_frequencies))
                zipped_sorted = sorted(zipped, key=lambda x: x[1], reverse=True)
                intervals_sorted, frequencies_sorted, relative_frequencies_sorted = zip(*zipped_sorted)

                interval_df = pd.DataFrame({
                    "Interval": intervals_sorted,
                    "Frequency": frequencies_sorted,
                    "Relative Frequency": relative_frequencies_sorted
                }).reset_index(drop=True)
                interval_df.index += 1
                st.subheader("Interval Frequency Table (Sorted by Frequency)")
                st.table(interval_df)

                cumulative_freq = np.cumsum(frequencies_sorted)
                cumulative_percentages = (cumulative_freq / cumulative_freq[-1]) * 100
                pareto_df = pd.DataFrame({
                    "Interval": intervals_sorted,
                    "Cumulative Frequency": cumulative_freq,
                    "Cumulative %": cumulative_percentages
                }).reset_index(drop=True)
                pareto_df.index += 1
                st.subheader("Pareto Table")
                st.table(pareto_df)

                fig, ax1 = plt.subplots(figsize=(8, 5))
                ax1.bar(intervals_sorted, frequencies_sorted, color='skyblue', alpha=0.7)
                ax1.set_xlabel("Intervals")
                ax1.set_ylabel("Frequency", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

                ax2 = ax1.twinx()
                ax2.plot(intervals_sorted, cumulative_percentages, color='red', marker='o', linestyle='-')
                ax2.set_ylabel("Cumulative Percentage (%)", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 110)

                plt.title("Pareto Chart (Numerical Intervals)")
                plt.grid(True, linestyle="--", alpha=0.7)
                st.subheader("Pareto Chart")
                st.pyplot(fig)

                bin_edges = [edge[0] for edge in interval_edges] + [interval_edges[-1][1]]
                plt.figure(figsize=(8,5))
                plt.hist(data, bins=bin_edges, edgecolor='black', color='lightgreen', alpha=0.7)
                plt.xlabel("Intervals")
                plt.ylabel("Frequency")
                plt.title("Histogram")
                plt.xticks(rotation=45)
                st.subheader("Histogram")
                st.pyplot(plt)

            except Exception as e:
                st.error(f"❌ Error processing numerical data: {str(e)}")
        else:
            if selected_type == "Numerical":
                st.warning("Please enter a valid number of intervals.")
    else:
        st.info("Please provide data using the sidebar above.")
