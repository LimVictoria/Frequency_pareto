import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

st.title("Frequency & Pareto Chart & Histogram Generator")

def get_excel_column_name(n):
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

# Sidebar for data input directly
with st.sidebar:
    st.header("Input Data")
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

    num_intervals = st.text_input("Enter number of intervals:", "")

    st.button("Finish")

# Main app part to process the data and generate the frequency & Pareto chart
col1, col2 = st.columns([1, 2])

with col2:
    if data and num_intervals.isdigit():
        try:
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
            st.subheader("User Input Frequency Table")
            st.table(freq_df)

            # Build intervals in "k < x ≤ k2" format
            intervals = []
            frequencies = []
            relative_frequencies = []

            start = smallest
            while start < largest:
                end = start + interval_length
                label = f"{round(start, 2)} < x ≤ {round(end, 2)}"
                count = sum(start < x <= end for x in data)
                intervals.append(label)
                frequencies.append(count)
                relative_frequencies.append(round(count / len(data), 4))
                start = end

            if intervals:
                interval_df = pd.DataFrame({
                    "Interval": intervals,
                    "Frequency": frequencies,
                    "Relative Frequency": relative_frequencies
                }).reset_index(drop=True)
                interval_df.index += 1

                st.subheader("Interval Frequency Table")
                st.table(interval_df)

                cumulative_freq = np.cumsum(frequencies)
                cumulative_percentages = (cumulative_freq / cumulative_freq[-1]) * 100

                pareto_df = pd.DataFrame({
                    "Interval": intervals,
                    "Cumulative Frequency": cumulative_freq,
                    "Cumulative %": cumulative_percentages
                }).reset_index(drop=True)
                pareto_df.index += 1

                st.subheader("Pareto (Cumulative Frequency Table)")
                st.table(pareto_df)

                fig, ax1 = plt.subplots(figsize=(8, 5))
                bars = ax1.bar(intervals, frequencies, color='skyblue', alpha=0.7)
                ax1.set_xlabel("Intervals")
                ax1.set_ylabel("Frequency", color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

                ax2 = ax1.twinx()
                ax2.plot(intervals, cumulative_percentages, color='red', marker='o', linestyle='-')
                ax2.set_ylabel("Cumulative Percentage (%)", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 110)

                plt.title("Pareto Chart")
                plt.grid(True, linestyle="--", alpha=0.7)
                st.subheader("Pareto Chart")
                st.pyplot(fig)

                # Generate Histogram with interval labels on x-axis
                interval_edges = []
                for interval in intervals:
                    start, end = interval.split("< x ≤ ")
                    interval_edges.append((float(start), float(end)))

                bin_edges = [start for start, _ in interval_edges] + [interval_edges[-1][1]]

                fig_hist, ax_hist = plt.subplots(figsize=(8, 5))

                # Use intervals as x-tick labels
                ax_hist.hist(data, bins=bin_edges, color='lightgreen', edgecolor='black', alpha=0.7)

                # Set x-ticks to match the interval labels
                ax_hist.set_xticks([start + (end - start) / 2 for start, end in interval_edges])  # Place tick in the middle of the interval
                ax_hist.set_xticklabels(intervals, rotation=45, ha='right')

                ax_hist.set_xlabel("Intervals")
                ax_hist.set_ylabel("Frequency")
                ax_hist.set_title("Histogram")
                plt.setp(ax_hist.get_xticklabels(), rotation=45, ha='right')
                st.subheader("Histogram")
                st.pyplot(fig_hist)

            else:
                st.warning("⚠️ No valid intervals generated. Check your data or number of intervals.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
