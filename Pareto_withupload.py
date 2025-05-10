import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

st.title("Frequency & Pareto Chart Generator")

if "started" not in st.session_state:
    st.session_state.started = False

if st.button("Start"):
    st.session_state.started = True

if st.session_state.started:
    uploaded_file = st.file_uploader("Upload Excel file or enter numbers manually:", type=["xlsx"])
    data = []

    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=0)  # Read with headers
        shape = df.shape

        if shape[1] == 1:  # One column
            column = df.columns[0]
            data = df[column].dropna().tolist()
        elif shape[0] == 1:  # One row
            data = df.iloc[0].dropna().tolist()
        else:
            selected_column = st.selectbox("Multiple columns detected. Select the column to use:", df.columns)
            data = df[selected_column].dropna().tolist()

        data = [float(x) for x in data if isinstance(x, (int, float))]  # Clean non-numeric
        user_input = ",".join(map(str, data))  # For re-use in manual logic

    else:
        user_input = st.text_input("Enter numbers (comma-separated):", "")

    num_intervals = st.text_input("Enter number of intervals:", "")

    if st.button("Finish"):
        try:
            # Parse input if not from file
            if not uploaded_file:
                data = list(map(float, user_input.split(",")))

            interval = int(num_intervals)

            largest = max(data)
            smallest = min(data)
            Diff = largest - smallest
            interval_length = math.ceil(Diff / interval)

            frequency_dict = {}
            for num in data:
                frequency_dict[num] = frequency_dict.get(num, 0) + 1

            numbers = [round(number, 2) for number, _ in sorted(frequency_dict.items())]
            freq_values = [freq for _, freq in sorted(frequency_dict.items())]

            freq_df = pd.DataFrame({
                "Number": numbers,
                "Frequency": freq_values
            })
            st.subheader("User Input Frequency Table")
            st.table(freq_df)

            intervals = []
            frequencies = []
            relative_frequencies = []

            start = smallest
            while start < largest:
                end = start + interval_length
                intervals.append(f"{round(start, 2)} < x <= {round(end, 2)}")
                count = sum(start < num <= end for num in data)
                frequencies.append(count)
                relative_frequencies.append(round(count / interval_length, 4))
                start += interval_length

            interval_df = pd.DataFrame({
                "Interval": intervals,
                "Frequency": frequencies,
                "Relative Frequency": relative_frequencies
            })
            st.subheader("Interval Frequency Table")
            st.table(interval_df)

            cumulative_frequencies = [sum(frequencies[:i + 1]) for i in range(len(frequencies))]
            cumulative_percentages = [round((cf / sum(frequencies)) * 100, 2) for cf in cumulative_frequencies]

            pareto_df = pd.DataFrame({
                "Interval": intervals,
                "Cumulative Frequency": cumulative_frequencies,
                "Cumulative %": cumulative_percentages
            })
            st.subheader("Pareto (Cumulative Frequency Table)")
            st.table(pareto_df)

            sorted_intervals = intervals
            sorted_frequencies = frequencies

            cumulative_sum = np.cumsum(sorted_frequencies)
            cumulative_percentages = (cumulative_sum / cumulative_sum[-1]) * 100

            fig, ax1 = plt.subplots(figsize=(7, 5))
            ax1.bar(sorted_intervals, sorted_frequencies, color='skyblue', alpha=0.7)
            ax1.set_xlabel('Intervals')
            ax1.set_ylabel('Frequency', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_xticklabels(sorted_intervals, rotation=45, ha='right')

            ax2 = ax1.twinx()
            ax2.plot(sorted_intervals, cumulative_percentages, color='red', marker='o')
            ax2.set_ylabel('Cumulative Percentage', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(0, 110)

            plt.title('Pareto Plot of Frequency Distribution')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            st.subheader("Pareto Chart")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}. Please make sure to provide valid inputs.")
