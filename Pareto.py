import streamlit as st
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import math

st.title("Frequency & Pareto Chart Generator")

if "started" not in st.session_state:
    st.session_state.started = False

if st.button("Start"):
    st.session_state.started = True

if st.session_state.started:
    user_input = st.text_input("Enter numbers (comma-separated):", "")
    num_intervals = st.text_input("Enter number of intervals:", "")

    if st.button("Finish"):
        try:
            # Step 1: Parse inputs
            data = list(map(int, user_input.split(",")))
            interval = int(num_intervals)

            largest = max(data)
            smallest = min(data)
            Diff = largest - smallest
            interval_length = math.ceil(Diff / interval)

            frequency_dict = {}
            for num in data:
                frequency_dict[num] = frequency_dict.get(num, 0) + 1

            # Step 2: Tabulated frequency
            numbers = [number for number, _ in sorted(frequency_dict.items())]
            freq_values = [freq for _, freq in sorted(frequency_dict.items())]

            table_data = [["Number"] + numbers, ["Frequency"] + freq_values]
            st.subheader("Tabulated Frequency")
            st.text(tabulate(table_data, tablefmt="grid"))

            # Step 3: Interval-based frequency
            intervals = []
            frequencies = []
            relative_frequencies = []

            start = smallest
            while start < largest:
                end = start + interval_length
                intervals.append(f"{start} < x <= {end}")
                count = sum(start < num <= end for num in data)
                frequencies.append(count)
                relative_frequencies.append(round(count / interval_length, 4))
                start += interval_length

            interval_data = list(zip(intervals, frequencies, relative_frequencies))
            headers = ["Interval", "Frequency", "Relative Frequency"]
            st.subheader("Interval Frequency Table")
            st.text(tabulate(interval_data, headers=headers, tablefmt="grid"))

            # Step 4: Cumulative (Pareto) table
            cumulative_frequencies = [sum(frequencies[:i + 1]) for i in range(len(frequencies))]
            cumulative_percentages = [round((cf / sum(frequencies)) * 100, 2) for cf in cumulative_frequencies]
            pareto_data = list(zip(intervals, cumulative_frequencies, cumulative_percentages))
            pareto_headers = ["Interval", "Cumulative Frequency", "Cumulative %"]
            st.subheader("Pareto (Cumulative Frequency Table)")
            st.text(tabulate(pareto_data, headers=pareto_headers, tablefmt="grid"))

            # Step 5: Pareto plot
            # Keep intervals in original order (numeric)
            sorted_intervals = intervals
            sorted_frequencies = frequencies

            # Calculate cumulative percentages
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
            st.error("❌ Invalid input. Please make sure to enter numbers only, separated by commas.")
