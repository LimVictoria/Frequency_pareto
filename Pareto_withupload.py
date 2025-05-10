import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import string

st.title("Frequency & Pareto Chart Generator")

def get_excel_column_name(n):
    """Convert numeric index to Excel-style column name."""
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

if "started" not in st.session_state:
    st.session_state.started = False

# Display "Start" button only if not started yet
if not st.session_state.started:
    if st.button("Start"):
        st.session_state.started = True
        st.experimental_rerun()  # Re-run the script to remove the "Start" button

if st.session_state.started:
    # Sidebar for the input panel
    with st.sidebar:
        st.header("Input Data")
        
        input_method = st.radio("How do you want to provide your data?", ["Upload Excel file", "Enter manually"])

        data = []

        if input_method == "Upload Excel file":
            uploaded_file = st.file_uploader("Upload Excel file:", type=["xlsx"])

            if uploaded_file:
                df = pd.read_excel(uploaded_file, header=0)

                # Drop fully empty columns
                df_cleaned = df.dropna(axis=1, how='all')

                if df_cleaned.shape[1] == 0:
                    st.error("❌ The uploaded file doesn't contain any non-empty columns.")
                else:
                    # Identify non-empty columns and map index to Excel-style name
                    column_mapping = {}
                    for i, col in enumerate(df.columns):
                        if not df[col].dropna().empty:
                            excel_col = get_excel_column_name(i)
                            column_mapping[excel_col] = col

                    if len(column_mapping) == 1:
                        selected_key = list(column_mapping.keys())[0]
                        st.info(f"Only one column with values found: Column {selected_key}")
                    else:
                        selected_key = st.selectbox("Select which column to use:", list(column_mapping.keys()))

                    selected_column = column_mapping[selected_key]
                    data = df[selected_column].dropna().tolist()

                    # Filter only numeric values
                    data = [float(x) for x in data if isinstance(x, (int, float))]

        elif input_method == "Enter manually":
            user_input = st.text_input("Enter numbers (comma-separated):", "")
            if user_input:
                try:
                    data = list(map(float, user_input.split(",")))
                except:
                    st.error("❌ Please enter valid comma-separated numbers.")

        num_intervals = st.text_input("Enter number of intervals:", "")

        st.button("Finish")

    # Main area for displaying results
    col1, col2 = st.columns([1, 2])  # Left column for input, right column for results

    with col2:
        if data and num_intervals.isdigit():
            try:
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

                # Reset index to start from 1
                freq_df = freq_df.reset_index(drop=True)
                freq_df.index += 1
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

                # Reset index to start from 1
                interval_df = interval_df.reset_index(drop=True)
                interval_df.index += 1
                st.subheader("Interval Frequency Table")
                st.table(interval_df)

                cumulative_frequencies = [sum(frequencies[:i + 1]) for i in range(len(frequencies))]
                cumulative_percentages = [round((cf / sum(frequencies)) * 100, 2) for cf in cumulative_frequencies]

                pareto_df = pd.DataFrame({
                    "Interval": intervals,
                    "Cumulative Frequency": cumulative_frequencies,
                    "Cumulative %": cumulative_percentages
                })

                # Reset index to start from 1
                pareto_df = pareto_df.reset_index(drop=True)
                pareto_df.index += 1
                st.subheader("Pareto (Cumulative Frequency Table)")
                st.table(pareto_df)

                cumulative_sum = np.cumsum(frequencies)
                cumulative_percentages = (cumulative_sum / cumulative_sum[-1]) * 100

                fig, ax1 = plt.subplots(figsize=(7, 5))
                ax1.bar(intervals, frequencies, color='skyblue', alpha=0.7)
                ax1.set_xlabel('Intervals')
                ax1.set_ylabel('Frequency', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                ax1.set_xticklabels(intervals, rotation=45, ha='right')

                ax2 = ax1.twinx()
                ax2.plot(intervals, cumulative_percentages, color='red', marker='o')
                ax2.set_ylabel('Cumulative Percentage', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(0, 110)

                plt.title('Pareto Plot of Frequency Distribution')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                st.subheader("Pareto Chart")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
