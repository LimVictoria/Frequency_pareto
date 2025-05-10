import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

st.set_page_config(page_title="Frequency & Pareto Chart Generator", layout="wide")
st.title("📊 Frequency & Pareto Chart Generator")

# Utility to convert column index to Excel-style letters
def get_excel_column_name(n):
    name = ''
    while n >= 0:
        name = chr(n % 26 + ord('A')) + name
        n = n // 26 - 1
    return name

# Session state to control Start screen
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    if st.button("Start"):
        st.session_state.started = True

# Main logic
if st.session_state.started:
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
                        column_mapping = {
                            get_excel_column_name(i): col
                            for i, col in enumerate(df.columns)
                            if not df[col].dropna().empty
                        }
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

    col1, col2 = st.columns([1, 2])

    with col2:
        if data and num_intervals.isdigit():
            try:
                interval = int(num_intervals)
                largest = max(data)
                smallest = min(data)
                diff = largest - smallest
                interval_length = math.ceil(diff / interval)

                # Raw frequency table
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

                # Interval construction
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

                    # Plotly Pareto Chart with hover interaction
                    fig = go.Figure()

                    # Bar chart for frequencies
                    fig.add_trace(go.Bar(
                        x=intervals,
                        y=frequencies,
                        name='Frequency',
                        marker_color='skyblue',
                        hovertemplate='Interval: %{x}<br>Frequency: %{y}<extra></extra>'
                    ))

                    # Line chart for cumulative %
                    fig.add_trace(go.Scatter(
                        x=intervals,
                        y=cumulative_percentages,
                        name='Cumulative %',
                        mode='lines+markers',
                        marker=dict(color='red'),
                        yaxis='y2',
                        hovertemplate='Interval: %{x}<br>Cumulative %: %{y:.2f}%<extra></extra>'
                    ))

                    fig.update_layout(
                        title="Pareto Chart",
                        xaxis=dict(title="Intervals"),
                        yaxis=dict(title="Frequency"),
                        yaxis2=dict(
                            title="Cumulative Percentage (%)",
                            overlaying='y',
                            side='right',
                            range=[0, 110]
                        ),
                        legend=dict(x=0.01, y=0.99),
                        margin=dict(t=40, b=40),
                        height=500
                    )

                    st.subheader("📈 Interactive Pareto Chart")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No valid intervals generated. Check your data or number of intervals.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
