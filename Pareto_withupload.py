import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Frequency & Pareto Chart Generator", layout="wide")
st.title("📊 Frequency & Pareto Chart Generator")

# Initialize session state
if "started" not in st.session_state:
    st.session_state.started = False

# Start button
if not st.session_state.started:
    if st.button("Start"):
        st.session_state.started = True
        st.rerun()

# Main app
if st.session_state.started:
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        column_name = st.text_input("Enter the column name to analyze")

    if uploaded_file is not None and column_name:
        try:
            df = pd.read_csv(uploaded_file)
            if column_name in df.columns:
                data = df[column_name].value_counts().reset_index()
                data.columns = [column_name, "Frequency"]
                data["Cumulative"] = data["Frequency"].cumsum()
                data["Cumulative %"] = 100 * data["Cumulative"] / data["Frequency"].sum()

                fig = go.Figure()

                # Bar Chart
                fig.add_trace(go.Bar(
                    x=data[column_name],
                    y=data["Frequency"],
                    name="Frequency",
                    marker=dict(color="skyblue")
                ))

                # Line Chart for Cumulative %
                fig.add_trace(go.Scatter(
                    x=data[column_name],
                    y=data["Cumulative %"],
                    name="Cumulative %",
                    yaxis="y2",
                    mode="lines+markers",
                    marker=dict(color="crimson")
                ))

                # Layout
                fig.update_layout(
                    title="Pareto Chart",
                    xaxis=dict(title=column_name),
                    yaxis=dict(title="Frequency"),
                    yaxis2=dict(
                        title="Cumulative %",
                        overlaying="y",
                        side="right",
                        range=[0, 110]
                    ),
                    legend=dict(x=0.8, y=1.15),
                    margin=dict(l=40, r=40, t=80, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Column '{column_name}' not found in uploaded CSV.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file and enter a valid column name.")
