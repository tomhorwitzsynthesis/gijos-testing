# utils/date_utils.py
import streamlit as st
from datetime import datetime
from calendar import month_name

# Manually define which months are available
AVAILABLE_MONTHS = [
    (2025, 5),
    (2025, 6),
    (2025, 7),
    (2025, 8),
    (2025, 9),
    (2025, 10),
]

def get_selected_date_range():
    selected = st.session_state.get("selected_months", [])
    if not selected:
        st.sidebar.error("No months selected.")
        st.stop()

    selected.sort()
    start_year, start_month = selected[0]
    end_year, end_month = selected[-1]

    start_date = datetime(start_year, start_month, 1)
    if end_month == 12:
        end_date = datetime(end_year + 1, 1, 1)
    else:
        end_date = datetime(end_year, end_month + 1, 1)

    return start_date, end_date

def init_month_selector():
    placeholder = st.sidebar.empty()
    with placeholder.container():
        st.markdown("### Select Start and End Month")

        # Prepare options for dropdowns
        options = [
            (year, month, f"{month_name[month]} {year}")
            for year, month in AVAILABLE_MONTHS
        ]
        labels = [opt[2] for opt in options]

        # Default to first and last available
        default_start = 0
        default_end = len(options) - 1

        start_idx = st.selectbox(
            "Start Month",
            options=range(len(options)),
            format_func=lambda i: labels[i],
            index=default_start,
            key="month_selector_start",
            label_visibility="collapsed",
        )
        end_idx = st.selectbox(
            "End Month",
            options=range(len(options)),
            format_func=lambda i: labels[i],
            index=default_end,
            key="month_selector_end",
            label_visibility="collapsed",
        )

        if start_idx > end_idx:
            st.error("Start month must be before or equal to end month.")
            st.stop()

        # Build selected months list
        selected = [
            (options[i][0], options[i][1])
            for i in range(start_idx, end_idx + 1)
        ]

        if not selected:
            st.error("Please select at least one month.")
            st.stop()

        st.session_state["selected_months"] = selected

    placeholder.empty()
