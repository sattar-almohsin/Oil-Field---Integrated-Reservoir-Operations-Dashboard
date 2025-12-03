"""
Integrated Reservoir & Operations Dashboard - Volve Field

A professional Streamlit application for reservoir surveillance and well operations
analysis using the Volve field production dataset. This dashboard demonstrates
understanding of reservoir engineering KPIs, production optimization, and
intervention candidate identification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (load_daily_production, load_monthly_production,
                             get_well_list, get_date_range, get_well_types)
from src.kpi_calculations import (compute_well_daily_kpis,
                                  compute_well_summary,
                                  rank_intervention_candidates,
                                  get_field_totals)
from src.visualizations import (
    plot_well_time_series, plot_well_time_series_with_rolling,
    plot_field_overview, plot_well_bar_kpis, plot_oil_loss_comparison,
    plot_water_cut_comparison, plot_intervention_scores,
    plot_well_performance_analysis, plot_cumulative_production,
    plot_dca_forecast, plot_eur_comparison, plot_decline_rate_comparison,
    plot_alerts_summary, plot_well_alerts_timeline)
from src.dca_analysis import (run_dca_analysis, get_dca_summary_all_wells)
from src.alerts import (get_all_alerts, get_alert_summary)
from src.pdf_reports import (generate_surveillance_report,
                             generate_intervention_report, generate_dca_report,
                             generate_executive_summary)

st.set_page_config(page_title="Volve Field",
                   page_icon="🛢️",
                   layout="wide",
                   initial_sidebar_state="expanded")

DATA_PATH = "data/Volve_production_data.xlsx"


def check_data_exists():
    """Check if the data file exists and return appropriate message."""
    if not os.path.exists(DATA_PATH):
        return False, f"Data file not found at: {DATA_PATH}"
    return True, "Data loaded successfully"


@st.cache_data
def load_data(file_path: str = DATA_PATH):
    """Load and cache all production data."""
    daily_df = load_daily_production(file_path)
    monthly_df = load_monthly_production(file_path)
    return daily_df, monthly_df


@st.cache_data
def load_uploaded_data(file_bytes, file_name: str):
    """Load data from uploaded file."""
    import io
    excel_file = io.BytesIO(file_bytes)
    daily_df = load_daily_production(excel_file)
    monthly_df = load_monthly_production(excel_file)
    return daily_df, monthly_df


def render_sidebar():
    """Render the sidebar with navigation and branding."""
    uploaded_file = None

    with st.sidebar:
        st.title("🛢️ Volve Field Dashboard")

        st.markdown("---")

        with st.expander("📁 Upload Custom Data", expanded=False):
            st.markdown(
                "Upload your own production data Excel file with the same format as the Volve dataset."
            )
            uploaded_file = st.file_uploader(
                "Choose an Excel file",
                type=['xlsx', 'xls'],
                help=
                "Excel file must have 'Daily Production Data' and 'Monthly Production Data' sheets with matching column headers"
            )

            if uploaded_file is not None:
                st.success(f"Using: {uploaded_file.name}")
            else:
                st.info("Using default Volve dataset")

        st.markdown("---")

        page = st.radio("Navigation", [
            "Overview", "Reservoir & Field Surveillance",
            "Well & Operations Performance", "Intervention Candidates",
            "Decline Curve Analysis", "Alerts & Monitoring", "Reports & Export"
        ],
                        label_visibility="collapsed")

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: gray; font-size: 12px;'>
            Wells Data Dashboard <br>
            for management and analysis
            </div>
            """,
                    unsafe_allow_html=True)

    return page, uploaded_file


def render_overview_page(daily_df: pd.DataFrame,
                         well_summary_df: pd.DataFrame):
    """Render the Overview page with high-level KPIs."""
    st.header("Field Overview")

    st.markdown("""
    This overview provides a snapshot of the Volve field's current production status.
    As a Production Engineer, monitoring these key metrics helps identify
    overall field health and prioritize operational focus areas.
    """)

    total_oil_rate = well_summary_df['LATEST_OIL_RATE'].sum()
    total_water_rate = well_summary_df['LATEST_WATER_RATE'].sum()

    well_types_df = get_well_types(daily_df)
    if 'WELL_TYPE' in well_types_df.columns:
        producers = len(well_types_df[well_types_df['WELL_TYPE'].isin(
            ['OP', 'PRODUCER', 'OIL'])])
        injectors = len(well_types_df[well_types_df['WELL_TYPE'].isin(
            ['WI', 'GI', 'INJECTOR'])])
    else:
        producers = len(well_types_df)
        injectors = 0

    underperforming_count = well_summary_df['UNDERPERFORMING'].sum()
    total_wells = len(well_summary_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Field Oil Rate",
                  value=f"{total_oil_rate:,.0f} BPD",
                  help="Sum of latest oil rates from all producing wells")

    with col2:
        st.metric(label="Producing Wells",
                  value=f"{producers}",
                  delta=f"{injectors} injectors" if injectors > 0 else None,
                  help="Number of active producing wells vs injectors")

    with col3:
        st.metric(label="Underperforming Wells",
                  value=f"{underperforming_count}",
                  delta=f"of {total_wells} total",
                  delta_color="inverse",
                  help="Wells producing <70% of their 12-month average")

    with col4:
        field_water_cut = total_water_rate / (total_oil_rate +
                                              total_water_rate) * 100 if (
                                                  total_oil_rate +
                                                  total_water_rate) > 0 else 0
        st.metric(label="Field Water Cut",
                  value=f"{field_water_cut:.1f}%",
                  help="Overall field water cut percentage")

    st.subheader("Field Production History")
    fig_overview = plot_field_overview(daily_df)
    st.plotly_chart(fig_overview, use_container_width=True)

    st.subheader("Well Summary Table")
    display_cols = [
        'NPD_WELL_BORE_NAME', 'LATEST_OIL_RATE', 'LATEST_WATER_RATE',
        'LATEST_WATER_CUT', 'CUM_OIL', 'UNDERPERFORMING'
    ]
    display_df = well_summary_df[display_cols].copy()
    display_df.columns = [
        'Well', 'Oil Rate (BPD)', 'Water Rate (BPD)', 'Water Cut',
        'Cum Oil (bbl)', 'Underperforming'
    ]
    display_df['Water Cut'] = (display_df['Water Cut'] *
                               100).round(1).astype(str) + '%'
    display_df['Oil Rate (BPD)'] = display_df['Oil Rate (BPD)'].round(1)
    display_df['Water Rate (BPD)'] = display_df['Water Rate (BPD)'].round(1)
    display_df['Cum Oil (bbl)'] = display_df['Cum Oil (bbl)'].round(0)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_surveillance_page(daily_df: pd.DataFrame, monthly_df: pd.DataFrame,
                             well_summary_df: pd.DataFrame):
    """Render the Reservoir & Field Surveillance page."""
    st.header("Reservoir & Field Surveillance")

    st.markdown("""
    Individual well surveillance is critical for understanding reservoir drainage and 
    optimizing production. This page allows to analyze each well's production 
    history and water cut evolution.
    """)

    well_list = get_well_list(daily_df)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_well = st.selectbox("Select Well",
                                     well_list,
                                     key="surveillance_well")
    with col2:
        resolution = st.radio("Time Resolution", ["Daily", "Monthly"],
                              horizontal=True)

    if selected_well:
        well_data = well_summary_df[well_summary_df['NPD_WELL_BORE_NAME'] ==
                                    selected_well].iloc[0]

        st.subheader(f"Well KPIs - {selected_well}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Cumulative Oil",
                      value=f"{well_data['CUM_OIL']:,.0f} bbl")

        with col2:
            st.metric(label="Cumulative Gas",
                      value=f"{well_data['CUM_GAS']:,.0f} Sm3")

        with col3:
            st.metric(label="Cumulative Water",
                      value=f"{well_data['CUM_WATER']:,.0f} bbl")

        with col4:
            st.metric(label="Latest Water Cut",
                      value=f"{well_data['LATEST_WATER_CUT']*100:.1f}%",
                      help="Water Cut = Water Rate / (Oil Rate + Water Rate)")

        st.subheader("Production Time Series")

        if resolution == "Daily":
            fig_ts = plot_well_time_series(daily_df, selected_well)
        else:
            well_monthly = monthly_df[monthly_df['NPD_WELL_BORE_NAME'] ==
                                      selected_well].copy()
            if len(well_monthly) > 0 and 'Oil' in well_monthly.columns:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig_ts = make_subplots(rows=1, cols=1)
                fig_ts.add_trace(
                    go.Scatter(x=well_monthly['DATE'],
                               y=well_monthly['Oil'],
                               name='Monthly Oil',
                               line=dict(color='green', width=2)))
                fig_ts.update_layout(
                    title=f'Monthly Production - {selected_well}',
                    xaxis_title='Date',
                    yaxis_title='Oil Volume (Sm3)',
                    height=400)
            else:
                fig_ts = plot_well_time_series(daily_df, selected_well)

        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Cumulative Production")
        fig_cum = plot_cumulative_production(daily_df, selected_well)
        st.plotly_chart(fig_cum, use_container_width=True)


def render_performance_page(daily_df: pd.DataFrame,
                            well_summary_df: pd.DataFrame):
    """Render the Well & Operations Performance page."""
    st.header("Well & Operations Performance")

    st.markdown("""
    This surveillance screen helps identify wells that are underperforming relative to 
    their historical production. Oil loss calculations compare current rates against 
    12-month averages to highlight potential intervention opportunities.
    """)

    st.subheader("Well Performance Summary")

    display_cols = [
        'NPD_WELL_BORE_NAME', 'LATEST_OIL_RATE', 'AVG_OIL_RATE_6M',
        'AVG_OIL_RATE_12M', 'OIL_LOSS', 'LATEST_WATER_CUT', 'UNDERPERFORMING'
    ]
    display_df = well_summary_df[display_cols].copy()
    display_df.columns = [
        'Well', 'Current Rate', '6M Avg Rate', '12M Avg Rate', 'Oil Loss',
        'Water Cut', 'Underperforming'
    ]

    for col in ['Current Rate', '6M Avg Rate', '12M Avg Rate', 'Oil Loss']:
        display_df[col] = display_df[col].round(1)
    display_df['Water Cut'] = (display_df['Water Cut'] *
                               100).round(1).astype(str) + '%'

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Oil Loss by Well")
        fig_loss = plot_oil_loss_comparison(well_summary_df)
        st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        st.subheader("Water Cut by Well")
        fig_wc = plot_water_cut_comparison(well_summary_df)
        st.plotly_chart(fig_wc, use_container_width=True)

    st.subheader("Detailed Well Analysis")

    well_list = get_well_list(daily_df)
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_well = st.selectbox("Select Well for Analysis",
                                     well_list,
                                     key="perf_well")

    with col2:
        rolling_window = st.slider("Rolling Average (days)", 7, 90, 30)

    if selected_well:
        fig_rolling = plot_well_time_series_with_rolling(
            daily_df, selected_well, rolling_window)
        st.plotly_chart(fig_rolling, use_container_width=True)


def render_intervention_page(daily_df: pd.DataFrame,
                             well_summary_df: pd.DataFrame):
    """Render the Intervention Candidates page."""
    st.header("Intervention Candidates")

    st.markdown("""
    This page ranks wells by intervention priority based on multiple factors:
    - **Oil Loss**: Difference between 12-month average and current production
    - **Water Cut**: Higher water cut may indicate water breakthrough requiring conformance work
    - **Downtime**: Frequent zero-production days suggest operational or mechanical issues
    
    Adjust the weights below to prioritize different factors based on your operational focus.
    """)

    st.subheader("Scoring Weights")
    col1, col2, col3 = st.columns(3)

    with col1:
        weight_oil_loss = st.slider("Oil Loss Weight", 0.0, 1.0, 0.5, 0.1)
    with col2:
        weight_water_cut = st.slider("Water Cut Weight", 0.0, 1.0, 0.3, 0.1)
    with col3:
        weight_downtime = st.slider("Downtime Weight", 0.0, 1.0, 0.2, 0.1)

    intervention_df = rank_intervention_candidates(
        well_summary_df,
        daily_df,
        weight_oil_loss=weight_oil_loss,
        weight_water_cut=weight_water_cut,
        weight_downtime=weight_downtime)

    st.subheader("Top 3 Intervention Candidates")

    display_df = intervention_df.head(3).copy()
    display_df.columns = [
        'Well', 'Current Rate', '12M Avg Rate', 'Oil Loss', 'Water Cut',
        'Downtime Ratio', 'Score', 'Recommended Action'
    ]

    for col in ['Current Rate', '12M Avg Rate', 'Oil Loss']:
        display_df[col] = display_df[col].round(1)
    display_df['Water Cut'] = (display_df['Water Cut'] *
                               100).round(1).astype(str) + '%'
    display_df['Downtime Ratio'] = (display_df['Downtime Ratio'] *
                                    100).round(1).astype(str) + '%'
    display_df['Score'] = display_df['Score'].round(3)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Priority Score Distribution")
        fig_scores = plot_intervention_scores(intervention_df)
        st.plotly_chart(fig_scores, use_container_width=True)

    with col2:
        st.subheader("Top Candidate Analysis")

        if len(intervention_df) > 0:
            top_candidates = intervention_df.head(
                5)['NPD_WELL_BORE_NAME'].tolist()
            selected_candidate = st.selectbox(
                "Select Candidate for Detailed Analysis",
                top_candidates,
                key="intervention_candidate")

            if selected_candidate:
                candidate_action = intervention_df[
                    intervention_df['NPD_WELL_BORE_NAME'] ==
                    selected_candidate]['RECOMMENDED_ACTION'].iloc[0]

                st.info(f"**Recommended Action:** {candidate_action}")

    if selected_candidate:
        st.subheader(f"Performance Analysis - {selected_candidate}")
        fig_analysis = plot_well_performance_analysis(daily_df,
                                                      selected_candidate,
                                                      months=12)
        st.plotly_chart(fig_analysis, use_container_width=True)

        st.markdown("""
        **Interpretation Guide:**
        - Red bars in on-stream hours indicate periods with less than 12 hours of production (potential downtime)
        - Orange bars indicate reduced runtime (12-20 hours)
        - Green bars indicate normal operation (>20 hours)
        - The dashed red line shows the average oil rate for the period
        """)


def render_dca_page(daily_df: pd.DataFrame, well_summary_df: pd.DataFrame):
    """Render the Decline Curve Analysis page."""
    st.header("Decline Curve Analysis (DCA)")

    st.markdown("""
    Decline Curve Analysis uses historical production data to forecast future production 
    and estimate ultimate recovery (EUR). This analysis helps as reservoir engineers understand 
    well depletion patterns and plan field development strategies.
    
    **Models Available:**
    - **Exponential Decline**: q(t) = qi * exp(-Di * t) - Constant percentage decline
    - **Hyperbolic Decline**: q(t) = qi / (1 + b*Di*t)^(1/b) - Decline rate decreases over time
    """)

    st.subheader("Analysis Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        well_list = get_well_list(daily_df)
        selected_well = st.selectbox("Select Well", well_list, key="dca_well")

    with col2:
        forecast_months = st.slider("Forecast Period (months)", 12, 120, 60)

    with col3:
        economic_limit = st.number_input("Economic Limit (Sm3/d)", 1.0, 100.0,
                                         10.0)

    if selected_well:
        with st.spinner("Running decline curve analysis..."):
            dca_results = run_dca_analysis(daily_df, selected_well,
                                           forecast_months, economic_limit)

        if dca_results['success']:
            st.subheader(f"DCA Results - {selected_well}")

            col1, col2, col3, col4 = st.columns(4)

            best_model = dca_results['best_fit']
            best_data = dca_results[best_model]

            with col1:
                st.metric("Best Fit Model",
                          best_model.title(),
                          help="Model with highest R-squared value")

            with col2:
                st.metric("Initial Rate (qi)",
                          f"{best_data['qi']:.1f} Sm3/d",
                          help="Fitted initial production rate")

            with col3:
                st.metric("Decline Rate",
                          f"{best_data['decline_rate_pct']:.2f}%/month",
                          help="Monthly decline rate")

            with col4:
                eur_mSm3 = best_data['eur'] / 1000 if not np.isnan(
                    best_data['eur']) else 0
                st.metric("EUR",
                          f"{eur_mSm3:.1f} MSm3",
                          help="Estimated Ultimate Recovery")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Exponential Model**")
                exp_data = dca_results['exponential']
                if not np.isnan(exp_data['qi']):
                    st.write(f"- qi: {exp_data['qi']:.1f} Sm3/d")
                    st.write(
                        f"- Di: {exp_data['decline_rate_pct']:.3f}%/month")
                    st.write(f"- R²: {exp_data['r_squared']:.4f}")
                    st.write(f"- EUR: {exp_data['eur']/1000:.1f} MSm3")
                else:
                    st.write("Could not fit exponential model")

            with col2:
                st.markdown("**Hyperbolic Model**")
                hyp_data = dca_results['hyperbolic']
                if not np.isnan(hyp_data['qi']):
                    st.write(f"- qi: {hyp_data['qi']:.1f} Sm3/d")
                    st.write(
                        f"- Di: {hyp_data['decline_rate_pct']:.3f}%/month")
                    st.write(f"- b-factor: {hyp_data['b']:.3f}")
                    st.write(f"- R²: {hyp_data['r_squared']:.4f}")
                    st.write(f"- EUR: {hyp_data['eur']/1000:.1f} MSm3")
                else:
                    st.write("Could not fit hyperbolic model")

            st.subheader("Decline Curve Plot")
            fig_dca = plot_dca_forecast(dca_results)
            st.plotly_chart(fig_dca, use_container_width=True)

        else:
            st.warning(dca_results.get('error', 'DCA analysis failed'))

    st.subheader("Field-Wide DCA Summary")

    with st.spinner("Calculating DCA for all wells..."):
        dca_summary = get_dca_summary_all_wells(daily_df, economic_limit)

    display_df = dca_summary.copy()
    display_df = display_df[display_df['EUR'].notna()]

    if len(display_df) > 0:
        display_df['QI_RATE'] = display_df['QI_RATE'].round(1)
        display_df['DECLINE_RATE_PCT'] = display_df['DECLINE_RATE_PCT'].round(
            2)
        display_df['R_SQUARED'] = display_df['R_SQUARED'].round(4)
        display_df['CURRENT_CUM'] = (display_df['CURRENT_CUM'] / 1000).round(1)
        display_df['EUR'] = (display_df['EUR'] / 1000).round(1)
        display_df['REMAINING_RESERVES'] = (display_df['REMAINING_RESERVES'] /
                                            1000).round(1)

        display_df.columns = [
            'Well', 'Best Model', 'qi (Sm3/d)', 'Decline (%/mo)', 'b-factor',
            'R²', 'Cum Oil (MSm3)', 'EUR (MSm3)', 'Remaining (MSm3)'
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_eur = plot_eur_comparison(dca_summary)
            st.plotly_chart(fig_eur, use_container_width=True)

        with col2:
            fig_decline = plot_decline_rate_comparison(dca_summary)
            st.plotly_chart(fig_decline, use_container_width=True)
    else:
        st.info("Not enough data to calculate DCA for wells.")


def render_alerts_page(daily_df: pd.DataFrame, well_summary_df: pd.DataFrame):
    """Render the Alerts & Monitoring page with data refresh and alerts."""
    st.header("Alerts & Monitoring")

    st.markdown("""
    This page provides real-time monitoring of production alerts and data refresh capabilities.
    Alerts are automatically generated based on configurable thresholds for critical production changes.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Alert Configuration")

    with col2:
        if st.button("Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    col1, col2, col3 = st.columns(3)

    with col1:
        rate_drop_threshold = st.slider(
            "Rate Drop Threshold (%)",
            10,
            70,
            30,
            help="Alert when oil rate drops by this percentage")

    with col2:
        water_cut_threshold = st.slider(
            "Water Cut Spike (%)",
            5,
            50,
            10,
            help="Alert when water cut increases by this percentage")

    with col3:
        st.markdown("**Last Data Update**")
        min_date, max_date = get_date_range(daily_df)
        st.write(f"Data through: {max_date.strftime('%Y-%m-%d')}")

    with st.spinner("Analyzing production data for alerts..."):
        alerts_df = get_all_alerts(daily_df, well_summary_df,
                                   rate_drop_threshold, water_cut_threshold)
        alert_summary = get_alert_summary(alerts_df)

    st.subheader("Alert Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Alerts",
                  alert_summary['total_alerts'],
                  help="Total number of active alerts")

    with col2:
        st.metric("High Severity",
                  alert_summary['high_severity'],
                  delta=None if alert_summary['high_severity'] == 0 else
                  "requires action",
                  delta_color="inverse"
                  if alert_summary['high_severity'] > 0 else "normal")

    with col3:
        st.metric("Medium Severity", alert_summary['medium_severity'])

    with col4:
        st.metric("Wells Affected", alert_summary['wells_affected'])

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_summary = plot_alerts_summary(alerts_df)
        st.plotly_chart(fig_summary, use_container_width=True)

    with col2:
        if len(alert_summary['alerts_by_type']) > 0:
            st.markdown("**Alerts by Type:**")
            for alert_type, count in alert_summary['alerts_by_type'].items():
                st.write(f"- {alert_type.replace('_', ' ').title()}: {count}")

    if len(alerts_df) > 0:
        st.subheader("Active Alerts")

        severity_filter = st.multiselect("Filter by Severity",
                                         options=['HIGH', 'MEDIUM', 'LOW'],
                                         default=['HIGH', 'MEDIUM'])

        filtered_alerts = alerts_df[alerts_df['severity'].isin(
            severity_filter)]

        for _, alert in filtered_alerts.iterrows():
            severity_icon = "🔴" if alert[
                'severity'] == 'HIGH' else "🟠" if alert[
                    'severity'] == 'MEDIUM' else "🟡"

            with st.expander(
                    f"{severity_icon} {alert['well']} - {alert['type'].replace('_', ' ').title()}",
                    expanded=(alert['severity'] == 'HIGH')):
                st.markdown(f"**Message:** {alert['message']}")
                st.markdown(f"**Recommended Action:** {alert['action']}")

                if alert['date'] is not None:
                    st.markdown(
                        f"**Date Detected:** {alert['date'].strftime('%Y-%m-%d') if hasattr(alert['date'], 'strftime') else alert['date']}"
                    )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Value", f"{alert['current_value']:.1f}")
                with col2:
                    st.metric("Previous Value",
                              f"{alert['previous_value']:.1f}")

        st.subheader("Well Alert Analysis")

        affected_wells = alerts_df['well'].unique().tolist()
        selected_alert_well = st.selectbox("Select Well for Detailed Analysis",
                                           affected_wells,
                                           key="alert_well_select")

        if selected_alert_well:
            fig_timeline = plot_well_alerts_timeline(daily_df,
                                                     selected_alert_well,
                                                     alerts_df)
            st.plotly_chart(fig_timeline, use_container_width=True)

        st.subheader("Download Alerts")

        csv_alerts = alerts_df.to_csv(index=False)
        st.download_button(label="Download Alerts Report (CSV)",
                           data=csv_alerts,
                           file_name="volve_production_alerts.csv",
                           mime="text/csv")
    else:
        st.success(
            "No active alerts - all wells operating within normal parameters.")

    st.markdown("---")
    st.markdown("""
    **Alert Types Explained:**
    - **Rate Drop**: Significant decrease in oil production rate compared to recent average
    - **Water Cut Spike**: Sudden increase in water production fraction
    - **High Water Cut**: Water cut exceeds critical threshold (80%+)
    - **GOR Anomaly**: Significant change in gas-oil ratio indicating reservoir changes
    - **Extended Downtime**: Well has low on-stream hours for multiple consecutive days
    - **Underperformance**: Well producing significantly below 12-month historical average
    """)


def render_reports_page(daily_df: pd.DataFrame, well_summary_df: pd.DataFrame):
    """Render the Reports & Export page with PDF report generation."""
    st.header("Reports & Export")

    st.markdown("""
    Generate professional PDF reports for reservoir surveillance, intervention planning,
    and management briefings.
    """)

    st.subheader("Available Reports")

    intervention_df = rank_intervention_candidates(well_summary_df, daily_df)
    alerts_df = get_all_alerts(daily_df, well_summary_df)

    with st.spinner("Preparing DCA summary..."):
        dca_summary_df = get_dca_summary_all_wells(daily_df, 10.0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Executive Summary")
        st.markdown("""
        Comprehensive overview including:
        - Production overview
        - Key alerts and issues
        - Recommended interventions
        - Reserves outlook
        """)

        if st.button("Generate Executive Summary", key="exec_report"):
            with st.spinner("Generating report..."):
                pdf_bytes = generate_executive_summary(daily_df,
                                                       well_summary_df,
                                                       intervention_df,
                                                       dca_summary_df,
                                                       alerts_df)
                st.download_button(
                    label="Download Executive Summary (PDF)",
                    data=pdf_bytes,
                    file_name=
                    f"volve_executive_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf")

    with col2:
        st.markdown("### Surveillance Report")
        st.markdown("""
        Field surveillance details:
        - Well production summary
        - Active alerts listing
        - Performance metrics
        """)

        if st.button("Generate Surveillance Report", key="surv_report"):
            with st.spinner("Generating report..."):
                pdf_bytes = generate_surveillance_report(
                    daily_df, well_summary_df, alerts_df)
                st.download_button(
                    label="Download Surveillance Report (PDF)",
                    data=pdf_bytes,
                    file_name=
                    f"volve_surveillance_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Intervention Report")
        st.markdown("""
        Intervention recommendations:
        - Priority ranking
        - Recommended actions
        - Well-by-well details
        """)

        if st.button("Generate Intervention Report", key="int_report"):
            with st.spinner("Generating report..."):
                econ_df = analyze_all_intervention_candidates(
                    well_summary_df, intervention_df, 75.0, 0.10)
                pdf_bytes = generate_intervention_report(
                    intervention_df, econ_df)
                st.download_button(
                    label="Download Intervention Report (PDF)",
                    data=pdf_bytes,
                    file_name=
                    f"volve_intervention_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf")

    with col4:
        st.markdown("### DCA Report")
        st.markdown("""
        Decline curve analysis:
        - EUR estimates
        - Remaining reserves
        - Decline parameters
        """)

        if st.button("Generate DCA Report", key="dca_report"):
            with st.spinner("Generating report..."):
                pdf_bytes = generate_dca_report(dca_summary_df)
                st.download_button(
                    label="Download DCA Report (PDF)",
                    data=pdf_bytes,
                    file_name=
                    f"volve_dca_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf")

    st.markdown("---")
    st.subheader("Data Export Options")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv_daily = daily_df.to_csv(index=False)
        st.download_button(label="Daily Production (CSV)",
                           data=csv_daily,
                           file_name="volve_daily_production.csv",
                           mime="text/csv")

    with col2:
        csv_summary = well_summary_df.to_csv(index=False)
        st.download_button(label="Well Summary (CSV)",
                           data=csv_summary,
                           file_name="volve_well_summary.csv",
                           mime="text/csv")

    with col3:
        csv_intervention = intervention_df.to_csv(index=False)
        st.download_button(label="Interventions (CSV)",
                           data=csv_intervention,
                           file_name="volve_interventions.csv",
                           mime="text/csv")

    with col4:
        csv_dca = dca_summary_df.to_csv(index=False)
        st.download_button(label="DCA Summary (CSV)",
                           data=csv_dca,
                           file_name="volve_dca_summary.csv",
                           mime="text/csv")

    st.markdown("---")
    st.markdown("""
    **Report Types Explained:**
    - **Executive Summary**: High-level overview for management, covering key metrics and recommendations
    - **Surveillance Report**: Detailed well-by-well production data and active alerts for operations team
    - **Intervention Report**: Prioritized list of recommended well interventions with economic analysis
    - **DCA Report**: Decline curve analysis results with EUR and remaining reserves estimates
    """)


def main():
    """Main application entry point."""
    page, uploaded_file = render_sidebar()

    try:
        if uploaded_file is not None:
            daily_df, monthly_df = load_uploaded_data(uploaded_file.getvalue(),
                                                      uploaded_file.name)
        else:
            data_exists, message = check_data_exists()

            if not data_exists:
                st.error(f"⚠️ {message}")
                st.info("""
                Please place the Volve production data Excel file at:
                
                `data/Volve_production_data.xlsx`
                
                Or upload your own dataset using the sidebar.
                
                The file should contain sheets named:
                - "Daily Production Data"
                - "Monthly Production Data"
                """)
                return

            daily_df, monthly_df = load_data()

        well_summary_df = compute_well_summary(daily_df)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info(
            "Please ensure the Excel file has the correct format with 'Daily Production Data' and 'Monthly Production Data' sheets."
        )
        return

    if page == "Overview":
        render_overview_page(daily_df, well_summary_df)
    elif page == "Reservoir & Field Surveillance":
        render_surveillance_page(daily_df, monthly_df, well_summary_df)
    elif page == "Well & Operations Performance":
        render_performance_page(daily_df, well_summary_df)
    elif page == "Intervention Candidates":
        render_intervention_page(daily_df, well_summary_df)
    elif page == "Decline Curve Analysis":
        render_dca_page(daily_df, well_summary_df)
    elif page == "Alerts & Monitoring":
        render_alerts_page(daily_df, well_summary_df)
    elif page == "Reports & Export":
        render_reports_page(daily_df, well_summary_df)


if __name__ == "__main__":
    main()
