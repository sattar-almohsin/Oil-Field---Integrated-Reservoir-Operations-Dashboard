"""
Visualization module for reservoir and operations dashboards.

This module provides reusable Plotly-based visualization functions for
production time series, KPI bar charts, and field overview plots.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List


def plot_well_time_series(daily_df: pd.DataFrame,
                          well_name: str,
                          show_water_cut: bool = True,
                          show_pressure: bool = False) -> go.Figure:
    """
    Create a time series plot for a single well showing production rates.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Name of the well to plot.
        show_water_cut: Whether to show water cut on secondary axis.
        show_pressure: Whether to show wellhead pressure.
        
    Returns:
        Plotly Figure object.
    """
    well_df = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name].copy()
    well_df = well_df.sort_values('DATEPRD')

    well_df = well_df[(well_df['OIL_RATE'].notna() & (well_df['OIL_RATE'] > 0))
                      | (well_df['WATER_RATE'].notna() &
                         (well_df['WATER_RATE'] > 0))]

    if show_water_cut:
        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=('Production Rates',
                                            'Water Cut & GOR'),
                            row_heights=[0.6, 0.4])
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(
        x=well_df['DATEPRD'],
        y=well_df['OIL_RATE'],
        name='Oil Rate (BPD)',
        line=dict(color='green', width=2),
        hovertemplate='%{x}<br>Oil: %{y:.1f} BPD<extra></extra>'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(
        x=well_df['DATEPRD'],
        y=well_df['WATER_RATE'],
        name='Water Rate (BPD)',
        line=dict(color='blue', width=2),
        hovertemplate='%{x}<br>Water: %{y:.1f} BPD<extra></extra>'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(
        x=well_df['DATEPRD'],
        y=well_df['GAS_RATE'],
        name='Gas Rate (Sm3/d)',
        line=dict(color='red', width=1),
        hovertemplate='%{x}<br>Gas: %{y:.1f} Sm3/d<extra></extra>'),
                  row=1,
                  col=1)

    if show_water_cut:
        fig.add_trace(go.Scatter(
            x=well_df['DATEPRD'],
            y=well_df['WATER_CUT'] * 100,
            name='Water Cut (%)',
            line=dict(color='navy', width=2),
            hovertemplate='%{x}<br>Water Cut: %{y:.1f}%<extra></extra>'),
                      row=2,
                      col=1)

        fig.add_trace(go.Scatter(
            x=well_df['DATEPRD'],
            y=well_df['GOR'],
            name='GOR (Sm3/Sm3)',
            line=dict(color='orange', width=2),
            yaxis='y4',
            hovertemplate='%{x}<br>GOR: %{y:.1f}<extra></extra>'),
                      row=2,
                      col=1)

    fig.update_layout(title=f'Production History - {well_name}',
                      height=600 if show_water_cut else 400,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      hovermode='x unified')

    fig.update_xaxes(title_text="Date", row=2 if show_water_cut else 1, col=1)
    fig.update_yaxes(title_text="Rate (BPD)", row=1, col=1)

    if show_water_cut:
        fig.update_yaxes(title_text="Water Cut (%) / GOR", row=2, col=1)

    return fig


def plot_well_time_series_with_rolling(daily_df: pd.DataFrame,
                                       well_name: str,
                                       rolling_window: int = 30) -> go.Figure:
    """
    Create a time series plot with rolling average overlay for trend analysis.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Name of the well to plot.
        rolling_window: Window size for rolling average in days.
        
    Returns:
        Plotly Figure object.
    """
    well_df = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name].copy()
    well_df = well_df.sort_values('DATEPRD')

    well_df['OIL_RATE_ROLLING'] = well_df['OIL_RATE'].rolling(
        window=rolling_window, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=well_df['DATEPRD'],
                   y=well_df['OIL_RATE'],
                   name='Daily Oil Rate',
                   mode='lines',
                   line=dict(color='lightgreen', width=1),
                   opacity=0.6,
                   hovertemplate='%{x}<br>Daily: %{y:.1f} BPD<extra></extra>'))

    fig.add_trace(
        go.Scatter(
            x=well_df['DATEPRD'],
            y=well_df['OIL_RATE_ROLLING'],
            name=f'{rolling_window}-Day Rolling Avg',
            mode='lines',
            line=dict(color='darkgreen', width=3),
            hovertemplate='%{x}<br>Rolling Avg: %{y:.1f} BPD<extra></extra>'))

    fig.update_layout(title=f'Oil Production Trend - {well_name}',
                      xaxis_title='Date',
                      yaxis_title='Oil Rate (BPD)',
                      height=400,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      hovermode='x unified')

    return fig


def plot_field_overview(daily_df: pd.DataFrame) -> go.Figure:
    """
    Create a field-level time series showing total production.
    
    Args:
        daily_df: Daily production DataFrame.
        
    Returns:
        Plotly Figure object.
    """
    field_df = daily_df.groupby('DATEPRD').agg({
        'OIL_RATE': 'sum',
        'WATER_RATE': 'sum',
        'GAS_RATE': 'sum'
    }).reset_index()

    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Field Oil & Water Rates',
                                        'Field Gas Rate'),
                        row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(
        x=field_df['DATEPRD'],
        y=field_df['OIL_RATE'],
        name='Total Oil Rate',
        fill='tozeroy',
        line=dict(color='green', width=2),
        hovertemplate='%{x}<br>Oil: %{y:,.0f} BPD<extra></extra>'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(
        x=field_df['DATEPRD'],
        y=field_df['WATER_RATE'],
        name='Total Water Rate',
        line=dict(color='blue', width=2),
        hovertemplate='%{x}<br>Water: %{y:,.0f} BPD<extra></extra>'),
                  row=1,
                  col=1)

    fig.add_trace(go.Scatter(
        x=field_df['DATEPRD'],
        y=field_df['GAS_RATE'],
        name='Total Gas Rate',
        fill='tozeroy',
        line=dict(color='red', width=2),
        hovertemplate='%{x}<br>Gas: %{y:,.0f} Sm3/d<extra></extra>'),
                  row=2,
                  col=1)

    fig.update_layout(title='Volve Field - Total Production History',
                      height=500,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      hovermode='x unified')

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Oil/Water Rate (BPD)", row=1, col=1)
    fig.update_yaxes(title_text="Gas Rate (Sm3/d)", row=2, col=1)

    return fig


def plot_well_bar_kpis(well_summary_df: pd.DataFrame,
                       metric: str = 'LATEST_OIL_RATE',
                       title: str = 'Well KPI Comparison') -> go.Figure:
    """
    Create a bar chart comparing a specific KPI across wells.
    
    Args:
        well_summary_df: Well summary DataFrame.
        metric: Column name to plot.
        title: Chart title.
        
    Returns:
        Plotly Figure object.
    """
    df = well_summary_df.sort_values(metric, ascending=True)

    color_map = {
        'LATEST_OIL_RATE': 'green',
        'LATEST_WATER_CUT': 'blue',
        'OIL_LOSS': 'red',
        'SCORE': 'orange'
    }

    color = color_map.get(metric, 'steelblue')

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df[metric],
               y=df['NPD_WELL_BORE_NAME'],
               orientation='h',
               marker_color=color,
               hovertemplate='%{y}<br>Value: %{x:.2f}<extra></extra>'))

    fig.update_layout(title=title,
                      xaxis_title=metric.replace('_', ' ').title(),
                      yaxis_title='Well',
                      height=max(300,
                                 len(df) * 40),
                      showlegend=False)

    return fig


def plot_oil_loss_comparison(well_summary_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing oil loss per well.
    
    Args:
        well_summary_df: Well summary DataFrame.
        
    Returns:
        Plotly Figure object.
    """
    df = well_summary_df[well_summary_df['OIL_LOSS'] > 0].sort_values(
        'OIL_LOSS', ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df['OIL_LOSS'],
               y=df['NPD_WELL_BORE_NAME'],
               orientation='h',
               marker_color='red',
               hovertemplate='%{y}<br>Oil Loss: %{x:.1f} BPD<extra></extra>'))

    fig.update_layout(
        title='Oil Production Loss by Well (vs 12-Month Average)',
        xaxis_title='Oil Loss (BPD)',
        yaxis_title='Well',
        height=max(300,
                   len(df) * 40),
        showlegend=False)

    return fig


def plot_water_cut_comparison(well_summary_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing water cut per well.
    
    Args:
        well_summary_df: Well summary DataFrame.
        
    Returns:
        Plotly Figure object.
    """
    df = well_summary_df.sort_values('LATEST_WATER_CUT', ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df['LATEST_WATER_CUT'] * 100,
               y=df['NPD_WELL_BORE_NAME'],
               orientation='h',
               marker_color='blue',
               hovertemplate='%{y}<br>Water Cut: %{x:.1f}%<extra></extra>'))

    fig.update_layout(title='Water Cut by Well',
                      xaxis_title='Water Cut (%)',
                      yaxis_title='Well',
                      height=max(300,
                                 len(df) * 40),
                      showlegend=False)

    return fig


def plot_intervention_scores(intervention_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing intervention priority scores.
    
    Args:
        intervention_df: Intervention candidates DataFrame.
        
    Returns:
        Plotly Figure object.
    """
    df = intervention_df.head(10).sort_values('SCORE', ascending=True)

    fig = go.Figure()

    colors = [
        '#ff7f0e' if s > 0.7 else '#2ca02c' if s < 0.3 else '#1f77b4'
        for s in df['SCORE']
    ]

    fig.add_trace(
        go.Bar(x=df['SCORE'],
               y=df['NPD_WELL_BORE_NAME'],
               orientation='h',
               marker_color=colors,
               hovertemplate='%{y}<br>Score: %{x:.2f}<extra></extra>'))

    fig.update_layout(title='Top 10 Intervention Candidates by Priority Score',
                      xaxis_title='Intervention Priority Score (0-1)',
                      yaxis_title='Well',
                      height=400,
                      showlegend=False)

    return fig


def plot_well_performance_analysis(daily_df: pd.DataFrame,
                                   well_name: str,
                                   months: int = 12) -> go.Figure:
    """
    Create a detailed performance analysis plot for intervention candidates.
    
    Shows oil rate, on-stream hours, and identifies periods of concern.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Name of the well to analyze.
        months: Number of months to display.
        
    Returns:
        Plotly Figure object.
    """
    from datetime import timedelta

    well_df = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name].copy()
    well_df = well_df.sort_values('DATEPRD')

    max_date = well_df['DATEPRD'].max()
    start_date = max_date - timedelta(days=months * 30)
    well_df = well_df[well_df['DATEPRD'] >= start_date]

    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Oil Production Rate',
                                        'On-Stream Hours'),
                        row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(
        x=well_df['DATEPRD'],
        y=well_df['OIL_RATE'],
        name='Oil Rate',
        line=dict(color='green', width=2),
        hovertemplate='%{x}<br>Oil Rate: %{y:.1f} BPD<extra></extra>'),
                  row=1,
                  col=1)

    avg_oil = well_df['OIL_RATE'].mean()
    fig.add_hline(y=avg_oil,
                  line_dash="dash",
                  line_color="red",
                  annotation_text=f"Avg: {avg_oil:.0f} BPD",
                  row=1,
                  col=1)

    on_stream_colors = [
        'red' if h < 12 else 'orange' if h < 20 else 'green'
        for h in well_df['ON_STREAM_HRS']
    ]

    fig.add_trace(go.Bar(
        x=well_df['DATEPRD'],
        y=well_df['ON_STREAM_HRS'],
        name='On-Stream Hours',
        marker_color=on_stream_colors,
        hovertemplate='%{x}<br>Hours: %{y:.1f}<extra></extra>'),
                  row=2,
                  col=1)

    fig.update_layout(
        title=f'Performance Analysis - {well_name} (Last {months} Months)',
        height=500,
        showlegend=True,
        legend=dict(orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1),
        hovermode='x unified')

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Oil Rate (BPD)", row=1, col=1)
    fig.update_yaxes(title_text="On-Stream Hours", row=2, col=1)

    return fig


def plot_cumulative_production(daily_df: pd.DataFrame,
                               well_name: str) -> go.Figure:
    """
    Create a cumulative production plot for a well.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Name of the well to plot.
        
    Returns:
        Plotly Figure object.
    """
    SM3_TO_BBL = 6.2898

    well_df = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name].copy()
    well_df = well_df.sort_values('DATEPRD')

    if 'BORE_OIL_VOL_BBL' in well_df.columns:
        well_df['CUM_OIL'] = well_df['BORE_OIL_VOL_BBL'].cumsum()
        well_df['CUM_WATER'] = well_df['BORE_WAT_VOL_BBL'].cumsum()
    else:
        well_df['CUM_OIL'] = (well_df['BORE_OIL_VOL'] * SM3_TO_BBL).cumsum()
        well_df['CUM_WATER'] = (well_df['BORE_WAT_VOL'] * SM3_TO_BBL).cumsum()
    well_df['CUM_GAS'] = well_df['BORE_GAS_VOL'].cumsum()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=well_df['DATEPRD'],
            y=well_df['CUM_OIL'],
            name='Cumulative Oil',
            line=dict(color='green', width=2),
            hovertemplate='%{x}<br>Cum Oil: %{y:,.0f} bbl<extra></extra>'))

    fig.add_trace(
        go.Scatter(
            x=well_df['DATEPRD'],
            y=well_df['CUM_WATER'],
            name='Cumulative Water',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Cum Water: %{y:,.0f} bbl<extra></extra>'))

    fig.update_layout(title=f'Cumulative Production - {well_name}',
                      xaxis_title='Date',
                      yaxis_title='Cumulative Volume (bbl)',
                      height=400,
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1),
                      hovermode='x unified')

    return fig


def plot_dca_forecast(dca_results: dict) -> go.Figure:
    """
    Create a decline curve analysis plot with historical data and forecasts.
    
    Args:
        dca_results: Dictionary from run_dca_analysis().
        
    Returns:
        Plotly Figure object.
    """
    if not dca_results.get('success', False):
        fig = go.Figure()
        fig.add_annotation(text=dca_results.get('error',
                                                'DCA analysis failed'),
                           xref="paper",
                           yref="paper",
                           x=0.5,
                           y=0.5,
                           showarrow=False)
        return fig

    fig = go.Figure()

    hist_time = dca_results['historical_data']['time']
    hist_rate = dca_results['historical_data']['rate']

    fig.add_trace(
        go.Scatter(
            x=hist_time,
            y=hist_rate,
            mode='markers',
            name='Historical Data',
            marker=dict(color='Brown', size=8),
            hovertemplate='Month %{x:.1f}<br>Rate: %{y:.1f} BPD<extra></extra>'
        ))

    exp_data = dca_results['exponential']
    if not np.isnan(exp_data['qi']):
        fig.add_trace(
            go.Scatter(x=exp_data['forecast_time'],
                       y=exp_data['forecast_rate'],
                       mode='lines',
                       name=f'Exponential (R²={exp_data["r_squared"]:.3f})',
                       line=dict(color='blue', width=2, dash='dash'),
                       hovertemplate=
                       'Month %{x:.1f}<br>Rate: %{y:.1f} BPD<extra></extra>'))

    hyp_data = dca_results['hyperbolic']
    if not np.isnan(hyp_data['qi']):
        fig.add_trace(
            go.Scatter(
                x=hyp_data['forecast_time'],
                y=hyp_data['forecast_rate'],
                mode='lines',
                name=
                f'Hyperbolic b={hyp_data["b"]:.2f} (R²={hyp_data["r_squared"]:.3f})',
                line=dict(color='red', width=2),
                hovertemplate=
                'Month %{x:.1f}<br>Rate: %{y:.1f} BPD<extra></extra>'))

    econ_limit = dca_results['economic_limit']
    fig.add_hline(y=econ_limit,
                  line_dash="dot",
                  line_color="gray",
                  annotation_text=f"Economic Limit: {econ_limit} BPD")

    fig.update_layout(
        title=f'Decline Curve Analysis - {dca_results["well_name"]}',
        xaxis_title='Time (Months from Start)',
        yaxis_title='Oil Rate (BPD)',
        height=500,
        showlegend=True,
        legend=dict(orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1),
        hovermode='x unified')

    return fig


def plot_eur_comparison(dca_summary_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart comparing EUR across wells.
    
    Args:
        dca_summary_df: DCA summary DataFrame.
        
    Returns:
        Plotly Figure object.
    """
    df = dca_summary_df[dca_summary_df['EUR'].notna()].copy()
    df = df.sort_values('EUR', ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df['CURRENT_CUM'] / 1000,
               y=df['NPD_WELL_BORE_NAME'],
               orientation='h',
               name='Current Cumulative',
               marker_color='green',
               hovertemplate='%{y}<br>Current: %{x:,.0f} Mbbl<extra></extra>'))

    fig.add_trace(
        go.Bar(
            x=df['REMAINING_RESERVES'] / 1000,
            y=df['NPD_WELL_BORE_NAME'],
            orientation='h',
            name='Remaining Reserves',
            marker_color='orange',
            hovertemplate='%{y}<br>Remaining: %{x:,.0f} Mbbl<extra></extra>'))

    fig.update_layout(title='Estimated Ultimate Recovery (EUR) by Well',
                      xaxis_title='Volume (Thousand bbl)',
                      yaxis_title='Well',
                      barmode='stack',
                      height=max(300,
                                 len(df) * 40),
                      showlegend=True,
                      legend=dict(orientation="h",
                                  yanchor="bottom",
                                  y=1.02,
                                  xanchor="right",
                                  x=1))

    return fig


def plot_decline_rate_comparison(dca_summary_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart comparing decline rates across wells.
    
    Args:
        dca_summary_df: DCA summary DataFrame.
        
    Returns:
        Plotly Figure object.
    """
    df = dca_summary_df[dca_summary_df['DECLINE_RATE_PCT'].notna()].copy()
    df = df.sort_values('DECLINE_RATE_PCT', ascending=True)

    colors = [
        'red' if d > 10 else 'orange' if d > 5 else 'green'
        for d in df['DECLINE_RATE_PCT']
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df['DECLINE_RATE_PCT'],
            y=df['NPD_WELL_BORE_NAME'],
            orientation='h',
            marker_color=colors,
            hovertemplate='%{y}<br>Decline: %{x:.1f}%/month<extra></extra>'))

    fig.update_layout(title='Monthly Decline Rate by Well',
                      xaxis_title='Decline Rate (%/month)',
                      yaxis_title='Well',
                      height=max(300,
                                 len(df) * 40),
                      showlegend=False)

    return fig


def plot_alerts_summary(alerts_df: pd.DataFrame) -> go.Figure:
    """
    Create a summary visualization of production alerts.
    
    Args:
        alerts_df: DataFrame with alert data.
        
    Returns:
        Plotly Figure object.
    """
    if len(alerts_df) == 0:
        fig = go.Figure()
        fig.add_annotation(x=0.5,
                           y=0.5,
                           text="No active alerts",
                           showarrow=False,
                           font=dict(size=20))
        fig.update_layout(height=200)
        return fig

    type_counts = alerts_df['type'].value_counts()
    severity_counts = alerts_df['severity'].value_counts()

    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=('Alerts by Type',
                                        'Alerts by Severity'),
                        specs=[[{
                            "type": "pie"
                        }, {
                            "type": "pie"
                        }]])

    type_colors = {
        'RATE_DROP': '#ff6b6b',
        'WATER_CUT_SPIKE': '#4ecdc4',
        'HIGH_WATER_CUT': '#45b7d1',
        'GOR_ANOMALY': '#f9c74f',
        'EXTENDED_DOWNTIME': '#90be6d',
        'UNDERPERFORMANCE': '#577590'
    }

    fig.add_trace(go.Pie(labels=type_counts.index,
                         values=type_counts.values,
                         marker_colors=[
                             type_colors.get(t, '#888888')
                             for t in type_counts.index
                         ],
                         hole=0.4,
                         textinfo='value+label',
                         textposition='outside'),
                  row=1,
                  col=1)

    severity_colors = {
        'HIGH': '#ff0000',
        'MEDIUM': '#ffa500',
        'LOW': '#00ff00'
    }

    fig.add_trace(go.Pie(labels=severity_counts.index,
                         values=severity_counts.values,
                         marker_colors=[
                             severity_colors.get(s, '#888888')
                             for s in severity_counts.index
                         ],
                         hole=0.4,
                         textinfo='value+label',
                         textposition='outside'),
                  row=1,
                  col=2)

    fig.update_layout(height=350,
                      showlegend=False,
                      title_text='Alert Distribution')

    return fig


def plot_well_alerts_timeline(daily_df: pd.DataFrame, well_name: str,
                              alerts_df: pd.DataFrame) -> go.Figure:
    """
    Create a timeline visualization showing alerts overlaid on production.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Name of the well.
        alerts_df: DataFrame with alerts.
        
    Returns:
        Plotly Figure object.
    """
    well_df = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name].copy()
    well_df = well_df.sort_values('DATEPRD')

    well_alerts = alerts_df[alerts_df['well'] == well_name]

    fig = make_subplots(rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Oil Rate', 'Water Cut'),
                        row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(x=well_df['DATEPRD'],
                             y=well_df['OIL_RATE'],
                             name='Oil Rate',
                             line=dict(color='green', width=2),
                             fill='tozeroy',
                             fillcolor='rgba(0,128,0,0.2)'),
                  row=1,
                  col=1)

    if 'WATER_CUT' in well_df.columns:
        fig.add_trace(go.Scatter(x=well_df['DATEPRD'],
                                 y=well_df['WATER_CUT'] * 100,
                                 name='Water Cut %',
                                 line=dict(color='blue', width=2),
                                 fill='tozeroy',
                                 fillcolor='rgba(0,0,255,0.2)'),
                      row=2,
                      col=1)

    for _, alert in well_alerts.iterrows():
        if alert['date'] is not None:
            color = 'red' if alert['severity'] == 'HIGH' else 'orange'

            fig.add_vline(x=alert['date'],
                          line=dict(color=color, dash='dash', width=2),
                          row=1,
                          col=1)

            fig.add_annotation(x=alert['date'],
                               y=well_df['OIL_RATE'].max() * 0.9,
                               text=alert['type'],
                               showarrow=True,
                               arrowhead=2,
                               arrowcolor=color,
                               font=dict(size=10, color=color),
                               row=1,
                               col=1)

    fig.update_layout(title=f'Production and Alerts - {well_name}',
                      height=500,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))

    fig.update_yaxes(title_text='Oil Rate (BPD)', row=1, col=1)
    fig.update_yaxes(title_text='Water Cut (%)', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    return fig


def plot_pressure_transient(results: dict) -> go.Figure:
    """
    Create a log-log diagnostic plot for pressure transient analysis.
    
    Args:
        results: Dictionary with well test results.
        
    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=('Log-Log Diagnostic Plot',
                                        'Semi-Log Plot'),
                        horizontal_spacing=0.12)

    time = results.get('time', np.array([]))
    pressure = results.get('pressure', np.array([]))
    deriv_time = results.get('derivative_time', np.array([]))
    derivative = results.get('derivative', np.array([]))

    if len(time) > 0 and len(pressure) > 0:
        p_change = np.abs(pressure - pressure[0])

        fig.add_trace(go.Scatter(
            x=time,
            y=p_change,
            name='Pressure Change',
            mode='markers+lines',
            marker=dict(size=4, color='blue'),
            line=dict(width=1),
            hovertemplate='t=%{x:.2f}hr<br>ΔP=%{y:.1f}bar<extra></extra>'),
                      row=1,
                      col=1)

        if len(deriv_time) > 0 and len(derivative) > 0:
            valid = derivative > 0
            fig.add_trace(go.Scatter(
                x=deriv_time[valid],
                y=derivative[valid],
                name='Pressure Derivative',
                mode='markers+lines',
                marker=dict(size=4, color='red'),
                line=dict(width=1),
                hovertemplate=
                't=%{x:.2f}hr<br>dP/d(lnt)=%{y:.1f}<extra></extra>'),
                          row=1,
                          col=1)

        fig.add_trace(go.Scatter(
            x=np.log10(time + 1e-10),
            y=pressure,
            name='Pressure vs log(t)',
            mode='markers+lines',
            marker=dict(size=4, color='green'),
            line=dict(width=1),
            hovertemplate='log(t)=%{x:.2f}<br>P=%{y:.1f}bar<extra></extra>'),
                      row=1,
                      col=2)

    fig.update_xaxes(type="log", title_text="Time (hours)", row=1, col=1)
    fig.update_yaxes(type="log",
                     title_text="Pressure Change (bar)",
                     row=1,
                     col=1)

    fig.update_xaxes(title_text="log(Time)", row=1, col=2)
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=2)

    test_type = results.get('test_type', 'Well Test')
    well_name = results.get('well_name', 'Unknown')

    fig.update_layout(
        title=f'Pressure Transient Analysis - {well_name} ({test_type})',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))

    return fig


def plot_flow_regime_interpretation(results: dict) -> go.Figure:
    """
    Create a visual interpretation of flow regimes.
    
    Args:
        results: Dictionary with well test results.
        
    Returns:
        Plotly Figure object.
    """
    time = results.get('time', np.array([]))
    deriv_time = results.get('derivative_time', np.array([]))
    derivative = results.get('derivative', np.array([]))
    flow_regimes = results.get('flow_regimes', {})

    fig = go.Figure()

    if len(deriv_time) > 0 and len(derivative) > 0:
        valid = derivative > 0
        fig.add_trace(
            go.Scatter(x=deriv_time[valid],
                       y=derivative[valid],
                       name='Pressure Derivative',
                       mode='markers+lines',
                       marker=dict(size=5, color='blue'),
                       line=dict(width=2)))

        if flow_regimes.get('wellbore_storage', False):
            ws_end = flow_regimes.get('wellbore_storage_end')
            if ws_end and ws_end > time[0]:
                fig.add_vrect(x0=time[0],
                              x1=ws_end,
                              fillcolor="yellow",
                              opacity=0.3,
                              layer="below",
                              line_width=0,
                              annotation_text="Wellbore Storage",
                              annotation_position="top left")

        if flow_regimes.get('radial_flow', False):
            rf_start = flow_regimes.get('radial_flow_start')
            rf_end = flow_regimes.get('radial_flow_end')
            if rf_start and rf_end:
                fig.add_vrect(x0=rf_start,
                              x1=rf_end,
                              fillcolor="green",
                              opacity=0.3,
                              layer="below",
                              line_width=0,
                              annotation_text="Radial Flow",
                              annotation_position="top left")

        if flow_regimes.get('boundary_effect', False):
            bd_start = flow_regimes.get('boundary_start')
            if bd_start:
                fig.add_vrect(x0=bd_start,
                              x1=time[-1],
                              fillcolor="red",
                              opacity=0.2,
                              layer="below",
                              line_width=0,
                              annotation_text="Boundary Effect",
                              annotation_position="top right")

    fig.update_xaxes(type="log", title_text="Time (hours)")
    fig.update_yaxes(type="log", title_text="Pressure Derivative")

    fig.update_layout(title='Flow Regime Identification',
                      height=400,
                      showlegend=True)

    return fig


def plot_well_test_comparison(summary_df: pd.DataFrame) -> go.Figure:
    """
    Create a comparison chart of well test parameters across wells.
    
    Args:
        summary_df: DataFrame with well test summary.
        
    Returns:
        Plotly Figure object.
    """
    buildup_df = summary_df[summary_df['TEST_TYPE'] ==
                            'Pressure Build-up'].copy()

    if len(buildup_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No well test data available",
                           x=0.5,
                           y=0.5,
                           showarrow=False)
        return fig

    fig = make_subplots(rows=1,
                        cols=2,
                        subplot_titles=('Permeability (mD)', 'Skin Factor'),
                        horizontal_spacing=0.15)

    buildup_df = buildup_df.sort_values('PERMEABILITY_MD', ascending=True)

    colors_k = [
        'green' if k > 100 else 'orange' if k > 10 else 'red'
        for k in buildup_df['PERMEABILITY_MD']
    ]

    fig.add_trace(go.Bar(y=buildup_df['NPD_WELL_BORE_NAME'],
                         x=buildup_df['PERMEABILITY_MD'],
                         orientation='h',
                         marker_color=colors_k,
                         name='Permeability',
                         hovertemplate='%{y}<br>k=%{x:.1f} mD<extra></extra>'),
                  row=1,
                  col=1)

    buildup_df = buildup_df.sort_values('SKIN_FACTOR', ascending=False)

    colors_s = [
        'green' if s < 0 else 'orange' if s < 10 else 'red'
        for s in buildup_df['SKIN_FACTOR']
    ]

    fig.add_trace(go.Bar(y=buildup_df['NPD_WELL_BORE_NAME'],
                         x=buildup_df['SKIN_FACTOR'],
                         orientation='h',
                         marker_color=colors_s,
                         name='Skin',
                         hovertemplate='%{y}<br>Skin=%{x:.1f}<extra></extra>'),
                  row=1,
                  col=2)

    fig.update_layout(title='Well Test Parameter Comparison',
                      height=max(300,
                                 len(buildup_df) * 35),
                      showlegend=False)

    fig.update_xaxes(title_text='Permeability (mD)', row=1, col=1)
    fig.update_xaxes(title_text='Skin Factor', row=1, col=2)

    return fig
