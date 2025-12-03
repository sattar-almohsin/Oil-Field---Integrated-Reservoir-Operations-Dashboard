"""
Production Alerts Module

Implements automated alert detection for critical production changes
including rate drops, water cut spikes, and underperformance detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


def detect_rate_drops(
    df: pd.DataFrame,
    threshold_pct: float = 30.0,
    lookback_days: int = 7
) -> List[Dict[str, Any]]:
    """
    Detect significant oil rate drops compared to recent average.
    
    Args:
        df: Daily production dataframe
        threshold_pct: Percentage drop that triggers alert
        lookback_days: Number of days to average for comparison
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    if 'OIL_RATE' not in df.columns or 'DATEPRD' not in df.columns:
        return alerts
    
    df = df.sort_values('DATEPRD')
    
    for well in df['NPD_WELL_BORE_NAME'].unique():
        well_df = df[df['NPD_WELL_BORE_NAME'] == well].copy()
        
        if len(well_df) < lookback_days + 1:
            continue
        
        well_df = well_df[well_df['OIL_RATE'] > 0]
        
        if len(well_df) < lookback_days + 1:
            continue
        
        recent_avg = well_df['OIL_RATE'].iloc[-lookback_days-1:-1].mean()
        latest_rate = well_df['OIL_RATE'].iloc[-1]
        
        if recent_avg > 0:
            drop_pct = ((recent_avg - latest_rate) / recent_avg) * 100
            
            if drop_pct >= threshold_pct:
                alerts.append({
                    'type': 'RATE_DROP',
                    'severity': 'HIGH' if drop_pct >= 50 else 'MEDIUM',
                    'well': well,
                    'message': f"Oil rate dropped {drop_pct:.1f}% from {recent_avg:.1f} to {latest_rate:.1f} Sm3/d",
                    'current_value': latest_rate,
                    'previous_value': recent_avg,
                    'change_pct': -drop_pct,
                    'date': well_df['DATEPRD'].iloc[-1],
                    'action': 'Investigate cause - check for mechanical issues, scale, or reservoir pressure decline'
                })
    
    return alerts


def detect_water_cut_spikes(
    df: pd.DataFrame,
    threshold_pct: float = 10.0,
    absolute_threshold: float = 0.80
) -> List[Dict[str, Any]]:
    """
    Detect significant water cut increases.
    
    Args:
        df: Daily production dataframe
        threshold_pct: Percentage increase that triggers alert
        absolute_threshold: Absolute water cut level that always triggers alert
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    if 'WATER_CUT' not in df.columns or 'DATEPRD' not in df.columns:
        return alerts
    
    df = df.sort_values('DATEPRD')
    
    for well in df['NPD_WELL_BORE_NAME'].unique():
        well_df = df[df['NPD_WELL_BORE_NAME'] == well].copy()
        well_df = well_df[well_df['WATER_CUT'].notna()]
        
        if len(well_df) < 8:
            continue
        
        recent_avg = well_df['WATER_CUT'].iloc[-8:-1].mean()
        latest_wc = well_df['WATER_CUT'].iloc[-1]
        
        if latest_wc >= absolute_threshold:
            alerts.append({
                'type': 'HIGH_WATER_CUT',
                'severity': 'HIGH',
                'well': well,
                'message': f"Water cut critically high at {latest_wc*100:.1f}%",
                'current_value': latest_wc,
                'previous_value': recent_avg,
                'change_pct': ((latest_wc - recent_avg) / recent_avg * 100) if recent_avg > 0 else 0,
                'date': well_df['DATEPRD'].iloc[-1],
                'action': 'Consider water shutoff intervention or well workover'
            })
        elif recent_avg > 0 and recent_avg < 1:
            increase_pct = ((latest_wc - recent_avg) / recent_avg) * 100
            
            if increase_pct >= threshold_pct:
                alerts.append({
                    'type': 'WATER_CUT_SPIKE',
                    'severity': 'MEDIUM',
                    'well': well,
                    'message': f"Water cut increased {increase_pct:.1f}% from {recent_avg*100:.1f}% to {latest_wc*100:.1f}%",
                    'current_value': latest_wc,
                    'previous_value': recent_avg,
                    'change_pct': increase_pct,
                    'date': well_df['DATEPRD'].iloc[-1],
                    'action': 'Monitor for water coning or breakthrough from injector'
                })
    
    return alerts


def detect_gor_anomalies(
    df: pd.DataFrame,
    threshold_pct: float = 50.0
) -> List[Dict[str, Any]]:
    """
    Detect significant GOR changes indicating gas breakthrough or depletion.
    
    Args:
        df: Daily production dataframe
        threshold_pct: Percentage change that triggers alert
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    if 'GOR' not in df.columns or 'DATEPRD' not in df.columns:
        return alerts
    
    df = df.sort_values('DATEPRD')
    
    for well in df['NPD_WELL_BORE_NAME'].unique():
        well_df = df[df['NPD_WELL_BORE_NAME'] == well].copy()
        well_df = well_df[well_df['GOR'].notna() & (well_df['GOR'] > 0)]
        
        if len(well_df) < 15:
            continue
        
        monthly_avg = well_df['GOR'].iloc[-31:-1].mean() if len(well_df) >= 31 else well_df['GOR'].iloc[:-1].mean()
        weekly_avg = well_df['GOR'].iloc[-7:].mean()
        
        if monthly_avg > 0:
            change_pct = ((weekly_avg - monthly_avg) / monthly_avg) * 100
            
            if abs(change_pct) >= threshold_pct:
                severity = 'HIGH' if abs(change_pct) >= 100 else 'MEDIUM'
                direction = 'increased' if change_pct > 0 else 'decreased'
                
                alerts.append({
                    'type': 'GOR_ANOMALY',
                    'severity': severity,
                    'well': well,
                    'message': f"GOR {direction} by {abs(change_pct):.1f}% from {monthly_avg:.0f} to {weekly_avg:.0f} Sm3/Sm3",
                    'current_value': weekly_avg,
                    'previous_value': monthly_avg,
                    'change_pct': change_pct,
                    'date': well_df['DATEPRD'].iloc[-1],
                    'action': 'Review reservoir pressure and gas cap status' if change_pct > 0 else 'Check for liquid loading issues'
                })
    
    return alerts


def detect_downtime_issues(
    df: pd.DataFrame,
    min_hours_threshold: float = 12.0,
    consecutive_days: int = 3
) -> List[Dict[str, Any]]:
    """
    Detect wells with extended low on-stream hours.
    
    Args:
        df: Daily production dataframe
        min_hours_threshold: Minimum hours considered as partial production
        consecutive_days: Days of low production to trigger alert
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    if 'ON_STREAM_HRS' not in df.columns or 'DATEPRD' not in df.columns:
        return alerts
    
    df = df.sort_values('DATEPRD')
    
    for well in df['NPD_WELL_BORE_NAME'].unique():
        well_df = df[df['NPD_WELL_BORE_NAME'] == well].copy()
        
        if len(well_df) < consecutive_days:
            continue
        
        recent_hrs = well_df['ON_STREAM_HRS'].iloc[-consecutive_days:]
        
        if (recent_hrs < min_hours_threshold).all():
            avg_hrs = recent_hrs.mean()
            
            alerts.append({
                'type': 'EXTENDED_DOWNTIME',
                'severity': 'HIGH' if avg_hrs < 6 else 'MEDIUM',
                'well': well,
                'message': f"Well has had low on-stream hours ({avg_hrs:.1f} hrs avg) for {consecutive_days} consecutive days",
                'current_value': avg_hrs,
                'previous_value': 24.0,
                'change_pct': ((avg_hrs - 24) / 24) * 100,
                'date': well_df['DATEPRD'].iloc[-1],
                'action': 'Investigate operational issues - check wellhead, choke, or surface facilities'
            })
    
    return alerts


def detect_underperformance(
    df: pd.DataFrame,
    well_summary: pd.DataFrame,
    threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Detect underperforming wells based on historical average.
    
    Args:
        df: Daily production dataframe
        well_summary: Well summary with historical averages
        threshold: Fraction of historical rate to flag as underperforming
    
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    if 'UNDERPERFORMING' not in well_summary.columns:
        return alerts
    
    underperforming = well_summary[well_summary['UNDERPERFORMING'] == True]
    
    for _, row in underperforming.iterrows():
        well = row['NPD_WELL_BORE_NAME']
        current_rate = row.get('LATEST_OIL_RATE', 0)
        avg_12m = row.get('AVG_12M_OIL_RATE', 0)
        
        if avg_12m > 0:
            pct_of_historical = (current_rate / avg_12m) * 100
            
            alerts.append({
                'type': 'UNDERPERFORMANCE',
                'severity': 'MEDIUM' if pct_of_historical >= 50 else 'HIGH',
                'well': well,
                'message': f"Producing at {pct_of_historical:.0f}% of 12-month average ({current_rate:.1f} vs {avg_12m:.1f} Sm3/d)",
                'current_value': current_rate,
                'previous_value': avg_12m,
                'change_pct': pct_of_historical - 100,
                'date': df[df['NPD_WELL_BORE_NAME'] == well]['DATEPRD'].max() if well in df['NPD_WELL_BORE_NAME'].values else None,
                'action': 'Review well performance - may be candidate for stimulation or artificial lift review'
            })
    
    return alerts


def get_all_alerts(
    df: pd.DataFrame,
    well_summary: Optional[pd.DataFrame] = None,
    rate_drop_threshold: float = 30.0,
    water_cut_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Generate all production alerts for the field.
    
    Args:
        df: Daily production dataframe
        well_summary: Well summary dataframe (optional)
        rate_drop_threshold: Percentage drop to flag as rate drop
        water_cut_threshold: Percentage increase to flag as water cut spike
    
    Returns:
        DataFrame with all alerts
    """
    all_alerts = []
    
    all_alerts.extend(detect_rate_drops(df, rate_drop_threshold))
    all_alerts.extend(detect_water_cut_spikes(df, water_cut_threshold))
    all_alerts.extend(detect_gor_anomalies(df))
    all_alerts.extend(detect_downtime_issues(df))
    
    if well_summary is not None:
        all_alerts.extend(detect_underperformance(df, well_summary))
    
    if not all_alerts:
        return pd.DataFrame(columns=[
            'type', 'severity', 'well', 'message', 
            'current_value', 'previous_value', 'change_pct', 'date', 'action'
        ])
    
    alerts_df = pd.DataFrame(all_alerts)
    
    severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    alerts_df['severity_order'] = alerts_df['severity'].map(severity_order)
    alerts_df = alerts_df.sort_values(['severity_order', 'well']).reset_index(drop=True)
    alerts_df = alerts_df.drop('severity_order', axis=1)
    
    return alerts_df


def get_alert_summary(alerts_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for alerts.
    
    Args:
        alerts_df: DataFrame of alerts
    
    Returns:
        Dictionary with alert summary
    """
    if len(alerts_df) == 0:
        return {
            'total_alerts': 0,
            'high_severity': 0,
            'medium_severity': 0,
            'low_severity': 0,
            'wells_affected': 0,
            'alerts_by_type': {}
        }
    
    return {
        'total_alerts': len(alerts_df),
        'high_severity': (alerts_df['severity'] == 'HIGH').sum(),
        'medium_severity': (alerts_df['severity'] == 'MEDIUM').sum(),
        'low_severity': (alerts_df['severity'] == 'LOW').sum(),
        'wells_affected': alerts_df['well'].nunique(),
        'alerts_by_type': alerts_df['type'].value_counts().to_dict()
    }
