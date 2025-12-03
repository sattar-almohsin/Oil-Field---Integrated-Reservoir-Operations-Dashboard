"""
KPI calculation module for reservoir and operations engineering metrics.

This module provides functions to compute well-level and field-level KPIs,
including production indices, underperformance detection, and intervention
candidate ranking.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import timedelta


def compute_well_daily_kpis(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare daily dataframe with derived KPIs ready for plotting.
    
    Args:
        daily_df: Raw daily production DataFrame.
        
    Returns:
        DataFrame with computed KPI columns.
    """
    df = daily_df.copy()
    
    required_cols = ['OIL_RATE', 'WATER_RATE', 'GAS_RATE', 'WATER_CUT', 'GOR']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    return df


def compute_well_summary(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each well.
    
    Calculates latest production rates, historical averages, cumulative
    production, and identifies underperforming wells.
    
    Args:
        daily_df: Daily production DataFrame.
        
    Returns:
        DataFrame with well-level summary statistics.
    """
    df = daily_df.copy()
    df = df.sort_values(['NPD_WELL_BORE_NAME', 'DATEPRD'])
    
    max_date = df['DATEPRD'].max()
    six_months_ago = max_date - timedelta(days=180)
    twelve_months_ago = max_date - timedelta(days=365)
    
    summary_list = []
    
    for well_name, well_df in df.groupby('NPD_WELL_BORE_NAME'):
        producing_records = well_df[
            (well_df['OIL_RATE'].notna()) & 
            (well_df['OIL_RATE'] > 0)
        ]
        
        if len(producing_records) > 0:
            latest_record = producing_records.iloc[-1]
        else:
            latest_record = well_df.iloc[-1]
        
        latest_oil_rate = latest_record['OIL_RATE'] if pd.notna(latest_record['OIL_RATE']) else 0
        latest_water_rate = latest_record['WATER_RATE'] if pd.notna(latest_record['WATER_RATE']) else 0
        latest_water_cut = latest_record['WATER_CUT'] if pd.notna(latest_record['WATER_CUT']) else 0
        latest_gor = latest_record['GOR'] if pd.notna(latest_record['GOR']) else 0
        
        six_month_data = well_df[well_df['DATEPRD'] >= six_months_ago]
        twelve_month_data = well_df[well_df['DATEPRD'] >= twelve_months_ago]
        
        avg_oil_rate_6m = six_month_data['OIL_RATE'].mean() if len(six_month_data) > 0 else np.nan
        avg_oil_rate_12m = twelve_month_data['OIL_RATE'].mean() if len(twelve_month_data) > 0 else np.nan
        
        cum_oil = well_df['BORE_OIL_VOL_BBL'].sum() if 'BORE_OIL_VOL_BBL' in well_df.columns else well_df['BORE_OIL_VOL'].sum() * 6.2898
        cum_gas = well_df['BORE_GAS_VOL'].sum()
        cum_water = well_df['BORE_WAT_VOL_BBL'].sum() if 'BORE_WAT_VOL_BBL' in well_df.columns else well_df['BORE_WAT_VOL'].sum() * 6.2898
        
        underperforming = False
        if pd.notna(avg_oil_rate_12m) and avg_oil_rate_12m > 0:
            underperforming = latest_oil_rate < (0.7 * avg_oil_rate_12m)
        
        if 'WELL_TYPE' in well_df.columns:
            well_type = well_df['WELL_TYPE'].mode().iloc[0] if len(well_df['WELL_TYPE'].mode()) > 0 else 'OP'
        else:
            well_type = 'OP'
        
        summary_list.append({
            'NPD_WELL_BORE_NAME': well_name,
            'WELL_TYPE': well_type,
            'LATEST_OIL_RATE': latest_oil_rate,
            'LATEST_WATER_RATE': latest_water_rate,
            'LATEST_WATER_CUT': latest_water_cut,
            'LATEST_GOR': latest_gor,
            'AVG_OIL_RATE_6M': avg_oil_rate_6m,
            'AVG_OIL_RATE_12M': avg_oil_rate_12m,
            'CUM_OIL': cum_oil,
            'CUM_GAS': cum_gas,
            'CUM_WATER': cum_water,
            'UNDERPERFORMING': underperforming,
            'LAST_DATE': latest_record['DATEPRD']
        })
    
    summary_df = pd.DataFrame(summary_list)
    
    summary_df['OIL_LOSS'] = summary_df['AVG_OIL_RATE_12M'] - summary_df['LATEST_OIL_RATE']
    summary_df['OIL_LOSS'] = summary_df['OIL_LOSS'].clip(lower=0)
    
    return summary_df


def compute_downtime_proxy(daily_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Calculate a downtime proxy based on zero-production days.
    
    Args:
        daily_df: Daily production DataFrame.
        days: Number of days to look back for downtime calculation.
        
    Returns:
        DataFrame with well names and downtime day counts.
    """
    df = daily_df.copy()
    max_date = df['DATEPRD'].max()
    lookback_date = max_date - timedelta(days=days)
    
    recent_df = df[df['DATEPRD'] >= lookback_date]
    
    downtime_list = []
    for well_name, well_df in recent_df.groupby('NPD_WELL_BORE_NAME'):
        zero_days = ((well_df['BORE_OIL_VOL'] == 0) | (well_df['ON_STREAM_HRS'] == 0)).sum()
        total_days = len(well_df)
        
        downtime_list.append({
            'NPD_WELL_BORE_NAME': well_name,
            'ZERO_PRODUCTION_DAYS': zero_days,
            'TOTAL_DAYS': total_days,
            'DOWNTIME_RATIO': zero_days / total_days if total_days > 0 else 0
        })
    
    return pd.DataFrame(downtime_list)


def rank_intervention_candidates(
    well_summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    weight_oil_loss: float = 0.5,
    weight_water_cut: float = 0.3,
    weight_downtime: float = 0.2
) -> pd.DataFrame:
    """
    Rank wells as intervention candidates based on weighted scoring.
    
    Uses a multi-criteria scoring approach combining:
    - Oil production loss (current vs 12-month average)
    - Water cut (higher = potential water breakthrough)
    - Downtime (zero-production days as proxy for operational issues)
    
    Args:
        well_summary_df: Well summary DataFrame from compute_well_summary().
        daily_df: Daily production DataFrame for downtime calculation.
        weight_oil_loss: Weight for oil loss in scoring (0-1).
        weight_water_cut: Weight for water cut in scoring (0-1).
        weight_downtime: Weight for downtime in scoring (0-1).
        
    Returns:
        DataFrame with intervention scores and recommendations, sorted by score.
    """
    df = well_summary_df.copy()
    
    downtime_df = compute_downtime_proxy(daily_df)
    df = df.merge(downtime_df[['NPD_WELL_BORE_NAME', 'DOWNTIME_RATIO']], 
                  on='NPD_WELL_BORE_NAME', how='left')
    df['DOWNTIME_RATIO'] = df['DOWNTIME_RATIO'].fillna(0)
    
    oil_loss_max = df['OIL_LOSS'].max()
    df['OIL_LOSS_NORM'] = df['OIL_LOSS'] / oil_loss_max if oil_loss_max > 0 else 0
    
    water_cut_max = df['LATEST_WATER_CUT'].max()
    df['WATER_CUT_NORM'] = df['LATEST_WATER_CUT'] / water_cut_max if water_cut_max > 0 else 0
    
    df['SCORE'] = (
        weight_oil_loss * df['OIL_LOSS_NORM'] +
        weight_water_cut * df['WATER_CUT_NORM'] +
        weight_downtime * df['DOWNTIME_RATIO']
    )
    
    score_max = df['SCORE'].max()
    if score_max > 0:
        df['SCORE'] = df['SCORE'] / score_max
    
    def get_recommendation(row):
        oil_loss_high = row['OIL_LOSS_NORM'] > 0.5
        water_cut_high = row['WATER_CUT_NORM'] > 0.5
        downtime_high = row['DOWNTIME_RATIO'] > 0.3
        
        if oil_loss_high and water_cut_high:
            return "Water shutoff / conformance / zonal review"
        elif oil_loss_high and not water_cut_high:
            return "Mechanical integrity review / stimulation candidate"
        elif downtime_high:
            return "Operational review / CT cleanout candidate"
        elif water_cut_high:
            return "Monitor water breakthrough / consider water shutoff"
        else:
            return "Continue monitoring"
    
    df['RECOMMENDED_ACTION'] = df.apply(get_recommendation, axis=1)
    
    result_columns = [
        'NPD_WELL_BORE_NAME', 'LATEST_OIL_RATE', 'AVG_OIL_RATE_12M',
        'OIL_LOSS', 'LATEST_WATER_CUT', 'DOWNTIME_RATIO',
        'SCORE', 'RECOMMENDED_ACTION'
    ]
    
    result_df = df[result_columns].sort_values('SCORE', ascending=False).reset_index(drop=True)
    
    return result_df


def compute_production_volatility(daily_df: pd.DataFrame, days: int = 90) -> pd.DataFrame:
    """
    Calculate production volatility (standard deviation) for each well.
    
    High volatility may indicate operational issues or unstable wells.
    
    Args:
        daily_df: Daily production DataFrame.
        days: Number of days to look back.
        
    Returns:
        DataFrame with volatility metrics per well.
    """
    df = daily_df.copy()
    max_date = df['DATEPRD'].max()
    lookback_date = max_date - timedelta(days=days)
    
    recent_df = df[df['DATEPRD'] >= lookback_date]
    
    volatility_list = []
    for well_name, well_df in recent_df.groupby('NPD_WELL_BORE_NAME'):
        oil_std = well_df['OIL_RATE'].std()
        oil_mean = well_df['OIL_RATE'].mean()
        cv = oil_std / oil_mean if oil_mean > 0 else 0
        
        volatility_list.append({
            'NPD_WELL_BORE_NAME': well_name,
            'OIL_RATE_STD': oil_std,
            'OIL_RATE_MEAN': oil_mean,
            'COEFFICIENT_OF_VARIATION': cv
        })
    
    return pd.DataFrame(volatility_list)


def get_field_totals(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily production to field level.
    
    Args:
        daily_df: Daily production DataFrame.
        
    Returns:
        DataFrame with daily field totals.
    """
    field_df = daily_df.groupby('DATEPRD').agg({
        'BORE_OIL_VOL': 'sum',
        'BORE_GAS_VOL': 'sum',
        'BORE_WAT_VOL': 'sum',
        'ON_STREAM_HRS': 'mean'
    }).reset_index()
    
    field_df['TOTAL_OIL_RATE'] = np.where(
        field_df['ON_STREAM_HRS'] > 0,
        field_df['BORE_OIL_VOL'] / (field_df['ON_STREAM_HRS'] / 24),
        np.nan
    )
    
    field_df['TOTAL_WATER_RATE'] = np.where(
        field_df['ON_STREAM_HRS'] > 0,
        field_df['BORE_WAT_VOL'] / (field_df['ON_STREAM_HRS'] / 24),
        np.nan
    )
    
    field_df['TOTAL_GAS_RATE'] = np.where(
        field_df['ON_STREAM_HRS'] > 0,
        field_df['BORE_GAS_VOL'] / (field_df['ON_STREAM_HRS'] / 24),
        np.nan
    )
    
    total_liquid = field_df['BORE_OIL_VOL'] + field_df['BORE_WAT_VOL']
    field_df['FIELD_WATER_CUT'] = np.where(
        total_liquid > 0,
        field_df['BORE_WAT_VOL'] / total_liquid,
        np.nan
    )
    
    return field_df
