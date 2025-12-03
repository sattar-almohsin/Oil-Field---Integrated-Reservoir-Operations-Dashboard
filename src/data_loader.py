"""
Data loading and cleaning module for Volve production data.

This module provides functions to load and clean daily and monthly production data
from the Volve field Excel file. It includes Streamlit caching for performance
optimization.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional

SM3_TO_BBL = 6.2898


@st.cache_data
def load_daily_production(path: str) -> pd.DataFrame:
    """
    Load and clean daily production data from the Volve Excel file.
    
    Args:
        path: Path to the Excel file containing production data.
        
    Returns:
        Cleaned DataFrame with daily production data and derived columns.
    """
    df = pd.read_excel(path, sheet_name="Daily Production Data")
    
    df['DATEPRD'] = pd.to_datetime(df['DATEPRD'], errors='coerce')
    
    numeric_columns = [
        'ON_STREAM_HRS', 'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL',
        'AVG_WHP_P', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
        'AVG_CHOKE_SIZE_P', 'AVG_ANNULUS_PRESS', 'AVG_DP_TUBING',
        'DP_CHOKE_SIZE', 'FLOW_KIND'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['NPD_WELL_BORE_NAME', 'DATEPRD'])
    
    df['OIL_RATE'] = np.where(
        df['ON_STREAM_HRS'] > 0,
        df['BORE_OIL_VOL'] / (df['ON_STREAM_HRS'] / 24) * SM3_TO_BBL,
        np.nan
    )
    
    df['WATER_RATE'] = np.where(
        df['ON_STREAM_HRS'] > 0,
        df['BORE_WAT_VOL'] / (df['ON_STREAM_HRS'] / 24) * SM3_TO_BBL,
        np.nan
    )
    
    df['GAS_RATE'] = np.where(
        df['ON_STREAM_HRS'] > 0,
        df['BORE_GAS_VOL'] / (df['ON_STREAM_HRS'] / 24),
        np.nan
    )
    
    df['BORE_OIL_VOL_BBL'] = df['BORE_OIL_VOL'] * SM3_TO_BBL
    df['BORE_WAT_VOL_BBL'] = df['BORE_WAT_VOL'] * SM3_TO_BBL
    
    total_liquid = df['BORE_OIL_VOL'] + df['BORE_WAT_VOL']
    df['WATER_CUT'] = np.where(
        total_liquid > 0,
        df['BORE_WAT_VOL'] / total_liquid,
        np.nan
    )
    
    df['GOR'] = np.where(
        df['BORE_OIL_VOL'] > 0,
        df['BORE_GAS_VOL'] / df['BORE_OIL_VOL'],
        np.nan
    )
    
    df = df.sort_values(['NPD_WELL_BORE_NAME', 'DATEPRD']).reset_index(drop=True)
    
    return df


@st.cache_data
def load_monthly_production(path: str) -> pd.DataFrame:
    """
    Load and clean monthly production data from the Volve Excel file.
    
    Args:
        path: Path to the Excel file containing production data.
        
    Returns:
        Cleaned DataFrame with monthly production data.
    """
    df = pd.read_excel(path, sheet_name="Monthly Production Data")
    
    if 'Year' not in df.columns or 'Month' not in df.columns:
        df = pd.read_excel(path, sheet_name="Monthly Production Data", header=0, skiprows=[1])
    
    first_row = df.iloc[0] if len(df) > 0 else None
    if first_row is not None and pd.isna(first_row.get('Year', 0)):
        df = df.iloc[1:].reset_index(drop=True)
    
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    if 'Month' in df.columns:
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').astype('Int64')
    
    df = df.dropna(subset=['Year', 'Month'])
    
    df['DATE'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01',
        errors='coerce'
    )
    
    numeric_columns = ['Oil', 'Gas', 'Water', 'GI', 'WI']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Wellbore name' in df.columns:
        df = df.rename(columns={'Wellbore name': 'NPD_WELL_BORE_NAME'})
    
    df = df.sort_values(['NPD_WELL_BORE_NAME', 'DATE']).reset_index(drop=True)
    
    return df


@st.cache_data
def load_all_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both daily and monthly production data from the Volve Excel file.
    
    Args:
        path: Path to the Excel file containing production data.
        
    Returns:
        Tuple of (daily_df, monthly_df) DataFrames.
    """
    daily_df = load_daily_production(path)
    monthly_df = load_monthly_production(path)
    
    return daily_df, monthly_df


def get_well_list(daily_df: pd.DataFrame) -> list:
    """
    Get list of unique well names from the daily production data.
    
    Args:
        daily_df: Daily production DataFrame.
        
    Returns:
        Sorted list of unique well names.
    """
    return sorted(daily_df['NPD_WELL_BORE_NAME'].unique().tolist())


def get_date_range(daily_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the date range of the production data.
    
    Args:
        daily_df: Daily production DataFrame.
        
    Returns:
        Tuple of (min_date, max_date).
    """
    return daily_df['DATEPRD'].min(), daily_df['DATEPRD'].max()


def get_well_types(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get a summary of well types from the production data.
    
    Args:
        daily_df: Daily production DataFrame.
        
    Returns:
        DataFrame with well type counts.
    """
    if 'WELL_TYPE' in daily_df.columns:
        well_types = daily_df.groupby('NPD_WELL_BORE_NAME')['WELL_TYPE'].first().reset_index()
        return well_types
    else:
        wells = daily_df['NPD_WELL_BORE_NAME'].unique()
        return pd.DataFrame({
            'NPD_WELL_BORE_NAME': wells,
            'WELL_TYPE': ['OP'] * len(wells)
        })
