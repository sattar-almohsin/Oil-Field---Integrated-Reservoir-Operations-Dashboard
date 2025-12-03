"""
Decline Curve Analysis (DCA) module for production forecasting.

This module implements Arps decline curve models for estimating future
production and calculating Estimated Ultimate Recovery (EUR).
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict, Any
from datetime import datetime, timedelta


def arps_exponential(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    """
    Exponential decline curve model.
    
    q(t) = qi * exp(-di * t)
    
    Args:
        t: Time array (months or days from start).
        qi: Initial production rate.
        di: Decline rate (nominal, per time unit).
        
    Returns:
        Production rate array.
    """
    return qi * np.exp(-di * t)


def arps_harmonic(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    """
    Harmonic decline curve model (b=1 case of hyperbolic).
    
    q(t) = qi / (1 + di * t)
    
    Args:
        t: Time array.
        qi: Initial production rate.
        di: Decline rate.
        
    Returns:
        Production rate array.
    """
    return qi / (1 + di * t)


def arps_hyperbolic(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """
    Hyperbolic decline curve model (general Arps equation).
    
    q(t) = qi / (1 + b * di * t)^(1/b)
    
    Args:
        t: Time array.
        qi: Initial production rate.
        di: Initial decline rate.
        b: Decline exponent (0 < b < 1 for hyperbolic).
        
    Returns:
        Production rate array.
    """
    b = np.clip(b, 0.001, 0.999)
    return qi / np.power(1 + b * di * t, 1/b)


def fit_exponential_decline(
    time_array: np.ndarray, 
    rate_array: np.ndarray
) -> Tuple[float, float, float]:
    """
    Fit exponential decline curve to production data.
    
    Args:
        time_array: Time values (months from start).
        rate_array: Production rate values.
        
    Returns:
        Tuple of (qi, di, r_squared).
    """
    mask = ~np.isnan(rate_array) & (rate_array > 0)
    t_clean = time_array[mask]
    q_clean = rate_array[mask]
    
    if len(t_clean) < 3:
        return np.nan, np.nan, np.nan
    
    try:
        qi_init = q_clean[0]
        di_init = 0.05
        
        popt, _ = curve_fit(
            arps_exponential, 
            t_clean, 
            q_clean,
            p0=[qi_init, di_init],
            bounds=([0, 0.0001], [qi_init * 3, 1.0]),
            maxfev=5000
        )
        
        qi, di = popt
        
        q_pred = arps_exponential(t_clean, qi, di)
        ss_res = np.sum((q_clean - q_pred) ** 2)
        ss_tot = np.sum((q_clean - np.mean(q_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return qi, di, r_squared
        
    except Exception:
        return np.nan, np.nan, np.nan


def fit_hyperbolic_decline(
    time_array: np.ndarray, 
    rate_array: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Fit hyperbolic decline curve to production data.
    
    Args:
        time_array: Time values (months from start).
        rate_array: Production rate values.
        
    Returns:
        Tuple of (qi, di, b, r_squared).
    """
    mask = ~np.isnan(rate_array) & (rate_array > 0)
    t_clean = time_array[mask]
    q_clean = rate_array[mask]
    
    if len(t_clean) < 4:
        return np.nan, np.nan, np.nan, np.nan
    
    try:
        qi_init = q_clean[0]
        di_init = 0.05
        b_init = 0.5
        
        popt, _ = curve_fit(
            arps_hyperbolic, 
            t_clean, 
            q_clean,
            p0=[qi_init, di_init, b_init],
            bounds=([0, 0.0001, 0.001], [qi_init * 3, 1.0, 0.999]),
            maxfev=5000
        )
        
        qi, di, b = popt
        
        q_pred = arps_hyperbolic(t_clean, qi, di, b)
        ss_res = np.sum((q_clean - q_pred) ** 2)
        ss_tot = np.sum((q_clean - np.mean(q_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return qi, di, b, r_squared
        
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def calculate_eur_exponential(
    qi: float, 
    di: float, 
    economic_limit: float = 10.0,
    current_cum: float = 0.0
) -> float:
    """
    Calculate EUR for exponential decline.
    
    EUR = qi / di (to economic limit or infinity approximation)
    
    Args:
        qi: Initial rate.
        di: Decline rate.
        economic_limit: Minimum economic rate (Sm3/d).
        current_cum: Current cumulative production.
        
    Returns:
        Estimated Ultimate Recovery (Sm3).
    """
    if np.isnan(qi) or np.isnan(di) or di <= 0:
        return np.nan
    
    if qi <= economic_limit:
        return current_cum
    
    t_limit = -np.log(economic_limit / qi) / di
    
    eur_remaining = (qi / di) * (1 - np.exp(-di * t_limit))
    
    return current_cum + eur_remaining * 30


def calculate_eur_hyperbolic(
    qi: float, 
    di: float, 
    b: float,
    economic_limit: float = 10.0,
    current_cum: float = 0.0,
    max_time: float = 360
) -> float:
    """
    Calculate EUR for hyperbolic decline (numerical integration).
    
    Args:
        qi: Initial rate.
        di: Initial decline rate.
        b: Decline exponent.
        economic_limit: Minimum economic rate (Sm3/d).
        current_cum: Current cumulative production.
        max_time: Maximum forecast time (months).
        
    Returns:
        Estimated Ultimate Recovery (Sm3).
    """
    if np.isnan(qi) or np.isnan(di) or np.isnan(b) or di <= 0:
        return np.nan
    
    if qi <= economic_limit:
        return current_cum
    
    t_array = np.linspace(0, max_time, 1000)
    q_array = arps_hyperbolic(t_array, qi, di, b)
    
    idx_limit = np.where(q_array < economic_limit)[0]
    if len(idx_limit) > 0:
        t_limit_idx = idx_limit[0]
        t_array = t_array[:t_limit_idx + 1]
        q_array = q_array[:t_limit_idx + 1]
    
    eur_remaining = np.trapezoid(q_array, t_array) * 30
    
    return current_cum + eur_remaining


def prepare_dca_data(
    daily_df: pd.DataFrame, 
    well_name: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare monthly production data for DCA fitting.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Well name to analyze.
        
    Returns:
        Tuple of (time_months, oil_rate, monthly_df).
    """
    well_df = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name].copy()
    well_df = well_df.sort_values('DATEPRD')
    
    well_df['YEAR_MONTH'] = well_df['DATEPRD'].dt.to_period('M')
    
    monthly_df = well_df.groupby('YEAR_MONTH').agg({
        'OIL_RATE': 'mean',
        'BORE_OIL_VOL': 'sum',
        'DATEPRD': 'first'
    }).reset_index()
    
    monthly_df = monthly_df[monthly_df['OIL_RATE'].notna() & (monthly_df['OIL_RATE'] > 0)]
    monthly_df = monthly_df.sort_values('DATEPRD')
    
    if len(monthly_df) > 0:
        start_date = monthly_df['DATEPRD'].iloc[0]
        monthly_df['TIME_MONTHS'] = (
            (monthly_df['DATEPRD'] - start_date).dt.days / 30.44
        ).astype(float)
    else:
        monthly_df['TIME_MONTHS'] = []
    
    time_array = monthly_df['TIME_MONTHS'].values
    rate_array = monthly_df['OIL_RATE'].values
    
    return time_array, rate_array, monthly_df


def run_dca_analysis(
    daily_df: pd.DataFrame, 
    well_name: str,
    forecast_months: int = 60,
    economic_limit: float = 10.0
) -> Dict[str, Any]:
    """
    Run complete DCA analysis for a well.
    
    Args:
        daily_df: Daily production DataFrame.
        well_name: Well name to analyze.
        forecast_months: Months to forecast.
        economic_limit: Economic limit rate (Sm3/d).
        
    Returns:
        Dictionary with DCA results and forecast data.
    """
    time_array, rate_array, monthly_df = prepare_dca_data(daily_df, well_name)
    
    if len(time_array) < 6:
        return {
            'success': False,
            'error': 'Insufficient data for DCA (need at least 6 months)',
            'well_name': well_name
        }
    
    qi_exp, di_exp, r2_exp = fit_exponential_decline(time_array, rate_array)
    qi_hyp, di_hyp, b_hyp, r2_hyp = fit_hyperbolic_decline(time_array, rate_array)
    
    current_cum = daily_df[daily_df['NPD_WELL_BORE_NAME'] == well_name]['BORE_OIL_VOL'].sum()
    
    eur_exp = calculate_eur_exponential(qi_exp, di_exp, economic_limit, current_cum)
    eur_hyp = calculate_eur_hyperbolic(qi_hyp, di_hyp, b_hyp, economic_limit, current_cum)
    
    last_time = time_array[-1] if len(time_array) > 0 else 0
    forecast_t = np.linspace(0, last_time + forecast_months, 200)
    
    forecast_exp = arps_exponential(forecast_t, qi_exp, di_exp) if not np.isnan(qi_exp) else np.full_like(forecast_t, np.nan)
    forecast_hyp = arps_hyperbolic(forecast_t, qi_hyp, di_hyp, b_hyp) if not np.isnan(qi_hyp) else np.full_like(forecast_t, np.nan)
    
    last_date = monthly_df['DATEPRD'].iloc[-1] if len(monthly_df) > 0 else datetime.now()
    forecast_dates = pd.date_range(
        start=monthly_df['DATEPRD'].iloc[0] if len(monthly_df) > 0 else datetime.now(),
        periods=len(forecast_t),
        freq='M'
    )
    
    results = {
        'success': True,
        'well_name': well_name,
        'historical_data': {
            'time': time_array,
            'rate': rate_array,
            'dates': monthly_df['DATEPRD'].values
        },
        'exponential': {
            'qi': qi_exp,
            'di': di_exp,
            'r_squared': r2_exp,
            'eur': eur_exp,
            'forecast_time': forecast_t,
            'forecast_rate': forecast_exp,
            'decline_rate_pct': di_exp * 100 if not np.isnan(di_exp) else np.nan
        },
        'hyperbolic': {
            'qi': qi_hyp,
            'di': di_hyp,
            'b': b_hyp,
            'r_squared': r2_hyp,
            'eur': eur_hyp,
            'forecast_time': forecast_t,
            'forecast_rate': forecast_hyp,
            'decline_rate_pct': di_hyp * 100 if not np.isnan(di_hyp) else np.nan
        },
        'current_cumulative': current_cum,
        'economic_limit': economic_limit,
        'forecast_months': forecast_months,
        'forecast_dates': forecast_dates,
        'best_fit': 'hyperbolic' if (not np.isnan(r2_hyp) and (np.isnan(r2_exp) or r2_hyp > r2_exp)) else 'exponential'
    }
    
    return results


def get_dca_summary_all_wells(
    daily_df: pd.DataFrame,
    economic_limit: float = 10.0
) -> pd.DataFrame:
    """
    Calculate DCA summary for all wells.
    
    Args:
        daily_df: Daily production DataFrame.
        economic_limit: Economic limit rate.
        
    Returns:
        DataFrame with DCA results for each well.
    """
    wells = daily_df['NPD_WELL_BORE_NAME'].unique()
    
    summary_list = []
    for well in wells:
        results = run_dca_analysis(daily_df, well, economic_limit=economic_limit)
        
        if results['success']:
            best_fit = results['best_fit']
            
            summary_list.append({
                'NPD_WELL_BORE_NAME': well,
                'BEST_FIT_MODEL': best_fit.title(),
                'QI_RATE': results[best_fit]['qi'],
                'DECLINE_RATE_PCT': results[best_fit]['decline_rate_pct'],
                'B_FACTOR': results['hyperbolic']['b'] if best_fit == 'hyperbolic' else np.nan,
                'R_SQUARED': results[best_fit]['r_squared'],
                'CURRENT_CUM': results['current_cumulative'],
                'EUR': results[best_fit]['eur'],
                'REMAINING_RESERVES': results[best_fit]['eur'] - results['current_cumulative']
            })
        else:
            summary_list.append({
                'NPD_WELL_BORE_NAME': well,
                'BEST_FIT_MODEL': 'N/A',
                'QI_RATE': np.nan,
                'DECLINE_RATE_PCT': np.nan,
                'B_FACTOR': np.nan,
                'R_SQUARED': np.nan,
                'CURRENT_CUM': daily_df[daily_df['NPD_WELL_BORE_NAME'] == well]['BORE_OIL_VOL'].sum(),
                'EUR': np.nan,
                'REMAINING_RESERVES': np.nan
            })
    
    return pd.DataFrame(summary_list)
