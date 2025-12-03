"""
PDF Reporting Module

Generates professional PDF reports for reservoir surveillance,
intervention recommendations, and production summaries.
"""

import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import datetime
from typing import Dict, Any, Optional, List
import io
import tempfile
import os


class ReservoirReport(FPDF):
    """Custom PDF class for reservoir reports."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Add header to each page."""
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Volve Field - Reservoir Operations Report', border=0, ln=True, align='C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', border=0, ln=True, align='C')
        self.ln(5)
    
    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')
    
    def chapter_title(self, title: str):
        """Add a chapter title."""
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 10, title, border=0, ln=True, fill=True)
        self.ln(2)
    
    def section_title(self, title: str):
        """Add a section title."""
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 8, title, border=0, ln=True)
        self.ln(1)
    
    def body_text(self, text: str):
        """Add body text."""
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def add_kpi_box(self, label: str, value: str, unit: str = ""):
        """Add a KPI metric box."""
        self.set_font('Helvetica', 'B', 9)
        self.cell(45, 6, label, border=1)
        self.set_font('Helvetica', '', 9)
        display_value = f"{value} {unit}".strip()
        self.cell(45, 6, display_value, border=1, ln=True)
    
    def add_table(self, headers: List[str], data: List[List[str]], col_widths: Optional[List[int]] = None):
        """Add a data table."""
        if col_widths is None:
            col_widths = [int(190 / len(headers))] * len(headers)
        
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(200, 200, 200)
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, str(header), border=1, fill=True)
        self.ln()
        
        self.set_font('Helvetica', '', 8)
        for row in data:
            for i, cell in enumerate(row):
                cell_text = str(cell)[:25]
                self.cell(col_widths[i], 6, cell_text, border=1)
            self.ln()


def generate_surveillance_report(
    daily_df: pd.DataFrame,
    well_summary_df: pd.DataFrame,
    alerts_df: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Generate a field surveillance summary report.
    
    Args:
        daily_df: Daily production dataframe
        well_summary_df: Well summary dataframe
        alerts_df: Optional alerts dataframe
    
    Returns:
        PDF file as bytes
    """
    pdf = ReservoirReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title("Field Overview")
    
    total_oil = well_summary_df['LATEST_OIL_RATE'].sum() if 'LATEST_OIL_RATE' in well_summary_df.columns else 0
    total_water = well_summary_df['LATEST_WATER_RATE'].sum() if 'LATEST_WATER_RATE' in well_summary_df.columns else 0
    total_wells = len(well_summary_df)
    underperforming = well_summary_df['UNDERPERFORMING'].sum() if 'UNDERPERFORMING' in well_summary_df.columns else 0
    
    pdf.add_kpi_box("Total Oil Rate", f"{total_oil:.0f}", "Sm3/d")
    pdf.add_kpi_box("Total Water Rate", f"{total_water:.0f}", "Sm3/d")
    pdf.add_kpi_box("Active Wells", str(total_wells), "")
    pdf.add_kpi_box("Underperforming Wells", str(int(underperforming)), "")
    
    if total_oil + total_water > 0:
        field_wc = total_water / (total_oil + total_water) * 100
        pdf.add_kpi_box("Field Water Cut", f"{field_wc:.1f}", "%")
    
    pdf.ln(5)
    
    pdf.chapter_title("Well Summary")
    
    headers = ['Well', 'Oil Rate', 'Water Rate', 'Water Cut', 'Status']
    
    table_data = []
    for _, row in well_summary_df.iterrows():
        well = row.get('NPD_WELL_BORE_NAME', 'Unknown')[:20]
        oil_rate = f"{row.get('LATEST_OIL_RATE', 0):.0f}"
        water_rate = f"{row.get('LATEST_WATER_RATE', 0):.0f}"
        wc = row.get('LATEST_WATER_CUT', 0)
        water_cut = f"{wc*100:.1f}%" if not pd.isna(wc) else "N/A"
        status = "Underperforming" if row.get('UNDERPERFORMING', False) else "Normal"
        table_data.append([well, oil_rate, water_rate, water_cut, status])
    
    pdf.add_table(headers, table_data, [55, 30, 30, 35, 40])
    
    if alerts_df is not None and len(alerts_df) > 0:
        pdf.add_page()
        pdf.chapter_title("Active Alerts")
        
        pdf.body_text(f"Total Active Alerts: {len(alerts_df)}")
        pdf.body_text(f"High Severity: {(alerts_df['severity'] == 'HIGH').sum()}")
        pdf.body_text(f"Medium Severity: {(alerts_df['severity'] == 'MEDIUM').sum()}")
        pdf.ln(3)
        
        headers = ['Well', 'Type', 'Severity', 'Message']
        
        table_data = []
        for _, alert in alerts_df.head(15).iterrows():
            well = str(alert.get('well', ''))[:15]
            alert_type = str(alert.get('type', '')).replace('_', ' ')[:15]
            severity = str(alert.get('severity', ''))
            message = str(alert.get('message', ''))[:40]
            table_data.append([well, alert_type, severity, message])
        
        pdf.add_table(headers, table_data, [35, 35, 25, 95])
    
    output = io.BytesIO()
    pdf_bytes = pdf.output()
    output.write(pdf_bytes)
    return output.getvalue()


def generate_intervention_report(
    intervention_df: pd.DataFrame,
    economic_df: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Generate an intervention recommendations report.
    
    Args:
        intervention_df: Intervention candidates dataframe
        economic_df: Optional economic analysis dataframe
    
    Returns:
        PDF file as bytes
    """
    pdf = ReservoirReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title("Intervention Candidates Summary")
    
    pdf.body_text(
        "This report summarizes the recommended well interventions based on "
        "production performance analysis. Wells are ranked by a composite score "
        "considering oil loss, water cut, and downtime."
    )
    pdf.ln(3)
    
    pdf.section_title("Top Intervention Candidates")
    
    headers = ['Rank', 'Well', 'Score', 'Recommended Action']
    
    table_data = []
    for rank, (_, row) in enumerate(intervention_df.head(10).iterrows(), 1):
        well = str(row.get('NPD_WELL_BORE_NAME', ''))[:20]
        score = f"{row.get('INTERVENTION_SCORE', 0):.3f}"
        action = str(row.get('RECOMMENDED_ACTION', ''))[:50]
        table_data.append([str(rank), well, score, action])
    
    pdf.add_table(headers, table_data, [15, 45, 25, 105])
    
    pdf.ln(5)
    pdf.section_title("Intervention Details")
    
    for _, row in intervention_df.head(5).iterrows():
        well = row.get('NPD_WELL_BORE_NAME', 'Unknown')
        
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, f"Well: {well}", ln=True)
        pdf.set_font('Helvetica', '', 9)
        
        oil_loss = row.get('OIL_LOSS', 0)
        water_cut = row.get('LATEST_WATER_CUT', 0)
        downtime = row.get('DOWNTIME_RATIO', 0)
        action = row.get('RECOMMENDED_ACTION', 'N/A')
        
        pdf.cell(0, 5, f"  Oil Loss: {oil_loss:.0f} Sm3/d", ln=True)
        pdf.cell(0, 5, f"  Water Cut: {water_cut*100:.1f}%" if not pd.isna(water_cut) else "  Water Cut: N/A", ln=True)
        pdf.cell(0, 5, f"  Downtime Ratio: {downtime*100:.1f}%" if not pd.isna(downtime) else "  Downtime Ratio: N/A", ln=True)
        pdf.cell(0, 5, f"  Recommended: {action}", ln=True)
        pdf.ln(3)
    
    if economic_df is not None and len(economic_df) > 0:
        pdf.add_page()
        pdf.chapter_title("Economic Analysis")
        
        profitable = economic_df['PROFITABLE'].sum() if 'PROFITABLE' in economic_df.columns else 0
        total = len(economic_df)
        
        pdf.body_text(f"Of {total} proposed interventions, {profitable} have positive NPV.")
        pdf.ln(3)
        
        headers = ['Well', 'Intervention', 'Cost', 'NPV', 'IRR', 'Payback']
        
        table_data = []
        for _, row in economic_df.head(10).iterrows():
            well = str(row.get('NPD_WELL_BORE_NAME', ''))[:15]
            intervention = str(row.get('INTERVENTION_TYPE', ''))[:15]
            cost = f"${row.get('INTERVENTION_COST', 0)/1000:.0f}K"
            npv = f"${row.get('NPV', 0)/1000:.0f}K" if not pd.isna(row.get('NPV')) else "N/A"
            irr = f"{row.get('IRR', 0)*100:.1f}%" if not pd.isna(row.get('IRR')) else "N/A"
            payback = f"{row.get('PAYBACK_YEARS', 0):.1f}y" if not pd.isna(row.get('PAYBACK_YEARS')) else "N/A"
            table_data.append([well, intervention, cost, npv, irr, payback])
        
        pdf.add_table(headers, table_data, [35, 35, 25, 30, 25, 25])
    
    output = io.BytesIO()
    pdf_bytes = pdf.output()
    output.write(pdf_bytes)
    return output.getvalue()


def generate_dca_report(dca_summary_df: pd.DataFrame) -> bytes:
    """
    Generate a Decline Curve Analysis report.
    
    Args:
        dca_summary_df: DCA summary dataframe
    
    Returns:
        PDF file as bytes
    """
    pdf = ReservoirReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title("Decline Curve Analysis Summary")
    
    pdf.body_text(
        "This report summarizes the decline curve analysis for all wells in the field. "
        "Analysis uses Arps decline models (exponential and hyperbolic) to estimate "
        "ultimate recovery (EUR) and remaining reserves."
    )
    pdf.ln(3)
    
    total_eur = dca_summary_df['EUR'].sum() / 1000 if 'EUR' in dca_summary_df.columns else 0
    total_remaining = dca_summary_df['REMAINING_RESERVES'].sum() / 1000 if 'REMAINING_RESERVES' in dca_summary_df.columns else 0
    
    pdf.add_kpi_box("Total Field EUR", f"{total_eur:.0f}", "MSm3")
    pdf.add_kpi_box("Remaining Reserves", f"{total_remaining:.0f}", "MSm3")
    
    pdf.ln(5)
    pdf.section_title("Well-by-Well Analysis")
    
    headers = ['Well', 'Model', 'qi (Sm3/d)', 'Decline (%/mo)', 'EUR (MSm3)', 'Remaining']
    
    valid_df = dca_summary_df[dca_summary_df['EUR'].notna()]
    
    table_data = []
    for _, row in valid_df.iterrows():
        well = str(row.get('NPD_WELL_BORE_NAME', ''))[:20]
        model = str(row.get('BEST_MODEL', ''))[:10]
        qi = f"{row.get('QI_RATE', 0):.0f}"
        decline = f"{row.get('DECLINE_RATE_PCT', 0):.2f}"
        eur = f"{row.get('EUR', 0)/1000:.1f}"
        remaining = f"{row.get('REMAINING_RESERVES', 0)/1000:.1f}"
        table_data.append([well, model, qi, decline, eur, remaining])
    
    pdf.add_table(headers, table_data, [45, 25, 25, 30, 30, 30])
    
    pdf.ln(5)
    pdf.section_title("Methodology")
    
    pdf.body_text(
        "The analysis uses the following decline curve models:\n"
        "- Exponential: q(t) = qi * exp(-Di * t)\n"
        "- Hyperbolic: q(t) = qi / (1 + b*Di*t)^(1/b)\n\n"
        "EUR is calculated by integrating the decline curve from current time "
        "to economic limit (default: 10 Sm3/d)."
    )
    
    output = io.BytesIO()
    pdf_bytes = pdf.output()
    output.write(pdf_bytes)
    return output.getvalue()


def generate_well_test_report(
    well_name: str,
    test_results: Dict[str, Any]
) -> bytes:
    """
    Generate a well test analysis report.
    
    Args:
        well_name: Name of the well
        test_results: Dictionary with test analysis results
    
    Returns:
        PDF file as bytes
    """
    pdf = ReservoirReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.chapter_title(f"Well Test Report - {well_name}")
    
    test_type = test_results.get('test_type', 'Unknown')
    pdf.body_text(f"Test Type: {test_type}")
    pdf.ln(3)
    
    pdf.section_title("Analysis Results")
    
    k = test_results.get('permeability_md', 0)
    skin = test_results.get('skin_factor', 0)
    pi = test_results.get('productivity_index', 0)
    ws = test_results.get('wellbore_storage', 0)
    
    pdf.add_kpi_box("Permeability", f"{k:.1f}", "mD")
    pdf.add_kpi_box("Skin Factor", f"{skin:.2f}", "")
    
    if not np.isnan(pi):
        pdf.add_kpi_box("Productivity Index", f"{pi:.2f}", "")
    if not np.isnan(ws):
        pdf.add_kpi_box("Wellbore Storage", f"{ws:.4f}", "")
    
    pdf.ln(5)
    pdf.section_title("Flow Regime Analysis")
    
    flow_regimes = test_results.get('flow_regimes', {})
    
    ws_detected = "Detected" if flow_regimes.get('wellbore_storage', False) else "Not Detected"
    rf_detected = "Detected" if flow_regimes.get('radial_flow', False) else "Not Detected"
    bd_detected = "Detected" if flow_regimes.get('boundary_effect', False) else "Not Detected"
    
    pdf.body_text(f"Wellbore Storage: {ws_detected}")
    pdf.body_text(f"Radial Flow: {rf_detected}")
    pdf.body_text(f"Boundary Effect: {bd_detected}")
    
    pdf.ln(5)
    pdf.section_title("Interpretation")
    
    if k < 1:
        k_interp = "Tight formation - may need hydraulic fracturing"
    elif k < 10:
        k_interp = "Low permeability - consider matrix stimulation"
    elif k < 100:
        k_interp = "Moderate permeability - good producer potential"
    else:
        k_interp = "High permeability - excellent reservoir quality"
    
    if skin < 0:
        s_interp = "Stimulated well (acidized/fractured)"
    elif skin < 5:
        s_interp = "Undamaged or slightly damaged well"
    elif skin < 10:
        s_interp = "Moderate formation damage"
    else:
        s_interp = "Significant formation damage - stimulation recommended"
    
    pdf.body_text(f"Permeability: {k_interp}")
    pdf.body_text(f"Skin Factor: {s_interp}")
    
    output = io.BytesIO()
    pdf_bytes = pdf.output()
    output.write(pdf_bytes)
    return output.getvalue()


def generate_executive_summary(
    daily_df: pd.DataFrame,
    well_summary_df: pd.DataFrame,
    intervention_df: pd.DataFrame,
    dca_summary_df: Optional[pd.DataFrame] = None,
    alerts_df: Optional[pd.DataFrame] = None
) -> bytes:
    """
    Generate a comprehensive executive summary report.
    
    Args:
        daily_df: Daily production dataframe
        well_summary_df: Well summary dataframe
        intervention_df: Intervention candidates dataframe
        dca_summary_df: Optional DCA summary dataframe
        alerts_df: Optional alerts dataframe
    
    Returns:
        PDF file as bytes
    """
    pdf = ReservoirReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 15, 'Volve Field Executive Summary', ln=True, align='C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 10, f'Report Date: {datetime.now().strftime("%B %d, %Y")}', ln=True, align='C')
    pdf.ln(10)
    
    pdf.chapter_title("1. Production Overview")
    
    total_oil = well_summary_df['LATEST_OIL_RATE'].sum() if 'LATEST_OIL_RATE' in well_summary_df.columns else 0
    total_water = well_summary_df['LATEST_WATER_RATE'].sum() if 'LATEST_WATER_RATE' in well_summary_df.columns else 0
    total_wells = len(well_summary_df)
    underperforming = int(well_summary_df['UNDERPERFORMING'].sum()) if 'UNDERPERFORMING' in well_summary_df.columns else 0
    
    pdf.body_text(f"Current field oil production: {total_oil:.0f} Sm3/d")
    pdf.body_text(f"Current field water production: {total_water:.0f} Sm3/d")
    pdf.body_text(f"Active wells: {total_wells}")
    pdf.body_text(f"Underperforming wells requiring attention: {underperforming}")
    
    pdf.ln(3)
    pdf.chapter_title("2. Key Alerts & Issues")
    
    if alerts_df is not None and len(alerts_df) > 0:
        high_severity = (alerts_df['severity'] == 'HIGH').sum()
        pdf.body_text(f"Total active alerts: {len(alerts_df)}")
        pdf.body_text(f"High severity alerts requiring immediate action: {high_severity}")
        
        if high_severity > 0:
            pdf.ln(2)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 5, "Critical Issues:", ln=True)
            pdf.set_font('Helvetica', '', 9)
            
            for _, alert in alerts_df[alerts_df['severity'] == 'HIGH'].head(3).iterrows():
                pdf.cell(0, 5, f"  - {alert['well']}: {alert['message'][:60]}", ln=True)
    else:
        pdf.body_text("No active alerts - all wells operating within normal parameters.")
    
    pdf.ln(3)
    pdf.chapter_title("3. Recommended Interventions")
    
    pdf.body_text(f"Wells identified for intervention: {len(intervention_df)}")
    
    if len(intervention_df) > 0:
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 5, "Top 3 Priority Wells:", ln=True)
        pdf.set_font('Helvetica', '', 9)
        
        for _, row in intervention_df.head(3).iterrows():
            well = row.get('NPD_WELL_BORE_NAME', 'Unknown')
            action = row.get('RECOMMENDED_ACTION', 'N/A')
            pdf.cell(0, 5, f"  - {well}: {action}", ln=True)
    
    if dca_summary_df is not None and len(dca_summary_df) > 0:
        pdf.ln(3)
        pdf.chapter_title("4. Reserves Outlook")
        
        total_eur = dca_summary_df['EUR'].sum() / 1000 if 'EUR' in dca_summary_df.columns else 0
        total_remaining = dca_summary_df['REMAINING_RESERVES'].sum() / 1000 if 'REMAINING_RESERVES' in dca_summary_df.columns else 0
        
        pdf.body_text(f"Estimated Ultimate Recovery (EUR): {total_eur:.0f} MSm3")
        pdf.body_text(f"Remaining Reserves: {total_remaining:.0f} MSm3")
    
    pdf.ln(5)
    pdf.chapter_title("5. Recommendations")
    
    recommendations = [
        "Continue daily surveillance of production rates and water cuts",
        "Address high-severity alerts within 48 hours",
        "Schedule interventions for top-priority wells",
        "Update decline curve analysis monthly for EUR tracking",
        "Review economic assumptions quarterly"
    ]
    
    for rec in recommendations:
        pdf.body_text(f"  - {rec}")
    
    output = io.BytesIO()
    pdf_bytes = pdf.output()
    output.write(pdf_bytes)
    return output.getvalue()
