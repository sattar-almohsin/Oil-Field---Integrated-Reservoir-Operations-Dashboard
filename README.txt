Oil Field - Integrated Reservoir & Operations Dashboard

A professional Streamlit-based web application for reservoir surveillance and well operations analysis using the Volve field production dataset from the Norwegian Continental Shelf.

Purpose
This dashboard provides comprehensive production monitoring, decline curve analysis, intervention candidate ranking, and automated reporting capabilities.

Features

Overview: Field-level KPIs including total oil rate, active well count, and underperforming well identification
Reservoir & Field Surveillance: Individual well production analysis with oil/water rates, gas rates, water cut, and GOR trends
Well & Operations Performance: Performance comparisons with 6-month and 12-month rolling averages
Intervention Candidates: Weighted scoring system to prioritize wells for workover based on oil loss, water cut, and downtime
Decline Curve Analysis: Arps exponential and hyperbolic decline models with EUR predictions
Alerts & Monitoring: Configurable threshold-based alerts for rate drops, water cut spikes, and production anomalies
Reports & Export: PDF report generation and CSV data export
Unit Conventions

Oil and water rates: BPD (barrels per day)
Gas rates: Sm3/d (standard cubic meters per day)
Cumulative volumes: bbl (barrels)
Conversion factor: 1 Sm3 = 6.2898 barrels
Data Requirements
Excel file with two sheets: "Daily Production Data" and "Monthly Production Data" containing well bore names, dates, on-stream hours, and production volumes. Users can upload custom datasets via the sidebar.

Technologies
Python, Streamlit, Pandas, Plotly, NumPy, SciPy, FPDF2
__________________________________________________________

Author

Developed by Sattar Almohsin
Developer & Engineer — AI, Energy, and Data Systems
Basra, Iraq
