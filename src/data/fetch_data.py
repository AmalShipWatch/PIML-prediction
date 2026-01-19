"""
Script to download or generate data and save to data/raw
"""

import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
import urllib.parse
import sqlalchemy
from typing import List, Dict, Any
import plotly.express as px
import plotly.io as pio
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML
import matplotlib.pyplot as plt


def generate_synthetic_data(imo_value, start_date, end_date):
    """Generate synthetic ship performance data when database is not available."""
    print("Generating synthetic noon data...")
    
    # Parse dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Generate date range (one entry per day)
    dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    n_samples = len(dates)
    
    # Generate synthetic data matching the expected schema
    np.random.seed(42)
    
    # Speed distribution (knots): normally distributed around 12-16 knots
    avg_speed = np.random.normal(14, 2, n_samples)
    avg_speed = np.clip(avg_speed, 8, 18)  # Reasonable ship speed range
    
    # Draft distribution (meters): separate laden/ballast
    is_laden = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    mean_draft = np.where(
        is_laden == 1,
        np.random.normal(11, 0.5, n_samples),  # Laden: ~11m
        np.random.normal(7, 0.5, n_samples)    # Ballast: ~7m
    )
    mean_draft = np.clip(mean_draft, 5, 13)
    
    # Power calculation based on physics (cubic relationship with adjustments)
    # P = k * V^3 * (1 + draft_factor)
    k_base = 50  # Base coefficient
    draft_factor = (mean_draft - 7) / 7  # Normalize draft influence
    power = k_base * (avg_speed ** 3) * (1 + 0.3 * draft_factor)
    
    # Add noise
    power += np.random.normal(0, 100, n_samples)
    power = np.clip(power, 1000, 8000)  # Reasonable power range
    
    # Load percentage
    load = (power / 7321.0) * 100  # Assuming MCR ~7321 kW
    load = np.clip(load, 20, 95)
    
    # Consumption (roughly proportional to power)
    consumption = power * 0.19 / 1000  # Approximate SFOC conversion
    consumption += np.random.normal(0, 0.5, n_samples)
    consumption = np.clip(consumption, 0.5, 3.0)
    
    # Environmental conditions
    bf_value = np.random.randint(0, 7, n_samples)
    dss_value = np.random.randint(0, 10, n_samples)
    wind_speed = bf_value * 5 + np.random.normal(0, 3, n_samples)
    wind_speed = np.clip(wind_speed, 0, 40)
    wave_height = bf_value * 0.5 + np.random.normal(0, 0.3, n_samples)
    wave_height = np.clip(wave_height, 0, 5)
    
    # Other fields
    wind_direction = np.random.uniform(0, 360, n_samples)
    wave_direction = np.random.uniform(0, 360, n_samples)
    course_direction = np.random.uniform(0, 360, n_samples)
    slip = np.random.uniform(0, 10, n_samples)
    rpm = avg_speed * 6 + np.random.normal(0, 5, n_samples)
    rpm = np.clip(rpm, 50, 120)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'steam_time': np.random.uniform(20, 24, n_samples),
        'avg_speed': avg_speed,
        'mean_draft': mean_draft,
        'consumption': consumption,
        'load': load,
        'power': power,
        'bf_value': bf_value,
        'dss_value': dss_value,
        'is_laden': is_laden,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'wave_height': wave_height,
        'wave_direction': wave_direction,
        'course_direction': course_direction,
        'consumption_vemon': consumption * 1.05,  # Slight variation
        'slip': slip,
        'rpm': rpm,
        'start_port': ['Port_A', 'Port_B', 'Port_C'][0],
        'end_port': ['Port_X', 'Port_Y', 'Port_Z'][0],
        'passage_id': range(1, n_samples + 1),
        'sea_passage_start_utc': dates - pd.Timedelta(days=1),
        'sea_passage_end_utc': dates,
        'instructed_speed_kts': avg_speed + np.random.normal(0, 0.5, n_samples),
        'observed_distance_nm': avg_speed * 24,
        'speed_over_ground_kts': avg_speed + np.random.normal(0, 0.3, n_samples),
        'speed_corr_current_weather_kts': avg_speed,
        'logged_speed_kts': avg_speed + np.random.normal(0, 0.2, n_samples),
        'reported_beaufort_force': bf_value,
        'reported_dss': dss_value,
        'draft_aft_metres': mean_draft + np.random.normal(0, 0.2, n_samples),
        'draft_fore_metres': mean_draft + np.random.normal(0, 0.2, n_samples),
        'rob_mt': np.random.uniform(500, 2000, n_samples)
    })
    
    print(f"Generated {len(df)} synthetic data points")
    return df




def collect_noon_data(imo_value, start_date, end_date):
    """Fetch noon data from database for a specific vessel and date range."""
    start_time = datetime.now()
    
    # Database connection
    ENGINE_URL = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=docdevsql02;"
    "DATABASE=ShipwatchStaging;"
    "Trusted_Connection=yes;"
    )
    query = f"""
        SELECT 
            c.TimestampUtc AS time, 
            c.SteamTimeHrs AS steam_time, 
            c.SpeedCorrCurrentKts AS avg_speed, 
            (c.DraftAftMetres + c.DraftForeMetres) / 2 AS mean_draft, 
            rt.MeConsMt AS consumption, 
            c.MeLoadPct AS load, 
            c.MePowerKw AS power,
            c.BeaufortForce as bf_value, 
            c.Dss as dss_value,
            c.IsLaden AS is_laden,
            c.WindSpeedKts AS wind_speed, 
            c.WindDirDeg AS wind_direction, 
            c.WaveHtMetres AS wave_height, 
            c.WaveDirDeg AS wave_direction, 
            c.CourseDeg AS course_direction, 
            c.CalculatedConsumptionMt AS consumption_vemon,
            c.SlipPct AS slip,
            c.AvgRpm AS rpm,
            c.ReportedPortRaw AS start_port,
            c.UpcomingPortRaw AS end_port,
            c.PassageId AS passage_id,
            c.SeaPassageStartUtc AS sea_passage_start_utc,
            c.SeaPassageEndUtc AS sea_passage_end_utc,
            c.InstructedSpeedKts AS instructed_speed_kts,
            c.ObservedDistanceNm AS observed_distance_nm,
            c.SpeedOverGroundKts AS speed_over_ground_kts,
            c.SpeedCorrCurrentWeatherKts AS speed_corr_current_weather_kts,
            c.LoggedSpeedKts AS logged_speed_kts,
            c.ReportedBeaufortForce AS reported_beaufort_force,
            c.ReportedDss AS reported_dss,
            c.DraftAftMetres AS draft_aft_metres,
            c.DraftForeMetres AS draft_fore_metres,
            rt.RobMt AS rob_mt
        FROM dbo.ConsVesselReports c 
        OUTER APPLY ( 
            SELECT SUM(MeConsMt * cf.CalorificValue / 40200.0) AS MeConsMt,
                   SUM(crf.RobMt) AS RobMt
            FROM dbo.ConsVesselReportFuels crf 
            INNER JOIN dbo.ConsFuelType cf ON crf.TypeId = cf.FuelTypeId 
            WHERE crf.ReportId = c.Id 
        ) rt 
        WHERE c.VesselImo = {imo_value} 
            AND c.TypeId IN (1, 2, 5) 
            AND c.TimestampUtc BETWEEN '{start_date}' AND '{end_date}' 
        ORDER BY c.TimestampUtc ASC;
    """
    
    try:
        engine = sqlalchemy.create_engine(ENGINE_URL)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        elapsed_time = datetime.now() - start_time
        print(f"Successfully fetched {len(df)} rows from database for noon data in {elapsed_time.total_seconds():.2f} seconds")
        return df
    except Exception as e:
        elapsed_time = datetime.now() - start_time
        print(f"Failed to fetch noon data from DB after {elapsed_time.total_seconds():.2f} seconds: {e}")
        print("Falling back to synthetic data generation...")
        return generate_synthetic_data(imo_value, start_date, end_date)
    
def get_vessel_info_local(imo_value: int) -> Dict[str, Any]:
    """Get vessel name and specifications from local database for given IMO.

    Args:
        imo_value: The IMO number to look up

    Returns:
        Dict containing vessel information with keys:
        - imo: The IMO number
        - name: Vessel name
        - mcr: Maximum Continuous Rating (kW)
        - laden_draft: Laden draft (m)
        - ballast_draft: Ballast draft (m)
        - customer_description: Customer description

    Raises:
        RuntimeError: If the local vessel_specifications table does not exist.
        LookupError: If no vessel record is found for the given IMO.
        Exception: Re-raises any unexpected database errors.
    """
    try:
        with sqlite3.connect(LOCAL_DB_PATH) as conn:
            cursor = conn.cursor()

            # Check if vessel_specifications table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='vessel_specifications'
            """
            )

            if not cursor.fetchone():
                (
                    "Vessel specifications table not found. Run update_vessel_specs.py first."
                )
                raise RuntimeError("Local vessel_specifications table not found.")

            # Fetch vessel information
            cursor.execute(
                """
                SELECT vessel_name, mcr, laden_draft, ballast_draft, customer_description
                FROM vessel_specifications 
                WHERE imo_value = ?
            """,
                (imo_value,),
            )

            result = cursor.fetchone()

        if result:
            return {
                "imo": imo_value,
                "name": result[0],
                "mcr": result[1],
                "laden_draft": result[2],
                "ballast_draft": result[3],
                "customer_description": result[4],
            }
        else:
            (f"No vessel information found for IMO {imo_value} in local database.")
            raise LookupError(
                f"No vessel information found for IMO {imo_value} in local database."
            )

    except Exception:
        (f"Error fetching vessel info for IMO {imo_value} from local database.")
        raise

# Fetch noon data
imo_value = 9935301
start_date = '2023-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
df_noon_data = collect_noon_data(imo_value, start_date, end_date)


def save_raw_data(data, filepath):
    """Save raw data to file."""
    data.to_csv(filepath, index=False)


if __name__ == '__main__':
    # Example usage
    print("Loading raw data...")
    # data = load_raw_data('data/raw/input.csv')
    # Save processed data
    # save_raw_data(data, 'data/raw/output.csv')
