import pandas as pd
import numpy as np
import pandas_datareader.data as web
from scipy.interpolate import CubicSpline
import datetime

def download_macro_data(start_date='2000-01-01', end_date='2024-01-01'):
    """
    Downloads GDP and Industrial Production data from FRED.
    Implements the 'Cubic Spline Interpolation' described in Section 3.2.
    """
    print("Downloading Macroeconomic Data from FRED...")
    
    # 1. Download Real GDP (Quarterly)
    # Series: GDPC1 (Real Gross Domestic Product)
    try:
        gdp = web.DataReader('GDPC1', 'fred', start_date, end_date)
    except Exception as e:
        print(f"Error downloading GDP data: {e}")
        return None, None
    
    # Calculate Logarithmic Growth Rate: ln(Y_t / Y_{t-1})
    # CORRECTION: We removed the '100 *' multiplier to match the S&P 500 scale.
    gdp['Growth'] = np.log(gdp['GDPC1'] / gdp['GDPC1'].shift(1))
    gdp = gdp.dropna()

    # 2. Cubic Spline Interpolation (Quarterly -> Daily)
    # As claimed in Section 3.2 of your paper
    gdp_daily = gdp['Growth'].resample('D').interpolate(method='cubic')
    
    # 3. Download Industrial Production (Monthly)
    # Series: INDPRO
    try:
        indpro = web.DataReader('INDPRO', 'fred', start_date, end_date)
        # CORRECTION: Removed '100 *' here as well
        indpro['Growth'] = np.log(indpro['INDPRO'] / indpro['INDPRO'].shift(1))
        
        # Align to daily for consistency
        indpro_daily = indpro['Growth'].resample('D').interpolate(method='linear')
        
        return gdp_daily.dropna(), indpro_daily.dropna()
        
    except Exception as e:
        print(f"Error downloading Industrial Production data: {e}")
        return gdp_daily.dropna(), None

if __name__ == "__main__":
    # Test the function
    gdp, ip = download_macro_data()
    if gdp is not None:
        print(f"Processed {len(gdp)} daily GDP points (interpolated).")
        print(f"Sample GDP value: {gdp.iloc[0]:.5f}") # Should be small (e.g., 0.005)