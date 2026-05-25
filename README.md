# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 25-05-26

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
"""
AIM:
To implement the SARIMA (Seasonal AutoRegressive Integrated Moving Average)
model using Python on the Nifty 50 stock dataset (HDFCBANK.NS) to forecast
monthly average close prices, and evaluate model performance using RMSE and MAE.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from itertools import product
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATASET
# ═══════════════════════════════════════════════════════════════════════════════
STOCK    = 'HDFCBANK.NS'
CSV_PATH = 'nifty50_2000_2025.csv'   # keep CSV in same folder as notebook

df = pd.read_csv(CSV_PATH)
df['Date'] = pd.to_datetime(df['Date'])

stock_df = df[df['Stock'] == STOCK].sort_values('Date').copy()
stock_df['YearMonth'] = stock_df['Date'].dt.to_period('M')

monthly = stock_df.groupby('YearMonth')['Close'].mean()
monthly.index = monthly.index.to_timestamp()
monthly.index.freq = 'MS'

print("=" * 60)
print("STEP 1 — DATASET EXPLORATION")
print("=" * 60)
print(f"\nStock       : {STOCK}")
print(f"Shape       : {monthly.shape}")
print(f"Date range  : {monthly.index[0].date()} → {monthly.index[-1].date()}")
print(f"\nFirst 5 rows:\n{monthly.head().to_frame()}")
print(f"\nBasic Stats:\n{monthly.describe().round(2)}")

plt.figure(figsize=(14, 4))
plt.plot(monthly, color='steelblue', linewidth=1.5)
plt.title(f'{STOCK} — Monthly Avg Close Price (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECK STATIONARITY — ADF, ACF, PACF, DIFFERENCING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — STATIONARITY CHECK")
print("=" * 60)

def adf_test(series, label='Series'):
    result = adfuller(series.dropna())
    print(f"\nADF Test — {label}")
    print(f"  ADF Statistic  : {result[0]:.4f}")
    print(f"  p-value        : {result[1]:.4f}")
    for k, v in result[4].items():
        print(f"  Critical ({k}) : {v:.4f}")
    if result[1] > 0.05:
        print("  → NON-STATIONARY (p > 0.05)")
    else:
        print("  → STATIONARY (p < 0.05) ✓")

adf_test(monthly, 'Original Close Price')

# ACF & PACF — Original
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(monthly,  lags=40, ax=axes[0])
axes[0].set_title('ACF — Original (Slow decay confirms non-stationarity)')
axes[0].grid(alpha=0.3)
plot_pacf(monthly, lags=40, ax=axes[1])
axes[1].set_title('PACF — Original')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 1st order + seasonal differencing
monthly_diff   = monthly.diff().dropna()              # regular differencing (d=1)
monthly_s_diff = monthly_diff.diff(12).dropna()       # seasonal differencing (D=1, s=12)

adf_test(monthly_diff,   '1st Order Differenced')
adf_test(monthly_s_diff, '1st Order + Seasonal Differenced')

plt.figure(figsize=(14, 4))
plt.plot(monthly_s_diff, color='tomato', linewidth=1.2)
plt.title(f'{STOCK} — After Regular + Seasonal Differencing (d=1, D=1, s=12)')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ACF & PACF — After differencing (to read p, q, P, Q)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(monthly_s_diff,  lags=40, ax=axes[0])
axes[0].set_title('ACF — After Differencing  (use to find q & Q)')
axes[0].grid(alpha=0.3)
plot_pacf(monthly_s_diff, lags=40, ax=axes[1])
axes[1].set_title('PACF — After Differencing (use to find p & P)')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. DETERMINE SARIMA PARAMETERS p, d, q, P, D, Q, s
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — SARIMA PARAMETERS")
print("=" * 60)
print("""
  SARIMA(p, d, q)(P, D, Q, s)
  ────────────────────────────────────────
  p  = AR order           (from PACF lags before cutoff)
  d  = differencing       → 1 (regular differencing for trend)
  q  = MA order           (from ACF  lags before cutoff)
  P  = Seasonal AR order  (from PACF at seasonal lags)
  D  = Seasonal diff      → 1 (seasonal differencing for yearly pattern)
  Q  = Seasonal MA order  (from ACF  at seasonal lags)
  s  = Seasonal period    → 12 (monthly data, yearly seasonality)

  Manually selected : SARIMA(1,1,1)(1,1,1,12)
  Auto-selected     : determined by AIC grid search below
""")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIT SARIMA(1,1,1)(1,1,1,12) — Manual
# ═══════════════════════════════════════════════════════════════════════════════
train = monthly[:-12]
test  = monthly[-12:]

print("=" * 60)
print("STEP 4 — FIT SARIMA(1,1,1)(1,1,1,12)  [Manual]")
print("=" * 60)
print(f"\nTrain : {len(train)} months  ({train.index[0].date()} → {train.index[-1].date()})")
print(f"Test  : {len(test)} months   ({test.index[0].date()} → {test.index[-1].date()}\n)")

sarima_manual = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

print(sarima_manual.summary())

# ═══════════════════════════════════════════════════════════════════════════════
# 5. AUTO-FIT SARIMA — AIC Grid Search (no pmdarima needed)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — AUTO-FIT SARIMA (AIC Grid Search)")
print("=" * 60)

p_values = [0, 1, 2]
d_values = [1]
q_values = [0, 1, 2]
P_values = [0, 1]
D_values = [1]
Q_values = [0, 1]
s        = 12

best_aic      = np.inf
best_order    = None
best_seasonal = None
aic_table     = []

print("Searching best order...")
for p, d, q, P, D, Q in product(p_values, d_values, q_values, P_values, D_values, Q_values):
    try:
        model = SARIMAX(
            train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        aic_table.append({'order': (p,d,q), 'seasonal': (P,D,Q,s), 'AIC': round(model.aic, 2)})
        if model.aic < best_aic:
            best_aic      = model.aic
            best_order    = (p, d, q)
            best_seasonal = (P, D, Q, s)
    except:
        continue

aic_df = pd.DataFrame(aic_table).sort_values('AIC').head(5)
print("\nTop 5 Models by AIC:")
print(aic_df.to_string(index=False))
print(f"\n✓ Best SARIMA Order    : {best_order}")
print(f"✓ Best Seasonal Order  : {best_seasonal}")
print(f"✓ Best AIC             : {best_aic:.2f}")

# Fit best auto model
sarima_auto = SARIMAX(
    train,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS — Manual & Auto SARIMA
# ═══════════════════════════════════════════════════════════════════════════════
manual_pred = sarima_manual.forecast(steps=12)
manual_pred.index = test.index

auto_pred = sarima_auto.forecast(steps=12)
auto_pred.index = test.index

# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATE MODEL PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — MODEL EVALUATION")
print("=" * 60)

for name, pred in [('SARIMA(1,1,1)(1,1,1,12) Manual', manual_pred),
                   (f'SARIMA{best_order}{best_seasonal} Auto',  auto_pred)]:
    rmse = np.sqrt(mean_squared_error(test, pred))
    mae  = mean_absolute_error(test, pred)
    print(f"\n  {name}")
    print(f"    RMSE : {rmse:.2f}")
    print(f"    MAE  : {mae:.2f}")

# Prediction plot
plt.figure(figsize=(14, 6))
plt.plot(train.iloc[-36:], color='steelblue', linewidth=1.5,               label='Train (last 36 months)')
plt.plot(test,             color='black',     linewidth=2,                  label='Actual Test')
plt.plot(manual_pred,      color='tomato',    linewidth=1.8, linestyle='--', label='SARIMA Manual (1,1,1)(1,1,1,12)')
plt.plot(auto_pred,        color='green',     linewidth=1.8, linestyle=':',  label=f'SARIMA Auto {best_order}{best_seasonal}')
plt.title(f'{STOCK} — SARIMA Test Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Residual diagnostics of best model
fig = sarima_auto.plot_diagnostics(figsize=(14, 10))
plt.suptitle(f'SARIMA Auto {best_order}{best_seasonal} — Residual Diagnostics', fontsize=12)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL PREDICTION — Fit on full data, forecast 12 months ahead
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL PREDICTION — Next 12 Months (2025)")
print("=" * 60)

final_model = SARIMAX(
    monthly,
    order=best_order,
    seasonal_order=best_seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

future_pred  = final_model.forecast(steps=12)
future_index = pd.date_range(start=monthly.index[-1], periods=13, freq='MS')[1:]
future_series = pd.Series(future_pred.values, index=future_index)

print(future_series.round(2).to_frame(name='Predicted Close (₹)'))

plt.figure(figsize=(14, 5))
plt.plot(monthly,       color='steelblue', linewidth=1.5,               label='Historical Data')
plt.plot(future_series, color='tomato',    linewidth=2,   linestyle='--', label=f'Forecast — SARIMA{best_order}{best_seasonal}')
plt.axvline(x=monthly.index[-1], color='gray', linestyle=':', linewidth=1, label='Forecast Start')
plt.title(f'{STOCK} — SARIMA Final Forecast (Jan–Dec 2025)')
plt.xlabel('Date')
plt.ylabel('Close Price (₹)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nRESULT:")
print("Thus the program ran successfully based on the SARIMA model.")
```

### OUTPUT:
```
============================================================
STEP 1 — DATASET EXPLORATION
============================================================

Stock       : HDFCBANK.NS
Shape       : (299,)
Date range  : 2000-02-01 → 2024-12-01

First 5 rows:
                Close
YearMonth            
2000-02-01  11.551923
2000-03-01  13.150000
2000-04-01  11.367125
2000-05-01  12.426304
2000-06-01  12.466023

Basic Stats:
count    299.00
mean     270.91
std      279.45
min        9.72
25%       37.88
50%      145.32
75%      500.53
max      913.28
Name: Close, dtype: float64
```
<img width="1117" height="327" alt="image" src="https://github.com/user-attachments/assets/7f2f3f1d-1835-4dae-ac56-32e71076d871" />

```
============================================================
STEP 2 — STATIONARITY CHECK
============================================================

ADF Test — Original Close Price
  ADF Statistic  : 2.4411
  p-value        : 0.9990
  Critical (1%) : -3.4537
  Critical (5%) : -2.8718
  Critical (10%) : -2.5722
  → NON-STATIONARY (p > 0.05)
```

<img width="1108" height="642" alt="image" src="https://github.com/user-attachments/assets/a55ba902-bbbf-4b26-afb8-6a7dcc7ade13" />

```
ADF Test — 1st Order Differenced
  ADF Statistic  : -6.0084
  p-value        : 0.0000
  Critical (1%) : -3.4537
  Critical (5%) : -2.8718
  Critical (10%) : -2.5722
  → STATIONARY (p < 0.05) ✓

ADF Test — 1st Order + Seasonal Differenced
  ADF Statistic  : -7.3855
  p-value        : 0.0000
  Critical (1%) : -3.4548
  Critical (5%) : -2.8723
  Critical (10%) : -2.5725
  → STATIONARY (p < 0.05) ✓
```

<img width="1125" height="956" alt="image" src="https://github.com/user-attachments/assets/4c800bd7-6b79-455a-a2c0-d3ca12c6725d" />

```
============================================================
STEP 3 — SARIMA PARAMETERS
============================================================

  SARIMA(p, d, q)(P, D, Q, s)
  ────────────────────────────────────────
  p  = AR order           (from PACF lags before cutoff)
  d  = differencing       → 1 (regular differencing for trend)
  q  = MA order           (from ACF  lags before cutoff)
  P  = Seasonal AR order  (from PACF at seasonal lags)
  D  = Seasonal diff      → 1 (seasonal differencing for yearly pattern)
  Q  = Seasonal MA order  (from ACF  at seasonal lags)
  s  = Seasonal period    → 12 (monthly data, yearly seasonality)

  Manually selected : SARIMA(1,1,1)(1,1,1,12)
  Auto-selected     : determined by AIC grid search below

============================================================
STEP 4 — FIT SARIMA(1,1,1)(1,1,1,12)  [Manual]
============================================================

Train : 287 months  (2000-02-01 → 2023-12-01)
Test  : 12 months   (2024-01-01 → 2024-12-01
)
                                     SARIMAX Results                                      
==========================================================================================
Dep. Variable:                              Close   No. Observations:                  287
Model:             SARIMAX(1, 1, 1)x(1, 1, 1, 12)   Log Likelihood               -1114.813
Date:                            Mon, 25 May 2026   AIC                           2239.625
Time:                                    10:59:21   BIC                           2257.429
Sample:                                02-01-2000   HQIC                          2246.783
                                     - 12-01-2023                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.3810      0.106     -3.607      0.000      -0.588      -0.174
ma.L1          0.7251      0.067     10.904      0.000       0.595       0.855
ar.S.L12      -0.0018      0.057     -0.031      0.975      -0.113       0.110
ma.S.L12      -0.8616      0.052    -16.491      0.000      -0.964      -0.759
sigma2       299.2777     11.294     26.500      0.000     277.142     321.413
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):              1645.73
Prob(Q):                              0.92   Prob(JB):                         0.00
Heteroskedasticity (H):              67.59   Skew:                            -0.88
Prob(H) (two-sided):                  0.00   Kurtosis:                        15.20
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).

============================================================
STEP 5 — AUTO-FIT SARIMA (AIC Grid Search)
============================================================
Searching best order...

Top 5 Models by AIC:
    order      seasonal     AIC
(2, 1, 2) (0, 1, 1, 12) 2229.89
(0, 1, 2) (0, 1, 1, 12) 2230.93
(2, 1, 2) (1, 1, 1, 12) 2231.98
(1, 1, 2) (0, 1, 1, 12) 2232.07
(0, 1, 2) (1, 1, 1, 12) 2233.80

✓ Best SARIMA Order    : (2, 1, 2)
✓ Best Seasonal Order  : (0, 1, 1, 12)
✓ Best AIC             : 2229.89

============================================================
STEP 6 — MODEL EVALUATION
============================================================

  SARIMA(1,1,1)(1,1,1,12) Manual
    RMSE : 67.61
    MAE  : 53.90

  SARIMA(2, 1, 2)(0, 1, 1, 12) Auto
    RMSE : 69.17
    MAE  : 55.12
```

<img width="1104" height="478" alt="image" src="https://github.com/user-attachments/assets/9d92ee5a-918a-4457-ac02-14ed5f3d4a02" />
<img width="1118" height="810" alt="image" src="https://github.com/user-attachments/assets/e1ef457e-6fa4-4897-86ab-5c41878efb3a" />


```
============================================================
FINAL PREDICTION — Next 12 Months (2025)
============================================================
            Predicted Close (₹)
2025-01-01               912.94
2025-02-01               898.48
2025-03-01               880.31
2025-04-01               887.56
2025-05-01               879.68
2025-06-01               898.46
2025-07-01               909.50
2025-08-01               906.25
2025-09-01               914.31
2025-10-01               916.87
2025-11-01               932.52
2025-12-01               952.62
```

<img width="1121" height="400" alt="image" src="https://github.com/user-attachments/assets/dcfd62df-0087-4a25-8196-ceb303076be1" />




### RESULT:
Thus the program run successfully based on the SARIMA model.
