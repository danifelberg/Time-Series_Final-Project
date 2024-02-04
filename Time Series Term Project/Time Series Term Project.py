#%% ------------- Importing Packages -------------

print("------------- Importing Packages -------------")

#%% Importing Packages

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.tsa.holtwinters as ets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
import numpy.linalg as linalg
import random
import levenberg_marquardt as LM
from sktime.forecasting.arima import AutoARIMA
import pmdarima as pm
from pmdarima.arima.utils import ndiffs, nsdiffs
from sktime.forecasting.sarimax import SARIMAX
from scipy.stats import chi2

#%% ------------- Data Preprocessing -------------

print("------------- Data Preprocessing -------------")

#%% Importing Dataset, Printing head, # of obs&feats, list of num feats, and NaNs

df_raw = pd.read_csv("powerconsumption.csv", parse_dates=["Datetime"])
print("First 5 rows:")
print(df_raw.head())

print(f'Number of observations: {len(df_raw)}')

all_features = df_raw.describe(include='all').columns.tolist()
print(f'Number of features: {len(all_features)}')

numerical_features = df_raw.describe(include=[np.number]).columns.tolist()
print(f'Number of numerical features: {len(numerical_features)}')
print("Numerical Features:")
for i in numerical_features:
    print(i)

print("Last 5 rows:")
print(df_raw.tail())

print("NA values in each column:")
nan_values = df_raw.isna().sum()
print(nan_values)
print("No NaN values found.")

#%% Checking dataset is equally sampled (every 10 mins)
lst_ = []
for i in range(len(df_raw)-1):
    lst_.append(df_raw.Datetime.iloc[i+1] - df_raw.Datetime.iloc[i])

lst_ = np.array(lst_)

print(f"Value {np.unique(lst_)} means data was in fact sampled every 10 minutes, with no outliers")

#%% Creating 'df' object & Indexing 'Datetime' column

df = df_raw.copy()
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

print("First 5 rows of 'df':")
print(df.head())

#%% Plotting dependent variables vs. Time (before down sampling)

Zone1 = df['PowerConsumption_Zone1']
Zone2 = df['PowerConsumption_Zone2']
Zone3 = df['PowerConsumption_Zone3']

plt.figure(figsize=(14, 6))
plt.plot(Zone1, label='Zone 1')
plt.plot(Zone2, label='Zone 2')
plt.plot(Zone3, label='Zone 3')
plt.legend(loc='upper left')
plt.title(f' Power Consumption vs. Time by Zone ({len(df)} observations)')
plt.xlabel('Datetime')
plt.ylabel('Power Consumption (KW)')
plt.grid()
plt.tight_layout()
plt.show()

print(f'Number of observations before down sampling: {len(df)}')

#%% Removing Unnecessary Target Variables (Zones 1 & 2) and Renaming Zone 3

print("Cannot sum all 3 dependent variables due to real-life applications. We will only keep 'Zone 3' for our analysis.")
df = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2'], axis=1)
df.rename(columns={'PowerConsumption_Zone3': 'PowerConsumption'}, inplace=True)

print(df.head())

#%% Performing Down Sampling ('df_hourly') to Improve Computational Efficiency

df_hourly = df.copy()
df_hourly = df_hourly.resample('1H').mean()
print("Hourly Data (First 5 rows):")
print(df_hourly.head())
print(f'Number of observations after down sampling: {len(df_hourly)}')

#%% Plotting 'PowerConsumption' vs. Time (after down sampling)

xlabel_hours = [len(df_hourly),744,168,24]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
fig.suptitle(f'Zone 3 Power Consumption vs. Time ({len(df_hourly)} observations)')

for i, hours in enumerate(xlabel_hours):
    row = i // 2
    col = i % 2

    axes[row, col].plot(df_hourly['PowerConsumption'][:hours])
    axes[row, col].set_title(f'First {hours} Hours')
    axes[row, col].set_xlabel("Datetime")
    axes[row, col].tick_params(axis='x', rotation=45)
    axes[row, col].set_ylabel("Power Consumption (KW)")
    axes[row, col].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

#%% ACF of 'PowerConsumption'

def Cal_autocorrelation(data, max_lag, title):
    def autocorrelation(data, max_lag):
        n = len(data)
        mean = np.mean(data)
        numerator = sum((data[i] - mean) * (data[i - max_lag] - mean) for i in range(max_lag, n))
        denominator = sum((data[i] - mean) ** 2 for i in range(n))
        ry = numerator / denominator if denominator != 0 else 1.0
        return ry

    acf_values = [autocorrelation(data, lag) for lag in range(max_lag + 1)]

# Plot the ACF
    a = acf_values
    b = a[::-1]
    c =  b + a[1:]
    plt.figure()
    x_values = range(-max_lag, max_lag + 1)
    (markers, stemlines, baseline) = plt.stem(x_values, c, markerfmt = 'o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(len(data))
    plt.axhspan(-m,m, alpha = 0.2, color = 'blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.show()



Cal_autocorrelation(df_hourly['PowerConsumption'], 75, "ACF of PowerConsumption")
Cal_autocorrelation(df_hourly['PowerConsumption'], 25, "ACF of PowerConsumption")
print("Magnitude peaks every 24 lags (hours). Plot appears to show the data is colored.")

#%% Correlation Matrix

corr = df_hourly.corr()
ax = sns.heatmap(corr, annot=True)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.title("Correlation Matrix with Pearson's Correlation Coefficient")
plt.show()

#%% Inital train/test split (non-stationary)

ind_vars = ['Temperature', 'Humidity', 'WindSpeed', 'DiffuseFlows', 'GeneralDiffuseFlows']
dep_var = ['PowerConsumption']

y_hourly = df_hourly[dep_var]
x_hourly = df_hourly[ind_vars]

yt_hourly, yf_hourly = train_test_split(df_hourly[dep_var], shuffle= False, test_size=0.2)
xt_hourly, xf_hourly = train_test_split(df_hourly[ind_vars], shuffle= False, test_size=0.2)

#%% ------------- Checking Stationarity -------------

print("------------- Checking Stationarity -------------")

#%% ACF/PACF Analysis

def ACF_PACF_Plot(y,lags,suptitle):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.subplot(211)
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.subplots_adjust(top=0.85)
    plt.show()

ACF_PACF_Plot(y_hourly,lags=75,suptitle='ACF/PACF of PowerConsumption')
print("Magnitude again peaks approximately every 24 lags. Graphs appear to show the data is colored.")

#%% ADF/KPSS tests
def ADF_Cal(x):
    x = x.interpolate()
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    timeseries = timeseries.interpolate()
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

ADF_Cal(y_hourly)
kpss_test(y_hourly)
print("Both tests suggest data is not stationary. Differencing is needed.")

#%% Plotting Rolling Mean and Rolling Variance

rol_mean_var_len = len(df_hourly)
def Cal_rolling_mean_var(value, y):
    fig, axs = plt.subplots(2, 1)

    rolling_means = []
    rolling_variances = []

    for i in range(1, rol_mean_var_len + 1):
        rolling_mean = value.head(i).mean()
        rolling_variance = value.head(i).var()

        rolling_means.append(rolling_mean)
        rolling_variances.append(rolling_variance)

    axs[0].plot(range(1, rol_mean_var_len + 1), rolling_means, label='Rolling Mean')
    axs[0].set_title(f'Rolling Mean - {y}')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Magnitude')
    axs[0].legend()

    axs[1].plot(range(1, rol_mean_var_len + 1), rolling_variances, label='Rolling Variance')
    axs[1].set_title(f'Rolling Variance - {y}')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

Cal_rolling_mean_var(y_hourly, "PowerConsumption")
print("Rolling mean and rolling variance confirm the data is not stationary.")

#%% Seasonal Differencing

df_stat = df_hourly.copy() # stationary df
df_stat = df_stat.drop(['PowerConsumption'], axis=1)

# seasonal differencing
df_stat['season_diff'] = df_hourly['PowerConsumption'].diff(periods=24)
ACF_PACF_Plot(df_stat['season_diff'].dropna(), 50, suptitle='ACF/PACF After Seasonal Differencing (Periods=24)')
Cal_rolling_mean_var(df_stat['season_diff'].dropna(), 'Seasonal Differencing (Periods=24)')

print("Rolling variance suggests non-seasonal first order differencing is needed")

#%% First Order Differencing

def differencing(data):
    diff_data = []
    for i in range(len(data)):
        if i == 0:
            diff_data.append(np.nan)
        else:
            diff_data.append((data[i]) - data[i-1])
    return diff_data

# first order diff
df_stat['1st_order&season_diff'] = differencing(df_stat['season_diff'])
ACF_PACF_Plot(df_stat['1st_order&season_diff'].dropna(), 50, suptitle='ACF/PACF After Seasonal & 1st Order Diff ACF')
Cal_rolling_mean_var(df_stat['1st_order&season_diff'].dropna(), 'Seasonal & 1st Order Differencing')
ADF_Cal(df_stat['1st_order&season_diff'].dropna())
kpss_test(df_stat['1st_order&season_diff'].dropna())

print("Data now appears to be stationary.")

# renaming, removing columns & removing NAs
df_stat = df_stat.drop(['season_diff'], axis=1)
df_stat.rename(columns={'1st_order&season_diff': 'PowerConsumption'}, inplace=True)
df_stat = df_stat.dropna()



#%% Additional train/test split (stationary)

y_stat = df_stat[dep_var]
x_stat = df_stat[ind_vars]

yt_stat, yf_stat = train_test_split(df_stat[dep_var], shuffle= False, test_size=0.2)
xt_stat, xf_stat = train_test_split(df_stat[ind_vars], shuffle= False, test_size=0.2)

#%% ------------- Time Series Decomposition -------------

print("------------- Time Series Decomposition -------------")

#%% Plotting Trend, Seasonality, and Residuals

from statsmodels.tsa.seasonal import STL

# STL Decomposition (stationary)
stl = STL(y_stat, period=24) #ACF peaks every 24 lags
res = stl.fit()
plt.figure(figsize=(10, 8))
fig = res.plot()
for ax in fig.get_axes():
    ax.tick_params(axis='x', labelsize=6.5) # Resize x-axis labels for all subplots
plt.tight_layout()
plt.show()

#%% Calculating Trend and Seasonality Strength

T = res.trend
S = res.seasonal
R = res.resid

def str_trend_seasonal(T, S, R):
    F = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(T + R))))
    print(f'Trend strength is {100 * F: .3f}%')  # returns percentage
    FS = np.maximum(0, 1 - (np.var(np.array(R)) / np.var(np.array(S + R))))
    print(f'Seasonality strength is {100 * FS: .3f}%')  # returns percentage

str_trend_seasonal(T, S, R)

print("STL shows stationary data is weakly trended, weakly seasonal.")

#%% ------------- Holt-Winters Method -------------

print("------------- Holt-Winters Method -------------")

#%% HW Model 1: trend='mul', seasonal='add', damped_trend=True (MSE = 24,174,088.983)
holtt = ets.ExponentialSmoothing(yt_hourly,trend='mul', seasonal='add', damped_trend=True, seasonal_periods=24).fit(optimized=True)
holtf = holtt.forecast(steps=len(yf_hourly))
holtf = pd.DataFrame(holtf).set_index(yf_hourly.index)
MSE1 = np.square(np.subtract(yf_hourly.values,np.ndarray.flatten(holtf.values))).mean()

#%% HW Model 2: trend='add', seasonal='add', damped_trend=True (MSE = 24,168,078.507)

holtt = ets.ExponentialSmoothing(yt_hourly,trend='add', seasonal='add', damped_trend=True, seasonal_periods=24).fit(optimized=True)
holtf = holtt.forecast(steps=len(yf_hourly))
holtf = pd.DataFrame(holtf).set_index(yf_hourly.index)
MSE2 = np.square(np.subtract(yf_hourly.values,np.ndarray.flatten(holtf.values))).mean()

#%% HW Model 3: trend='mul', seasonal='mul', damped_trend=True (MSE = 21,220,610.392)

holtt = ets.ExponentialSmoothing(yt_hourly,trend='mul', seasonal='mul', damped_trend=True, seasonal_periods=24).fit(optimized=True)
holtf = holtt.forecast(steps=len(yf_hourly))
holtf = pd.DataFrame(holtf).set_index(yf_hourly.index)
MSE3 = np.square(np.subtract(yf_hourly.values,np.ndarray.flatten(holtf.values))).mean()
print(f"Model 1 MSE for Holt-Winters: {MSE1:.3f}")
print(f"Model 2 MSE for Holt-Winters: {MSE2:.3f}")
print(f"Model 3 MSE for Holt-Winters: {MSE3:.3f}")

#%% Plotting model with best MSE score (model 3):

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yt_hourly,label= "Training Data")
ax.plot(yt_hourly,label= "Test Data")
ax.plot(holtf,label= "Prediction")
plt.legend(loc='upper left')
plt.title(f'Holt-Winter Method (MSE: {round(MSE3,2)})')
plt.xlabel('Time')
plt.ylabel('PowerConsumption')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yf_hourly,label= "Test Data")
ax.plot(holtf,label= "Prediction")
plt.legend(loc='upper left')
plt.title(f'Holt-Winter Method (MSE: {round(MSE3,2)})')
plt.xlabel('Time')
plt.ylabel('PowerConsumption')
plt.tight_layout()
plt.show()

#%% ------------- Feature selection/dimensionality reduction -------------

print("------------- Feature selection/dimensionality reduction -------------")

#%% SVD Analysis

U, S, V = np.linalg.svd(x_hourly)
print(f'Singular Values', S)
print("No singular values close to zero; indicates potentially low multi-collinearity")

#%% Condition number

print(f'Condition number of x is = {np.linalg.cond(x_hourly):.3f}')
print("Condition number indicates there may be a moderate Degree of Collinearity")

#%% Standardizing Dataset (creating 'df_std')

scaler = StandardScaler()
array_std = scaler.fit_transform(df_hourly)
df_std = pd.DataFrame(array_std, columns=['Temperature', 'Humidity', 'WindSpeed',
                                       'DiffuseFlows', 'GeneralDiffuseFlows', 'PowerConsumption'])

print("Standardized Dataset:")
print(df_std.head())
#%% PCA (created using ChatGPT)

pca = PCA()
pca_result = pca.fit_transform(array_std)

explained_var_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_var_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_var_ratio) + 1), cumulative_var_ratio, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio')
plt.grid(True)
plt.show()

#%% Backward Stepwise Regression (created using ChatGPT)

X_std = df_std[ind_vars]
y_std = df_std[dep_var]

def backward_stepwise_regression(X, y, significance_level=0.05):
    selected_features = list(X.columns)
    features_removed = []

    while True:
        X_with_constant = sm.add_constant(X[selected_features])
        model = sm.OLS(y, X_with_constant).fit()
        p_values = model.pvalues.iloc[1:]

        # Find the feature with the highest p-value
        max_p_value = p_values.max()
        feature_to_remove = p_values.idxmax()

        if max_p_value > significance_level:
            # Remove the feature with the highest p-value
            selected_features.remove(feature_to_remove)
            features_removed.append(feature_to_remove)
        else:
            break

    return selected_features, features_removed

selected_features, features_removed = backward_stepwise_regression(X_std, y_std)

print("Selected Features:", selected_features)
print("Features Removed:", features_removed)

print("No features were removed after performing backward stepwise regression.")

#%% VIF (created using ChatGPT)

def calculate_vif(data_frame, variables):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables
    vif_data["VIF"] = [variance_inflation_factor(data_frame[variables].values, i) for i in range(len(variables))]
    return vif_data

high_vif_predictors = list(X_std.columns)
vif_threshold = 5

while len(high_vif_predictors) > 0:
    # Calculate VIF for the current set of predictors
    vif_data = calculate_vif(X_std, high_vif_predictors)

    # Find the predictor with the highest VIF
    max_vif_index = vif_data['VIF'].idxmax()
    max_vif_predictor = vif_data.loc[max_vif_index, 'Variable']

    # Check if the predictor with the highest VIF exceeds the threshold
    if vif_data.loc[max_vif_index, 'VIF'] > vif_threshold:
        # Remove the predictor with high VIF from the list of high VIF predictors and from the model
        high_vif_predictors.remove(max_vif_predictor)
        X = X.drop(max_vif_predictor, axis=1)
    else:
        # If no predictors exceed the threshold, exit the loop
        break

vif_model = sm.OLS(y_std, sm.add_constant(X_std)).fit()
print("VIF Method:")
print(f'Features kept:')
print(vif_model.params)

print("No features were removed after VIF analysis.")

#%% ------------- Base Models -------------

print("------------- Base Models -------------")

#%% Simple Average (Used code from "https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/#h-method-2-simple-average")

y_hat_avg = yf_hourly.copy()
y_hat_avg['avg_forecast'] = yt_hourly['PowerConsumption'].mean()
plt.figure(figsize=(12,6))
plt.plot(yt_hourly, label='Training Data')
plt.plot(yf_hourly, label='Test Data')
plt.plot(y_hat_avg['avg_forecast'], label='Prediction')
plt.legend(loc='upper left')
plt.title('Train, Test and Predicted Values (Simple Average Forecast)')
plt.xlabel('Datetime')
plt.ylabel('PowerConsumption')
plt.grid()
plt.tight_layout()
plt.show()

MSE_mean = mean_squared_error(yf_hourly.PowerConsumption, y_hat_avg.avg_forecast)
RMSE_mean = sqrt(mean_squared_error(yf_hourly.PowerConsumption, y_hat_avg.avg_forecast))

#%% Naive (Used code from "https://www.analyticsvidhya.com/blog/2018/02/time-series-forecasting-methods/#h-method-1-naive-forecast-python")

dd = np.asarray(yt_hourly.PowerConsumption)
y_hat = yf_hourly.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(12,6))
plt.plot(yt_hourly.index, yt_hourly['PowerConsumption'], label='Train')
plt.plot(yf_hourly.index,yf_hourly['PowerConsumption'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Prediction')
plt.legend(loc='upper left')
plt.title('Train, Test and Predicted Values (Naive Forecast)')
plt.xlabel('Datetime')
plt.ylabel('PowerConsumption')
plt.grid()
plt.tight_layout()
plt.show()

MSE_naive = mean_squared_error(yf_hourly.PowerConsumption, y_hat.naive)
RMSE_naive = sqrt(mean_squared_error(yf_hourly.PowerConsumption, y_hat.naive))

#%% SES

h = len(yf_hourly)
alphas = [0, 0.2, 0.4, 0.6, 0.8, 0.99]

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
fig.suptitle("SES Method Forecast at Different Alpha Values")

for i, alpha in enumerate(alphas):
    model = SimpleExpSmoothing(yt_hourly)
    model = model.fit(smoothing_level=alpha)
    forecast = model.forecast(h)
    ax = axes[i // 2, i % 2]
    ax.plot(yt_hourly, label="Training Data")
    ax.plot(yf_hourly, label="Test Data")
    ax.plot(forecast, label=f"Prediction")
    ax.set_title(f"Alpha = {alpha}")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("PowerConsumption")
    ax.legend(loc='upper left')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

#MSE/RMSE
print(f'Simple Average MSE: {MSE_mean}')
print(f'Simple Average RMSE: {RMSE_mean}')
print(f'Naive MSE: {MSE_naive}')
print(f'Naive RMSE: {RMSE_naive}')
print("SES Model Evaluations:")
for alpha in alphas:
    SESmodel = SimpleExpSmoothing(yt_hourly)
    SESmodel = SESmodel.fit(smoothing_level=alpha)
    SESforecast = SESmodel.forecast(h)
    SESerrors = abs(SESforecast - yf_hourly['PowerConsumption'])
    SES_MSE = (SESerrors**2).mean()
    SES_RMSE = sqrt(SES_MSE)
    print(f'MSE at alpha {alpha}: {SES_MSE:3f}')
    print(f'RMSE at alpha {alpha}: {SES_RMSE:3f}')

print("SES at alpha = 0.6 resulted in lowest MSE and RMSE. "
      "MSE = 10,671,852.516823; RMSE = 3,266.780145)")

#%% ------------- Multiple Linear Regression -------------

print("------------- Multiple Linear Regression -------------")

#%% OLS (reindexing standardized values, additional train/test split)

print("Using standardized data as it resulted in lower AIC and BIC values.")

X_std_idx = X_std.set_index(df_hourly.index) # reindexing features
y_std_idx = y_std.set_index(df_hourly.index) # reindexing target

X_train, X_test, y_train, y_test = train_test_split(X_std_idx,y_std_idx,shuffle=False, test_size=0.2)
model = sm.OLS(y_train,X_train).fit()

print(model.summary())

# Forecast
predictions_1step = model.predict(X_train)
predictions_hstep = model.predict(X_test)

plt.figure(figsize=(14,6))
plt.plot(y_train, label='Training Data')
plt.plot(y_test, label='Test Data')
plt.plot(predictions_1step, label='1-step')
plt.plot(predictions_hstep, label='Prediction')
plt.title('1-Step Ahead & Prediction')
plt.xlabel('Datetime')
plt.ylabel('PowerConsumption (Standardized)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% Hypothesis Test Analysis

print("H0 (t-test): βi = 0")
print("HA (t-test): βi ̸= 0")
print("H0 (F-Test): The fit of the "
      "intercept-only model and the OLS Regression model are equal.")
print("HA (F-Test): The fit of the intercept-only model is significantly reduced compared "
      "to the OLS Regression model.")

#t-test
print(f'P>|t|:')
print(round(model.pvalues,3))

#F-Test
print(f'F-statistic: {round(model.fvalue,3)}')
print(f'Prob (F-statistic): {model.f_pvalue}')

print("Results suggest we can reject the null, and accept the alternative for both tests.")

#%% Displaying MSE, AIC, BIC, RMSE, R^2 and Adj R^2

print(f'MSE: {model.mse_model:3f}')
print(f'AIC: {model.aic:3f}')
print(f'BIC: {model.bic:3f}')
print(f'RMSE: {np.sqrt(model.mse_model):3f}')
print(f'R^2: {model.rsquared:3f}')
print(f'Adj R^2: {model.rsquared_adj:3f}')

print("Results show this is our best model so far (based on MSE and RMSE).")

#%% ACF of Residuals

Cal_autocorrelation(model.resid, 75, 'ACF of MLR Residuals')

#%% Variance and mean of the residuals

Cal_rolling_mean_var(model.resid, "MLR Residuals")

#%% Plot the train, test, and predicted values in one plot

plt.figure(figsize=(14,6))
plt.plot(y_train, label='Training Data')
plt.plot(y_test, label='Test Data')
plt.plot(predictions_hstep, label='Prediction')
plt.title('MLR Forecast')
plt.xlabel('Datetime')
plt.ylabel('PowerConsumption (Standardized)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%% ------------- ARMA/ARIMA/SARIMA/Multiplicative model -------------

print("------------- ARMA/ARIMA/SARIMA/Multiplicative model -------------")

#%% ARMA model order determination (GPAC)

def gpac_values(ry, j_val, k_val):
    den = np.array([ry[np.abs(j_val + k - i)] for k in range(k_val) for i in range(k_val)]).reshape(k_val, k_val)
    col = np.array([ry[j_val+i+1] for i in range(k_val)])
    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    return np.inf if np.linalg.det(den) == 0 else round(np.linalg.det(num)/np.linalg.det(den), 10)

def GPAC_table(ry, j_val, k_val):
    gpac_arr = np.full((j_val, k_val), np.nan)
    for k in range(1, k_val):
        for j in range(j_val):
            gpac_arr[j][k] = gpac_values(ry, j, k)
    gpac_arr = np.delete(gpac_arr, 0, axis=1)
    df = pd.DataFrame(gpac_arr, columns=list(range(1, k_val)), index=list(range(j_val)))

    plt.figure()
    sns.heatmap(df, annot=True, fmt='0.3f', linewidths=.5)
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.tight_layout()
    plt.show()
    print(df)

j=12
k=12


gpac_values(np.array(y_stat), j, k)
GPAC_table(np.array(y_stat), j, k)

# potential ARMA orders: (4,6),(10,6)

#%% ACF/PACF Plot (stationary & non-stationary data)

ACF_PACF_Plot(y_stat, 50, suptitle='Stationary ACF')

print("AR Order: 0")
print("MA Order: 0")

#%% Initial SARIMA

forecaster = AutoARIMA(start_p=0, # AR Non-seasonal
                      d=1, # order of non-seasonal differencing
                      start_q=0, # MA Non-seasonal
                      max_p=1, # AR Non-seasonal
                      max_q=7, # MA Non-seasonal
                      start_P=0,  # AR seasonal
                      D=24, # order of seasonal differencing
                      start_Q=1, # MA seasonal
                      max_P=20, # AR seasonal
                      max_Q=20, # MA seasonal
                      seasonal=True, # Seasonality (true=yes)
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=False,
                      n_fits=20
)
forecaster = forecaster.fit(yt_hourly)
print(forecaster.summary())

#%% SARIMA 2

na = 0
nb = 0

model = sm.tsa.arima.ARIMA(yt_hourly,order=(na,1,nb), trend='n', seasonal_order=(0,1,1,24)).fit()
print(model.summary())
#===========
# Prediction
model_hat = model.predict(start=yf_hourly.index[0], end=yf_hourly.index[-1])
#======================================
lags = 50
# Residuals Testing and Chi-Square test
e = yf_hourly['PowerConsumption'] - model_hat

def autocorrelation(data, max_lag):
    n = len(data)
    mean = np.mean(data)
    numerator = sum((data[i] - mean) * (data[i - max_lag] - mean) for i in range(max_lag, n))
    denominator = sum((data[i] - mean) ** 2 for i in range(n))
    ry = numerator / denominator if denominator != 0 else 1.0
    return ry

ry = [autocorrelation(e, lags) for lags in range(lags + 1)]
Cal_autocorrelation(e, lags, 'ACF of Residuals')

Q = len(yf_hourly)*np.sum(np.square(ry[lags:]))
DOF = lags - 0 - 0
alfa = 0.01

chi_critical = chi2.ppf(1-alfa, DOF)

if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
print(sm.stats.acorr_ljungbox(e, lags=[lags]))

#%% GPAC 2

gpac_values(np.array(ry), j, k)
GPAC_table(np.array(ry), j, k)

# AR = 1, MA = 2
na = 1
nb = 2

model2 = sm.tsa.arima.ARIMA(yt_hourly,order=(na,1,nb), trend='n', seasonal_order=(0,1,1,24)).fit()
print(model2.summary())
print(model2.aic)

#%% SARIMA 3

# Prediction
model_hat2 = model2.predict(start=yf_hourly.index[0], end=yf_hourly.index[-1])
#======================================

# Residuals Testing and Chi-Square test
e2 = yf_hourly['PowerConsumption'] - model_hat2

ry2 = [autocorrelation(e2, lags) for lags in range(lags + 1)]
Cal_autocorrelation(e2, 100, 'ACF of Residuals')

Q2 = len(yf_hourly)*np.sum(np.square(ry2[lags:]))
DOF2 = lags - na - nb

chi_critical2 = chi2.ppf(1-alfa, DOF2)

if Q2< chi_critical2:
    print("The residual is white ")
else:
    print("The residual is NOT white ")

#%% GPAC 3

gpac_values(np.array(ry2), j, k)
GPAC_table(np.array(ry2), j, k)

# AR = 1, MA = 0
na = 1 + 1
nb = 0 + 2

model3 = sm.tsa.arima.ARIMA(yt_hourly,order=(na,1,nb), trend='n', seasonal_order=(0,1,1,24)).fit()
print(model3.summary())
print(model3.aic)

#%% SARIMA 4

# Prediction
model_hat3 = model3.predict(start=yf_hourly.index[0], end=yf_hourly.index[-1])
#======================================

# Residuals Testing and Chi-Square test
e3 = yf_hourly['PowerConsumption'] - model_hat3

ry3 = [autocorrelation(e3, lags) for lags in range(lags + 1)]
Cal_autocorrelation(e3, lags, 'ACF of Residuals')

Q3 = len(yf_hourly)*np.sum(np.square(ry3[lags:]))
DOF3 = lags - na - nb

chi_critical3 = chi2.ppf(1-alfa, DOF3)

if Q3< chi_critical3:
    print("The residual is white ")
else:
    print("The residual is NOT white ")

#%% Plotting Final Model

print(sm.stats.acorr_ljungbox(e3, lags=[lags]))
plt.figure()
plt.plot(y_hourly,'r', label = "True data")
plt.plot(model_hat3,'b', label = "Predicted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.legend()
plt.title("ststsmodels SARIMA parameter estimation and prediction")
plt.show()

#%% SARIMA 3?

# forecaster = AutoARIMA(start_p=1, # AR Non-seasonal
#                       d=1, # order of non-seasonal differencing
#                       start_q=2, # MA Non-seasonal
#                       max_p=1, # AR Non-seasonal
#                       max_q=7, # MA Non-seasonal
#                       start_P=0,  # AR seasonal
#                       D=1, # order of seasonal differencing
#                       start_Q=1, # MA seasonal
#                       sp=24,
#                       max_P=7, # AR seasonal
#                       max_Q=7, # MA seasonal
#                       seasonal=True, # Seasonality (true=yes)
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=False,
#                       n_fits=20
# )
# forecaster = forecaster.fit(yt_hourly)
# print(forecaster.summary())