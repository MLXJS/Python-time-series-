import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('Facebook Histor Stock Price.csv')

# Preprocess data
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.drop(['open', 'high', 'low', 'adj_close', 'volume'], axis=1)
df = df.resample('W').last().dropna()

# Train the ARIMA model
train = df[:'2020']
test = pd.date_range(start='2021-01-03', end='2022-12-25', freq='W')
model = ARIMA(train, order=(0, 10, 2))
model_fit = model.fit()
forecast = pd.DataFrame(model_fit.forecast(steps=len(test))[0], index=test, columns=['forecast'])

# Fill any NaN values in the forecast with linear interpolation
forecast = forecast.interpolate(method='linear', limit_direction='forward')

# Evaluate the model
mse = mean_squared_error(np.zeros(len(test)), forecast.values)
rmse = np.sqrt(mse)
print('RMSE: %.2f' % rmse)

# Visualize the results
plt.figure(figsize=(12, 8))
plt.plot(df.index, df.values, label='Actual')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.title('Facebook Stock Price Forecast')
plt.show()