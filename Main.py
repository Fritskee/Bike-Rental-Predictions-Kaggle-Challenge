import numpy as np
import pandas as pd
import scipy as sp
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")
print("shape: of matrix: ", train_data.shape, "\n", train_data.head())
print(train_data.isna().sum()) #Check if we have any missing values


### Meaning of the values in our variables:
# season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
# mnth - month of the data point
# hr - hour of the data point
# holiday - binary, whether the day is considered a holiday (1-yes, 0-no)
# weekday - week of day of the data point
# workingday - binary, whether the day is neither a weekend nor holiday (1-yes, 0-no)
# weathersit - 1: Clear, Few clouds, Partly cloudy, Partly cloudy, 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog temp - temperature in Celsius, normalized
# atemp - "feels like" temperature in Celsius, normalized
# hum - relative humidity, normalized
# windspeed - wind speed, normalized
# cnt - number of total rentals

### Understanding the data:
# ## Seasons
# plt.title('Distribution of seasons')
# plt.xlabel('Season')
# plt.ylabel('Renting')
# train_data['season'].hist(bins=7) #I put 8 bins to have a nice spacing between each season
# plt.show()
# ## Seems like 2 and 3 (summer and fall) are the most popular seasons
#
# ## Weather
# plt.title('Distribution of weather')
# plt.xlabel('Weather')
# plt.ylabel('Renting')
# train_data['weathersit'].hist(bins=7) #I put 8 bins to have a nice spacing between each season
# plt.show()
# ## Seems like Clear weather is by far superior
#
# ## Hourly
# x = train_data['hr']
# y = train_data['cnt']
# sns.barplot(x=x, y=y, palette='pastel')
# plt.show()
# ## Around 8h00 and 17h00 - 18h00 it's peak moment
#
# ## Daily
# x = train_data['weekday']
# y = train_data['cnt']
# sns.barplot(x=x, y=y, palette='pastel')
# plt.show()
# ## Amount of bikes rented is pretty consistent per day
#
# ## Rentals Per hour Per day
# daily_rentals = pd.DataFrame(train_data.groupby(['hr', 'weekday'], sort = True)['cnt'].mean()).reset_index()
# x = daily_rentals['hr']
# y = daily_rentals['cnt']
# h = daily_rentals['weekday']
# sns_plt = sns.lineplot(x=x, y=y, hue = h, data = daily_rentals, palette='rainbow')
# ## 0 and 6 are Sunday and Saturday respectively. This plot shows the difference in rental behavior during workdays and weekends
# plt.show()
#
# ## Rentals with respect to temperature
# ## Temperature is normalized between 0 and 1
# x = train_data['temp']
# y = train_data['cnt']
# sns.lineplot(x=x, y=y, data=train_data)
# plt.show()
# ## Shows that around temp=0.68 there is a huge increase in rentals
#
# ## Plot of amount of rentals with respect to the temperature PER season
# daily_rentals = pd.DataFrame(train_data.groupby(['temp', 'season'], sort = True)['cnt'].mean()).reset_index()
# x = daily_rentals['temp']
# y = daily_rentals['cnt']
# h = daily_rentals['season']
# sns_plt = sns.lineplot(x=x, y=y, hue = h, data = daily_rentals, palette='rainbow')
# plt.show()
# ## Not sure if this one is useful


### Setting up Linear Regression
output_data = train_data.loc[:, 'cnt']
input_data = train_data.drop(['cnt'], axis=1)
print("y data: ", output_data.shape)
print("x data: ", input_data.shape)

lin_reg = LinearRegression().fit(input_data, output_data)
prediction = lin_reg.predict(test_data)
print("prediction", prediction.shape)

### Setting up the output doc
id_column = [x for x in range(1, 4345)]
cnt_column = ['cnt']
prediction_df = pd.DataFrame(index=id_column, columns=cnt_column)
prediction_df.columns.name = 'Id'
prediction_df['cnt'] = [int(pred) for pred in prediction]

# to make sure each value is 0 or higher
prediction_df = prediction_df.clip(lower = 0)

# print("prediction df")
# print(prediction_df.head())

### Putting everything in the output file
prediction_df.to_csv('./output.csv', index_label='Id')

