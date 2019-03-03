import numpy as np
import pandas as pd
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

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
## Seasons
plt.title('Distribution of seasons')
plt.xlabel('Season')
plt.ylabel('Renting')
train_data['season'].hist(bins=8) #I put 8 bins to have a nice spacing between each season
plt.show()
## Seems like 2 and 3 (summer and fall) are the most popular seasons

## Weather
plt.title('Distribution of weather')
plt.xlabel('Weather')
plt.ylabel('Renting')
train_data['weathersit'].hist(bins=8) #I put 8 bins to have a nice spacing between each season
plt.show()
## Seems like Clear weather is by far superior

## Hourly
x = train_data['hr']
y = train_data['cnt']
sns.barplot(x=x, y=y, palette='pastel')
plt.show()
## Around 8h00 and 17h00 - 18h00 it's peak moment

## Daily
x = train_data['weekday']
y = train_data['cnt']
sns.barplot(x=x, y=y, palette='pastel')
plt.show()
## Amount of bikes rented is pretty consistent per day

## Rentals Per hour Per day
daily_rentals = pd.DataFrame(train_data.groupby(['hr', 'weekday'], sort = True)['cnt'].mean()).reset_index()
x = daily_rentals['hr']
y = daily_rentals['cnt']
h = daily_rentals['weekday']
sns_plt = sns.lineplot(x=x, y=y, hue = h, data = daily_rentals, palette='rainbow')
plt.show()
## 0 and 6 are Sunday and Saturday respectively. This plot shows the difference in rental behavior during workdays and weekends