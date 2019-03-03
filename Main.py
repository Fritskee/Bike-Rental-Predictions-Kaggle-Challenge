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
# weathersit - 1: Clear, Few clouds, Partly cloudy, Partly cloudy, 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog temp - temperature in Celsius, normalized
# atemp - "feels like" temperature in Celsius, normalized
# hum - relative humidity, normalized
# windspeed - wind speed, normalized
# cnt - number of total rentals

