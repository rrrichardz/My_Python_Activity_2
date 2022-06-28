## 1. Import the required packages

# Install and import libraries for time series analysis:
install.packages("forecast")
install.packages("tseries")
library(ggplot2)
library(forecast)
library(tseries)

# Import data set and assign to new object:
FRED_base <- read.csv(file.choose(), header = TRUE)

# Sense-check the data set:
summary(FRED_base)

## 2. Change the column names

# Specify the column, change the variables, and format the date:
FRED_base$DATE <- as.Date(FRED_base$DATE, format = "%d/%m/%Y")
colnames(FRED_base) <- c("Date", "Value") # Specify the new columns.
head(FRED_base) # Check the data set.

## 3. Create time-series objects

# Specify the min and max functions and set the parameters:
min(FRED_base$Date, na.rm = TRUE)
max(FRED_base$Date, na.rm = TRUE)

# Create new data frame and assign time-series value, and specify the 'ts'
# function:
FRED_ts <- ts(FRED_base$Value, start = c(1972, 1), end = c(2021, 12),
              frequency = 12)
# Sense-check the new object.
head(FRED_ts)
# Check the object's structure.
str(FRED_ts)

## 4. Check for missing values and the frequency

# Check for missing values.
sum(is.na(FRED_ts))

# Check the frequency of the data.
frequency(FRED_ts)

# Show the datapoint's position in the cycle.
cycle(FRED_ts)

## 5. Plot the data (line plot)

# Specify the plot function; set the data source, label both axes, and add
# a title:
plot(FRED_ts, xlab = "Year", ylab = "Value", main = "Years from 1972 to 2022")

## 6. Display the data/check for outliers

# Specify the boxplot function; specific operands, label both axes, and add
# a title:
boxplot(FRED_ts~cycle(FRED_ts), xlab = "Month", ylab = "Value",
        main = "Months from 1972 to 2022")

## 7. Decompose the data

# Extract and plot the main components to decompose the time series:
components_FRED_ts <- decompose(FRED_ts)
plot(components_FRED_ts)

### Testing for stationarity and autocorrelation

## 8. Testing stationarity

# Test stationarity with augmented ADF test:
adf.test(FRED_ts)

## 9. Testing autocorrelation

# Review random time-series variables.
components_FRED_ts$random

# Plot valuese removing NA values while doing so:
autoplot(acf(na.remove(components_FRED_ts$random), plot = FALSE)) +
  labs(title = "Randomness value") + # Add a title.
  theme_classic() # Set the theme.

# Plot random variables to check the distribution:
hist((components_FRED_ts$random))

### Predicting with the ARIMA model

# Fit the model to our time series:
arima_FRED_ts <- auto.arima(FRED_ts)

# Make a forecast for the next three months:
forecast3_FRED_ts <- forecast(arima_FRED_ts, 3)

# Plot the forecast on a graph:
autoplot(forecast3_FRED_ts) + theme_classic()

# Print the values in the data frame.
forecast3_FRED_ts

# Extend the prediction; set the data source, the time span (number of months), 
# and assign a new object:
forecast2_FRED_ts <- forecast(arima_FRED_ts, 24)

# Plot the output and set the theme:
autoplot(forecast2_FRED_ts) + theme_classic()

# Print the values in the data frame.
forecast2_FRED_ts

### Testing for accuracy

## 1. Extract a subset of the values

# (Training data) Create a new time series object and assign the values and 
# parameters:
FRED_train_ts <- window(FRED_ts, start = c(1972, 1), end = c(2020, 12),
                        frequency = 12)

# (Test data) Create a new time series object and assign the values and
# parameters:
FRED_test_ts <- window(FRED_ts, start = c(2021, 1), end = c(2021, 12),
                       frequency = 12)

## 2. Fit the model to the training data

# Create a new object and specify the forecast function and pass the ARIMA
# model:
forecast_FRED_train_ts <- forecast(auto.arima(FRED_train_ts), 12)

# Plot the values and forecast and add a theme:
autoplot(forecast_FRED_train_ts) + theme_classic()

## 3. Plot the test set

# Plot test set onto the graph: specify the lines function, set the source,
# and specify the line colour:
lines(FRED_test_ts, col = "red")

## 4. Check the MAPE value (accuracy)

# Check the accuracy of the prediction:
accuracy(forecast_FRED_train_ts, FRED_test_ts)
