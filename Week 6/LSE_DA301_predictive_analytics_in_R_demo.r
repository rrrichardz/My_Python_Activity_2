#############################################################################

## Predictive analytics (Simple linear regression)
## in R convenor demo

############################################################################

## Load the tidyverse library
library(tidyverse)

## Import the data set
## cpi <- read.csv("cpi.csv")

cpi <- read.csv(file.choose (), header = T)

## Explore the data set
summary(cpi)
head(cpi)

## Identify relationships between the two variables - year and index
## Find correlation
cor(cpi)

# Plot the relationship with base R graphics
plot(cpi$Year, cpi$Index)

## Fit the simple linear regression model
model1 <- lm(Index ~ Year, data = cpi)
## View the model
model1

## View more outputs for the model - the full regression table
summary(model1)

## Year is a highly significant value, it explains over 83% of the variability

## View residuals on a plot
plot(model1$residuals)

## Add line-of-best-fit
plot(cpi$Year, cpi$Index)
abline(coefficients(model1))

## Complete a log transformation with dplyr's mutate() function
cpi <- mutate(cpi,
              logIndex = log(Index))

## View new object with new variable
head(cpi)

## Create a new model using logIndex
model2 <- lm(logIndex ~ Year, data = cpi)

## View full regression table
summary(model2)

## Plot the relationship between year and logIndex
plot(cpi$Year, cpi$logIndex)

## Add a line-of-best fit
abline(coefficients(model2))


## Make a forecast with this model
## View the last six rows of the data set
tail(cpi)

## Create a new data frame for the forecast values
cpiForecast <- data.frame(Year = 2022:2025)

## Predict from 2022 to 2025
predict(model2, newdata = cpiForecast)

## Add the values to the cpiForecast data frame
cpiForecast$logIndex <- predict(model2, newdata = cpiForecast)

## Add the actual index as opposed to the log index by exponentiation
cpiForecast <- mutate(cpiForecast,
                      Index = exp(logIndex))

## View the cpiForecast data frame
cpiForecast

## Remember to save your work