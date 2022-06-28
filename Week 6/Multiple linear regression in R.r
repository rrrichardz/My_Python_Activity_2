## 1. Import the data and check correlation

# Import library
library(tidyverse)

# Import the data set.
wine <- read.csv(file.choose(), header = TRUE)

# Sense-check the data set.
summary(wine)

# Check for correlation between variables.
cor(wine)

## 2. Create a new regression model

# Create a new object and specify the lm function and the variables:
model1 <- lm(Price ~ AGST + HarvestRain, data = wine)

# Print the summary statistics.
summary(model1)

## 3. Change the name of the model and add x-variables

# Add new variables:
model2 <- lm(Price ~ AGST + HarvestRain + WinterRain + Age + FrancePop,
             data = wine)

# Print the summary.
summary(model2)

## 4. Improve the model

# First remove 'FrancePop' as it is insignificant.
model3 <- lm(Price ~ AGST + HarvestRain + WinterRain + Age,
             data = wine)

# Print the summary.
summary(model3)

# Again remove 'Age' as it is insignificant.
model4 <- lm(Price ~ AGST + HarvestRain + WinterRain, data = wine)

# Print the summary.
summary(model4)

## 5. Test the model

# Load the new data file and view its structure:
wineTest <- read.csv(file.choose(), header = TRUE)

str(wineTest)

# Create a new object and specify the predict function:
predictTest <- predict(model3, newdata = wineTest,
                       interval = "confidence")

# Print the object.
predictTest
