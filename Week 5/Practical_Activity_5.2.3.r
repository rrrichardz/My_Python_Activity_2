### Practical Activity 5.2.3

## 1. Prepare your workstation

# Import the necessary libraries
library(ggplot2)
library(tidyverse)
library(dplyr)
library(moments)

# Import the data file.
health <- read.csv(file.choose(), header = TRUE)

# Sense check the data set
head(health)
str(health)

## 2. Statistical analysis

# Measure the central tendencies of the 'BMI' variables:
mean(health$bmi)
median(health$bmi)

# Measure statistics of extreme values (max and min):
min(health$bmi)
max(health$bmi)

# Measure statistics of variability (range, Q1, Q3, IQR, variance, sd):
max(health$bmi) - min(health$bmi)  # Range = Maximum - Minimum.
quantile(health$bmi, 0.25)  # Function to calculate Q1.
quantile(health$bmi, 0.75)   # Function to calculate Q2.
IQR(health$bmi)    # Function to calculate IQR.
var(health$bmi) # Function to determine the variance.
sd(health$bmi) # Function to return t.

# Check for normality with qqplot, Shapiro-Wilk test:
# qqplot.
qqnorm(health$bmi)
# Add a reference line.
qqline(health$bmi, col = "green")
# Shapiro-Wilk test.
shapiro.test(health$bmi)
# Our p-value is well above 0.05, and we can conclude normal distribution.

# Check for skewness and kurtosis:
skewness(health$bmi)
# Our ouput suggests a postive skewness.
kurtosis(health$bmi)
# Our kurtosis value is less than 3, suggesting our data is platykurtic.

# Check the correlation between 'BMI' and 'age' variables:

# First check normality for 'age' variables.
shapiro.test(health$age)
# Our output is greater than 0.5, and we can assume normality.
# # Check correlation using Pearson's correlation.
cor(health$bmi, health$age)
# Our correlation coefficient of around 0.11 suggests a weak positive correlation.