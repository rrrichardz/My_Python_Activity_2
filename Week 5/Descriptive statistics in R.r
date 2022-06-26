### 1. Import tidyverse and create a new object

# Import the tidyverse package.
library(tidyverse)

# Create a new object 'speedy' and import the data file:
speedy <- read.csv(file.choose(), header = TRUE)

# View the new object as a 'tibble'.
as_tibble(speedy)

### 2. Find the mean and median

# Call the funciton to calculate the mean:
mean(speedy$total_minutes)

# Call the function to calculate the median:
median(speedy$total_minutes)

### 3. Measure the variability in values

# Determine the minimum value.
min(speedy$total_minutes)

# Determine the maximum value.
max(speedy$total_minutes)

# Determine the range = max - min.
max(speedy$total_minutes) - min(speedy$total_minutes)

# Calculate Q1.
quantile(speedy$total_minutes, 0.25)

# Calculate Q3.
quantile(speedy$total_minutes, 0.75)

# Calculate IQR.
IQR(speedy$total_minutes)

# Determine the variance.
var(speedy$total_minutes)

# Return the standard deviation.
sd(speedy$total_minutes)

### 4. Determine normality (histogram and boxplot)

# Specify histogram function:
hist(speedy$total_minutes)

# Specify boxplot function:
boxplot(speedy$total_minutes)

### 5. Determine normality (qqplots)

# Specify qqnorm function (draw a qqplot):
qqnorm(speedy$total_minutes)

# Specify qqline function.
qqline(speedy$total_minutes)

### 6. Determine normality (Shapiro-Wilk test)

# Specify shapiro.test function (Shapiro-Wilk test):
shapiro.test(speedy$total_minutes)

### 7. Determine normality (skewness and kurtosis)

# Install the moments package and load the library.
install.packages("moments")
library(moments)

# Specify the skewness and kurtosis functions:
skewness(speedy$total_minutes)
kurtosis(speedy$total_minutes)
