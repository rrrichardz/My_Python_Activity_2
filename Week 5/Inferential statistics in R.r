### 1. Install the packages

# Activate the tidyverse package.
library(tidyverse)
# Activate the moments package.
library(moments)

### 2. Import the sense-check the data

# Import the data file:
drive1 <- read.csv(file.choose(), header = TRUE)

# View the data set.
View(drive1)

# View a summary of the data set.
summary(drive1)

### 3. Remove unnecessary columns

# Assign data to the object and  remove the 'n' column:
drive2 <- select(drive1, -n)

# Prinit/convert the data set into a tibble:
as_tibble(drive2)

# Create new object and specify the function:
drive3 <- round(drive2, digits = 2)

# Print/convert the data set into a tibble:
as_tibble(drive3)

# Check for measures of central tendency (mean and median):
mean(drive3$total_seconds)
median(drive3$total_seconds)

# Check for measures of variability (i.e. range, IQR, max and min):
quantile(drive3$total_seconds, 0.25)
quantile(drive3$total_seconds, 0.75)
IQR(drive3$total_seconds)
sd(drive3$total_seconds)
max(drive3$total_seconds)
min(drive3$total_seconds)
max(drive3$total_seconds) - min(drive3$total_seconds)

### 4. Check for normality distribution

# Specify the qqnorm function; draw a qqplot using the total_seconds column:
qqnorm(drive3$total_seconds, col = "blue", xlab = "z Value",
       ylab = "Time (Seconds)")

# Specify the qqline function; add a reference line to the qqplot:
qqline(drive3$total_seconds, col = "red", lwd = 2)

# Run a Shapiro-Wilk test:
shapiro.test(drive3$total_seconds)

# Run a skewness and kurtosis test on the data.
skewness(drive3$total_seconds)
kurtosis(drive3$total_seconds)

### 5. Perform the t-test

# Specify the t.test function; set the data source, the confidence interval
# (95%) and the theoretical mean:
t.test(drive3$total_seconds, conf.level = 0.95, mu = 120)

### 6. Perform the z-test (On separate demonstration)


### Pearson's correlation coefficient in R

# Check for normal distribution with Shapiro-Wilk test:
shapiro.test(drive3$car_stop)
shapiro.test(drive3$car_go)
shapiro.test(drive3$take_order)
shapiro.test(drive3$hand_over_order)

# Specify the cor function; set the first and second variables:
cor(drive1$car_stop, drive1$car_go)

# Specify the cor function again and set first and second variables:
cor(drive1$take_order, drive1$hand_over_order)

# Check the correlation between all the variables in the data set:
round(cor(drive1), digits = 2) # Round to 2 decimal places.
