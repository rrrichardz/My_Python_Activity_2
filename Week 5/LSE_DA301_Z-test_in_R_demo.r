######################################################################

# Standard deviation = average amount of variability in a data set


#   Z-test basic syntax:
#   z.test(x, y, alternative='two.sided', mu=0, sigma.x=NULL, sigma.y=NULL,conf.level=.95)

#   Where:
#   x: values for the first sample
#   y: values for the second sample (when performing a two sample z-test)
#   alternative: the alternative hypothesis ('greater', 'less', 'two.sided')
#   mu: mean under the null (pop. mean) or mean difference (in two sample case)
#   sigma.x: population standard deviation of first sample
#   sigma.y: population standard deviation of second sample
#   conf.level: confidence level to use

######################################################################

# Null hypothesis: mu ≥ 90
# Alternate hypothesis: mu < 90

#####################################################################

# Install the BSDA package, which contains the z-test function
install.packages("BSDA")

library(BSDA)
library (tidyverse) 

# One sample z-test

# Create a data set of package weights from the sorting facility
weight = c(90.1, 90.3, 89.8, 89.2, 89.1, 89.4, 90.2,
           89.9, 88.6, 89.6, 89.7, 88.8, 90.5, 89.5, 89.5)

# Check normal distribution
qqnorm(weight)
qqline(weight)

# We know that the population mean should be 90
# Let's assume the standard deviation is 1.2

# Run our test
z.test(weight, mu = 90, sigma.x = 1.2)

# Our p-value is greater than 0.05
############################################################################

# We'll try a sample from another sorting facility and compare means

# Null: mu1 - mu2 = 0 
# Alternate: mu1 - mu2  ≠ 0

fac1 = c(90.1, 90.3, 89.8, 89.2, 89.1, 89.4, 90.2,
         89.9, 88.6, 89.6, 89.7, 88.8, 90.5, 89.5, 89.5)

fac2 = c(90.1, 91.0, 90.8, 89.1, 90.1, 88.4, 89.2,
         89.9, 89.6, 88.7, 90.1, 88.5, 90.4, 89.0, 89.5)

# We'll check normal distribution from fac2
qqnorm(fac2)
qqline(fac2)

# Run our test
z.test(fac1, fac2, mu = 90, sigma.x = 1.2, sigma.y = 1.1)

# Our p-value is greater than 0.05