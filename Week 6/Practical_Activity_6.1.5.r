### Practical Activity 6.1.5

## Prepare your workstation

# Import package.
library(tidyverse)

# Import data set.
df_ether <- read.csv(file.choose(), header = TRUE)

# Sense-check the data set.
summary(df_ether)

## Create predictive model

# Create linear regression model.
model1 <- lm(Adj.Close_Eth ~ Open_sp, data = df_ether)

# Print the summary.
summary(model1)

# Calculate sum of least squares for this model.
SSE <- sum(model1$residuals^2)
SSE

# Create a second linear regression model.
model2 <- lm(Adj.Close_Eth ~ Close_sp, data = df_ether)

# Print the summary.
summary(model2)

# Calculate the sum of least squares for the second model.
SSE2 <- sum(model2$residuals^2)
SSE2

# Create a third linear regression model.
model3 <- lm(Adj.Close_Eth ~ Adj.Close_sp, data = df_ether)

# Print the summary.
summary(model3)

# Calculate the sum of least squares for the third model.
SSE3 <- sum(model3$residuals^2)
SSE3

## After comparing the summary statistics, it shows that model2 and
## model 3 are identical, and are both better than model1.