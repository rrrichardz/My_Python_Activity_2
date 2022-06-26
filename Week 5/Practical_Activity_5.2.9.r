### Practical Activity 5.2.9

## Prepare your workstation

# Import the necessary libraries
library(ggplot2)
library(tidyverse)
library(dplyr)

# Import the data set.
health <- read.csv(file.choose(), header = TRUE)

# Sense check the data set.
head(health)
str(health)

## Create visualisations

# Plot age on a histogram.
ggplot(health, aes(x = age)) +
  geom_histogram(stat = "count")

# Plot number of children on a histogram.
ggplot(health, aes(x = children)) +
  geom_histogram(stat = "count")

# Plot region and sex on a stacked bar chart
ggplot(health, aes(x = region, fill = sex)) +
  geom_bar()

# Plot smoker and sex on a grouped bar chart.
ggplot(health, aes(x = smoker, fill = sex)) +
  geom_bar(position = "dodge")

# Plot BMI and sex on a side-by-side boxplot. 
ggplot(health, aes(x = sex, y = bmi)) +
  geom_boxplot()

# Plot BMI and region on a side-by-side violin plot.
ggplot(health, aes(x = region, y = bmi)) +
  geom_violin()

# Plot BMI and smoker on a side-by-side boxplot.
ggplot(health, aes(x = smoker, y = bmi)) +
  geom_boxplot()

## Adjust visualisations

# Select two plots to add colour, titles, and labels.
ggplot(health, aes(x = smoker, y = bmi)) +
  geom_boxplot(fill = "green",
               notch = TRUE,
               outlier.color = "blue") +
  labs(title = "BMI of individual by smoking",
       x = "Is individual a smoker",
       y = "BMI")
  
ggplot(health,aes(x = smoker, fill = sex)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("purple", 'orange')) +
  labs(title = "Count of male and female smokers",
       x = "Is individual a smoker",
       y = "Count")

# Select two plots and and add a suitable theme.
# For a website:
ggplot(health, aes(x = region, y = bmi)) +
  geom_violin() +
  theme_dark()

# For a publication:
ggplot(health, aes(x = region, fill = sex)) +
  geom_bar() +
  theme_classic()
