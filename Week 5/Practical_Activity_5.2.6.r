### Practical Activity 5.2.6

## 1. Prepare your workstation

# Import necessary libraries.
library(tidyverse)
library(dplyr)
library(ggplot2)

# Import the data set.
health <- read.csv(file.choose(), header = TRUE)

# Check the data set.
str(health)
summary(health)

## 2. Create visualisations

# Create a scatterplot.
ggplot(health, mapping = aes(x = age, y = charges)) + geom_point()

# Remove the outliers >50000.
new_health <- filter(health, charges < 50000)

# Change colours, size and alpha of points of the plot.
ggplot(health, mapping = aes(x = age, y = charges)) + 
  geom_point(colour = "green", alpha = 0.5, size = 1.5)

# Add labels to the axes and change scales.
ggplot(health, mapping = aes(x = age, y = charges)) + 
  geom_point(colour = "green", alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 70, 5), "Age of the Individual") +
  scale_y_continuous(breaks = seq(0, 55000, 5000), "Monthly charges (in $)")

# Add a third variable 'smoker'.
ggplot(health, mapping = aes(x = age, y = charges, colour = smoker)) + 
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 70, 5), "Age of the Individual") +
  scale_y_continuous(breaks = seq(0, 55000, 5000), "Monthly charges (in $)")

# Add a title and subtitle.
ggplot(health, mapping = aes(x = age, y = charges, colour = smoker)) + 
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 70, 5), "Age of the Individual") +
  scale_y_continuous(breaks = seq(0, 55000, 5000), "Monthly charges (in $)") +
  labs(title = "Relationship between age and charges",
       subtitle = "A survey from a health insurance provider")

# Facet the plot by gender.
ggplot(health, mapping = aes(x = age, y = charges, colour = smoker)) + 
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 70, 5), "Age of the Individual") +
  scale_y_continuous(breaks = seq(0, 55000, 5000), "Monthly charges (in $)") +
  labs(title = "Relationship between age and charges",
       subtitle = "A survey from a health insurance provider") +
  facet_wrap(~sex)

# Facet the plot by region.
ggplot(health, mapping = aes(x = age, y = charges, colour = smoker)) + 
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 70, 5), "Age of the Individual") +
  scale_y_continuous(breaks = seq(0, 55000, 5000), "Monthly charges (in $)") +
  labs(title = "Relationship between age and charges",
       subtitle = "A survey from a health insurance provider") +
  facet_wrap(~region)

# Facet the plot by number of children.
ggplot(health, mapping = aes(x = age, y = charges, colour = smoker)) + 
  geom_point(alpha = 0.5, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 70, 5), "Age of the Individual") +
  scale_y_continuous(breaks = seq(0, 55000, 5000), "Monthly charges (in $)") +
  labs(title = "Relationship between age and charges",
       subtitle = "A survey from a health insurance provider") +
  facet_wrap(~children)
