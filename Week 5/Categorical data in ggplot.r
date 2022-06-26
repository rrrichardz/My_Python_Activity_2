### Categorical data

## 1. Set up your workstation

# Import tidyverse library and load the wage CSV file:
library(tidyverse)
wages <- read.csv(file.choose(), header = TRUE)

# Create new data with individuals with wages of less than 300.
new_wages <- filter(wages, wage < 300)

## 2. Plot the distribution

# Set data source and map the aes function to define x:
ggplot(new_wages, aes(x = maritl)) +
  geom_histogram(stat = "count") # Add a geom layer to specify the plot type.

## 3. Add attributes

# Specify the ggplot function:
ggplot(new_wages, aes(x = maritl)) +
  geom_histogram(fill = "red",
                 colour = "black",
                 stat = "count") + # Add fill, colour and a statistic.
  labs(x = "Marital status",
       y = "Frequency",
       title = "Individuals by marital status") # Add the labs function for labels.

## 4. Visualise the percentages

# Specify the ggplot function:
ggplot(new_wages, aes(x = maritl,
                      y = ..count.. / sum(..count..))) + # Specify 'y' to create a %.
  geom_histogram(fill = "red",
                 colour = "black",
                 stat = "count") + # Specify attributes.
  labs(x = "Marital status",
       y = "Frequency",
       title = "Individuals by marital status") + # Specify titles.
  scale_y_continuous(label = scales::percent) # Pass labels to the scale.  

## 5. Flip the axes

# Specify the ggplot function:
ggplot(new_wages, aes(x = maritl,
                      y = ..count.. / sum(..count..))) + 
  geom_histogram(fill = "red",
                 colour = "black",
                 stat = "count") + 
  labs(x = "Marital status",
       y = "Frequency",
       title = "Individuals by marital status") + 
  scale_y_continuous(label = scales::percent) +
  coord_flip() # Flip the axes.
