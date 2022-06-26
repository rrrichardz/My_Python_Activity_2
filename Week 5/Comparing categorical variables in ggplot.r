### Comparing categorical variables

## Stacked bar chart

# Import library and create a new data set.
library(tidyverse)
wages <- read.csv(file.choose(), header = TRUE)
new_wage <- filter(wages, wage < 300)

# Specify the ggplot function; set data source and set and pass x:
ggplot(new_wages, aes(x = education, fill = jobclass)) +
  geom_bar() # Specify the geom_bar function.

## Segmented bar charts

# Specify ggplot function:
ggplot(new_wages, aes(x = education, fill = jobclass)) +
  geom_bar(position = "fill") + # Add position.
  labs(y = "Proportion") # Add a label to y.

## Grouped bar chart

# Specify ggplot function:
ggplot(new_wages, aes(x = education, fill = jobclass)) +
  geom_bar(position = "dodge") + # Set position to 'dodge'.
  scale_fill_manual(values = c("blue", "dark green")) # Change fill colours.


# Specify ggplot function:
ggplot(new_wages, aes(x = education, fill = jobclass)) +
  geom_bar(position = "dodge") + 
  scale_fill_manual(values = c("yellow", "purple")) +
  labs(x = "Levels of education",
       y = "Count",
       title = "Jobclass by levels of education",
       fill = "Jobclass") +
  scale_y_continuous(breaks = seq(0, 700, 100))
  