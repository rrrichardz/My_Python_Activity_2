### Multivariate data

# Import library and create a new data set.
library(tidyverse)
wages <- read.csv(file.choose(), header = TRUE)
new_wage <- filter(wages, wage < 300)

# Specify ggplot function and add 'shape = jobclass' and remove the line-of-best-fit to 
# the scatterplot:
ggplot(new_wages, aes(x = age, y = wage, colour = education, shape = jobclass)) +
  geom_point(alpha = 1, size = 3) + # Change 'alpha' values.
  scale_x_continuous(breaks = seq(0, 90, 5), "Age of the individual") +
  scale_y_continuous(breaks = seq(0, 350, 50), "Wage in $1000s") +
  scale_fill_brewer("set2") # Specify preloaded colour.
