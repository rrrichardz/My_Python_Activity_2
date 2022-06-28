# Import library.
library(tidyverse)
library(ggplot2)
library(plotly)

# Import the data set:
wages <- read.csv(file.choose(), header = TRUE)

# Create an object 'plot' and specify the function.
plot <- ggplot(wages, aes(x = age, y = wage)) +
  geom_point()
ggplotly(plot) # Specify the function and pass the plot.
