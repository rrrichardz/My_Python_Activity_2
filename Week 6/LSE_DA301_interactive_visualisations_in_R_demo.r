#############################################################################

## Creating interactive visualisations in R with Plotly

############################################################################

## Install the packages and import the libraries
install.packages("plotly")
install.packages("ggplot2") 

library(plotly)
library(tidyverse)

## Import the built-in motor cars data set
cars <- mtcars

## View the object type
typeof(cars)

## Convert to a data frame
cars_df <- as.data.frame(cars)

## Sense-check the data viewing the top six rows
head(cars_df)

## Use Plotly to create a plot with one variable
plot_ly(cars_df,
        x = ~wt)

## Add another variable and select a plot type - bar charts
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        type = "bar")

## With two variables, allow Plotly to select the chart type
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg)

## Add a third variable to the plot, cylinder (cyl)
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        type = "scatter",
        mode = "markers",
        color = ~cyl)

## Convert the cylinder variables to categorical values (factors)
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        type = "scatter",
        mode = "markers",
        color = ~factor(cyl))

## Add symbols to the plot
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        type = "scatter",
        mode = "markers",
        color = ~factor(cyl),
        symbol = ~cyl,
        symbols = c('circle', 'x', 'o'))


## Increase  the symbol size and transparency
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        type = "scatter",
        mode = "markers",
        color = ~factor(cyl),
        symbol = ~cyl,
        symbols = c('circle','x','o'),
        size = 2,
        alpha = 1)

## Create a 3D plot with an x, y, and z-axis.
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        z = ~cyl,
        color = ~factor(gear))

## Create an animated scatter plot using cylinder in the frame parameter
plot_ly(cars_df,
        x = ~wt,
        y = ~mpg,
        type = "scatter",
        mode = "markers",
        frame = ~cyl,
        showlegend = FALSE)

## Assign the animated plot to the object viz
viz <- plot_ly(cars_df,
               x = ~wt,
               y = ~mpg,
               type = "scatter",
               mode = "markers",
               frame = ~cyl,
               showlegend = FALSE)
  
## Edit and alter animation features, such as the button, slider, and transitions
  viz %>%
  animation_button(x = 1, xanchor = "right", y = 1, yanchor = "bottom")%>% 
  animation_slider(currentvalue = list(prefix = "Cylinders ", font = list(color="blue")) 
  ) %>% 
  animation_opts(frame = 10000, easing = 'circle-in')

  