### Adding interactivity to visualisations

## Create a boxplot

# Install the package.
install.packages("plotly")

# Call the plotly library.
library(plotly)

# Create the plot - this is a basic plot and data set included in the
# package:
fig <- plot_ly(midwest, x = ~percollege, colour = ~state,
               type = "box")

# Print the plot.
fig
