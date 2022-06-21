### 1. Import the data

# Install the tidyverse package.
install.packages("tidyverse")

# Import tidyverse library.
library(tidyverse)

# Import the data file.
wages <- read.csv(file.choose(), header = TRUE)

# Print/return the data frame.
wages

# View the data frame.
View(wages)

### 2. Summarise the data

# Check the data with summary() funciton.
summary(wages)

### 3. Create a scatterplot

# Specify X as age, y as wages, and wages as the data source (x-axis is passed
# first, followed by the y-axis, and then the source of the data):
qplot(age, wage, data = wages)

### 4. Create a histogram with an x-variable

# First pass the x-variable, then specify the data source:
qplot(age, data = wages)

### 5. Adjust histogram bins

# Pass the x-variable, then set the number of bins, and end with pasing the
# data source:
qplot(age, bins = 5, data = wages)

### 6. Create a scatterplot with a y-variable

# Assign wage to the y variable, followed by the data source:
qplot( y = wage, data = wages)

### Bar graphs

# Plot a bar chart by passing the x-variable and data source, then set the
# geom type:
qplot(age, data = wages, geom = "bar")

### Stacked bar chart
qplot(age, fill = education, data = wages, geom = "bar")

### Boxplot
qplot(wage, race, data = wages, geom = "boxplot")

### Changing aesthetics in qplots
qplot(wage, race, data = wages, colour = I("red"), geom = "boxplot")

### Practices

# Looking back at the age and wage scatterplot, what does the smoothing
# curve look like?
qplot(age, wage, data = wages, geom = c("point", "smooth"))

# How does the level of education map across age and wage?
qplot(age, wage, colour = education, data = wages)

# How do individuals' race and level of education relate to their wage?
qplot(race, education, colour = wage, data = wages)

qplot(race, education, colour = wage, data = wages, geom = c("point", "jitter"))

### Heatmaps
qplot(race, education, fill = wage, data = wages, geom = "raster")

### Facets in qplot
qplot(age, wage, data = wages, facets = education ~ race)
