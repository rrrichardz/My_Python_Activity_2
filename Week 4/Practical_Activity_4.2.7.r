### 1. Prepare your workstation

# Install tidyverse package.
install.packages("tidyverse")

# Import tidyverse library.
library(tidyverse)

# Import the data set.
wages <- read.csv(file.choose(), header = TRUE)

# Print the data frame.
wages

# View the data frame.
View(wages)

# Summary of the data frame.
summary(wages)

### 2. View the visualisations: plot 1

# View the first plot
qplot(age, jobclass, colour = education, data = wages)

# Looks at age, job class, and education (swap axes and add jitter).
qplot(jobclass, age, colour = education, data = wages, geom = c("point",
                                                                "jitter"))

### 3. View the visualisations: plot 2 

# View the second plot
qplot(age, education, colour = race, data = wages, geom = "col")

# Looks at age, education, and race (add jitter).
qplot(age, education, colour = race, data = wages, geom = c("point", "jitter"))

### 4. View the visualisations: plot 3

# View the third plot
qplot(education, wage, shape = race, data = wages, geom = "boxplot")

# Looks at education, wage, and race (change variables around, making wage 
# the x-axis and race the y-axis, and change colour to fill).
qplot(wage, race, colour = education, data = wages, geom = "boxplot")
