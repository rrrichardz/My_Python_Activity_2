### 1. Simplify the view

# Import tidyverse library.
library(tidyverse)

# Import the data file.
wages <- read.csv(file.choose(), header = TRUE)

# Print/return the data frame.
wages

# View the wages data as a tibble.
as_tibble(wages)

### 2. Summarise the data set

# Create a summary view of the wages data set.
summary(wages)

# Create a glimpse of the wages data set.
glimpse(wages)

### 3. Remove columns

# Create a new data frame from a subset of the wages data frame, and remove
# the sex and region columns.
wages2 <- subset(wages, select = -c(sex, region))

glimpse(wages2)

### 4. Subset observations (filter)

filter(wages, age == 25, wage > 110)

### 5. Order observations (arrange)

arrange(wages, race, jobclass, education)

arrange(wages, desc(year))

arrange(wages, year)

### 6. Add a new columns (mutate)

# To create a new element by dividing wage by logwage.
mutate(wages, new_var = wage / logwage)
