### Comparing categorical and quantitative variables

## Boxplots

# # Import library and create a new data set.
library(tidyverse)
wages <- read.csv(file.choose(), header = TRUE)
new_wage <- filter(wages, wage < 300)

# Specify the ggplot function:
ggplot(new_wages, aes(x = education, y = wage)) +
  geom_boxplot() # Specify the geom_boxplot function.

# Specify the ggplot function:
ggplot(new_wages, aes(x = education, y = wage)) +
  geom_boxplot(fill = "red", notch = TRUE, outlier.colour = "red") +
  labs(title = "Wages by education level") + # Specify the titles.
  theme_minimal() # Add a 'minimal' theme.

## Violinplots

# Specify the ggplot function:
ggplot(new_wages, aes(x = education, y = wage)) +
  geom_violin() # Specify the geom_violin function.

# Specify the ggplot function:
ggplot(new_wages, aes(x = education, y = wage)) +
  geom_violin(fill = "red", alpha = 0.5) +
  labs(title = "Wages by education level",
       x = "Education level",
       y = "Wage in $ (x1000)")

# Specify the ggplot function:
ggplot(new_wages, aes(x = education, y = wage)) +
  geom_violin(fill = "red") + #Specify the geom_violin function and fill.
  geom_boxplot(fill = "green",
               width = 0.25,
               outlier.colour = "green",
               outlier.size = 1,
               outlier.shape = "square") # Specify the geom_boxplot and adjust.
