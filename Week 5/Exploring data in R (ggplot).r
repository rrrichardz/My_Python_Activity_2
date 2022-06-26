### 1. Set up your workstation

# Import tidyverse library.
library(tidyverse)

# Import data set.
wages <- read.csv(file.choose(), header = TRUE)

# Check the data set.
summary(wages)

### 2. Build a plot

# Set the data source and add mapping elements.
ggplot(data = wages, mapping = aes(x = age, y = wage))

### 3. Add geoms

# Add layers to a plot using a plus sysbol:
ggplot(data = wages, mapping = aes(x = age, y = wage)) + geom_point()

### 4. Create a scatterplot

# Create a new object, specify the data to pass to the object (with assignment
# operator), and remove outliers with filter function:
new_wages <- filter(wages, wage < 300)

# Specify the ggplot function and the geom_point function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage)) + 
  geom_point(colour = "red", alpha = 0.5, size = 3)

### 5. Adding a smoothing line

# Add the line-of-best-fit to the plot.
ggplot(data = new_wages, mapping = aes(x = age, y = wage)) + 
  geom_point(colour = "red", alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", colour = "green", alpha = 0.5)

### 6. Group variables

# Run the ggplot function (specify colour in aes):
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5)

### 7. Add scales

# Run the ggplot function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 90, 5)) + # Add a scale layer for x.
  scale_y_continuous(breaks = seq(0, 350, 50)) # Add a scale layer for y.

# Run the ggplot function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 90, 5), "Age of the Individual") + # Add a scale
  # layer for x and argument/title.
  scale_y_continuous(breaks = seq(0, 350, 50), "Wage in $1000s") + # Add a scale layer
  # for y and argument/title.
  scale_colour_manual(values = c("red", "blue", "green", "orange", "yellow")) # Add layers
  # (colours).

### 8. Create faceted plots

# Run the ggplot function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 90, 5), "Age of the Individual") + 
  scale_y_continuous(breaks = seq(0, 350, 50), "Wage in $1000s") + 
  scale_colour_manual(values = c("red", "blue", "green", "orange", "yellow")) +
  facet_wrap(~jobclass) # Add a facet layer.

### 9. Add labels

# Run the ggplot function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 90, 5), "Age of the Individual") + 
  scale_y_continuous(breaks = seq(0, 350, 50), "Wage in $1000s") + 
  scale_colour_manual(values = c("red", "blue", "green", "orange", "yellow")) +
  facet_wrap(~jobclass) +
  labs(title = "Relationship between wages and age",
  subtitle = "Survey from the mid-Atlantic region, USA",
  caption = "Source: US govt data") # Add labels for title, subtitle and caption.

# Run the ggplot function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 90, 5)) + 
  scale_y_continuous(breaks = seq(0, 350, 50)) + 
  scale_colour_manual(values = c("red", "blue", "green", "orange", "yellow")) +
  facet_wrap(~jobclass) +
  labs(title = "Relationship between wages and age",
       subtitle = "Survey from the mid-Atlantic region, USA",
       caption = "Source: US govt data",
       x = "Individuals' Age",
       y = "Wage in $ (x1000)",
       color = "Education Level")
       # Add labels to labs function.

### 10 Customise the theme

# Run the ggplot function:
ggplot(data = new_wages, mapping = aes(x = age, y = wage, colour = education)) + 
  geom_point(alpha = 0.5, size = 3) +
  geom_smooth(method = "lm", se = FALSE, size = 1.5) +
  scale_x_continuous(breaks = seq(0, 90, 5)) + 
  scale_y_continuous(breaks = seq(0, 350, 50)) + 
  scale_colour_manual(values = c("red", "blue", "green", "orange", "yellow")) +
  facet_wrap(~jobclass) +
  labs(title = "Relationship between wages and age",
       subtitle = "Survey from the mid-Atlantic region, USA",
       caption = "Source: US govt data",
       x = "Individuals' Age",
       y = "Wage in $ (x1000)",
       color = "Education Level") +
  theme_bw() # Add a theme layer
