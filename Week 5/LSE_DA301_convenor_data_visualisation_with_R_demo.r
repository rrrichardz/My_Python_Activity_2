## Install library
library(ggplot2)

## Import data set wages_plots.csv
wages <- read.csv(file.choose(), header = TRUE)

## View top 6 rows
head(wages)

## Using qplot ##
## Examine a variable (wages) through a visualisation
qplot(wage, data = wages)

## Examine a variable (marital) through a visualisation
qplot(marital, data = wages)

## ## Examine a variable (age and wage) through a visualisation
qplot(age, wage, data = wages)

## Using ggplot ##

## Histogram
ggplot(wages, aes(x = wage)) + geom_histogram()

## Histogram with 20 bins
ggplot(wages, aes(x = wage)) + geom_histogram(bins = 20)

## Smoothed density plot
ggplot(wages, aes(x = wage)) + geom_density()

## Scatterplot
ggplot(wages, aes(x = age, y = wage)) + geom_point()

## Scatterplot with jitter
ggplot(wages, aes(x = jitter(age), y = wage)) + geom_point()

## Scatterplot with a line-of-best-fit from linear regression model
ggplot(wages, aes(x = age, y = wage)) + geom_point() + geom_smooth(method = lm)

## Scatterplot with no method in geom_smooth() (spline)
ggplot(wages, aes(x = age, y = wage)) + geom_point() + geom_smooth()

## Add a further/third variable as a colour and remove the smoothing line
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_point()

## Add a further/third variable as a colour with a smoothing line (spline)
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_point() + geom_smooth()

## Add a further/third variable and no standard error
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_point() + geom_smooth(se = FALSE)

## Add a further/third variable and no points
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_smooth()

## Categorical variable = marital
ggplot(wages, aes(x = marital)) + geom_bar()

## Comparing marital and education with colour (outline only)
ggplot(wages, aes(x = marital, col = education)) + geom_bar()

## Comparing marital and education with colour (fill)
ggplot(wages, aes(x = marital, fill = education)) + geom_bar()

## Comparing marital and education with colour (fill and side-by-side)
ggplot(wages, aes(x = marital, fill = education)) + geom_bar(position = 'dodge')

## Comparing marital and education add a title
ggplot(wages, aes(x = marital, fill = education)) + geom_bar(position = 'dodge') +
  ggtitle("Customer education level by marital status")

## Comparing age and wage and add a theme (minimal)
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_point() + geom_smooth() +
  theme_minimal()

## Comparing age and wage and add a theme (classic)
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_point() + geom_smooth() +
  theme_classic()

## Comparing age and wage and change line width
ggplot(wages, aes(x = age, y = wage, col = job_class)) + geom_point() + 
  geom_smooth(lwd = 2, se = FALSE) + theme_classic()

## Age, wages, and education with smoothing lines only 
ggplot(wages, aes(x = age, y = wage, col = education)) + geom_smooth(se = FALSE)
