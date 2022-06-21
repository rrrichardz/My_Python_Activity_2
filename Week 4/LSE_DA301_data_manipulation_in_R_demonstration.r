## Install the tidyverse package
library('tidyverse')

## Import the data set
wages <- read.csv(file.choose(), header = TRUE)

## Explore the data set
head(wages)

## Convert data frame to a tibble
as.tibble(wages)

## Use the glimpse() function
glimpse(wages)

## Use the summary() function
summary(wages)

## Return a frequency table for the 'sex' column
table(wages$sex)

## Return a frequency table for the 'region' column
table(wages$region)

## Get rid of 'sex' and 'region'

# Remove the columns
wages2 <- select(wages, -sex, -region)

# Check the new data frame
head(wages2)

## Convert 'race' to factor (categorical variable)

wages3 <- mutate(wages2,
                 race = as.factor(race))

# View as a tibble
as_tibble(wages3)

# View summary
summary(wages3)

## Check why 'age' is not numeric

wages4 <- mutate(wages3,
                 ageNum = as.numeric(age))

# Check the ageNum data
summary(wages4$ageNum)

# If we want to delete a row
wages5 <- filter(wages4,
                 !is.na(ageNum))

# Check the wages5 data set dimensions
dim(wages5)

# Find the row where ageNum is NA
filter(wages4, is.na(ageNum))

# Fix the error 5o in the ageNum column
# Find the row with the NA value
which(is.na(wages4$ageNum))

# Correcting the value "5o" to 50
wages4$ageNum[which(is.na(wages4$ageNum))] <- 50

# Check the data set to see if the error is fixed
summary(wages4$ageNum)

## View summary of wages4
summary(wages4)

# Explore the height data in a plot
plot(hist(wages4$height))

# View top rows of wages4 height column to see values
head(wages4$height)

# Create a data set with cm values
wagesCm <- filter(wages4,
                  height > 100)

# Quick look at wagesCm height
summary(wagesCm$height)

## Convert centimeters to inches
wagesCm <- mutate(wagesCm,
                  height = round(height / 2.54))

# Check the data
summary(wagesCm$height)

## Combine the two data sets

wagesInches <- filter(wages4, 
                      height <= 100)

wagesClean <- rbind(wagesInches, wagesCm)

# Final look at wagesClean
summary(wagesClean)

# Save the data set as a csv file
write.csv(wagesClean, "Wages_clean.csv")

# Save the script using Ctrl+S