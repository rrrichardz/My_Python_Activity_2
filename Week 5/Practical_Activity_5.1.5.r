### Practical Activity 5.1.5

## 1. Prepare your workstation

# Import necessary libraries.
library(tidyverse)
# Useful for importing data.
library(readr) 
# Useful for data wrangling.
library(dplyr) 
# Useful for data wrangling.
library(tidyr) 
# Useful for creating tidy tables.
library(knitr) 
# Useful for working with vectors and functions.
library(purrr)
# Useful to create insightful summaries of data set.
library(skimr)
# Useful to create insightful reports on data set.
library(DataExplorer)

# Import data file.
seatbelt <- read.csv(file.choose(), header = TRUE)

# Sense-check the data set.
as_tibble(seatbelt)
View(seatbelt)

## 2. Understand the data set

# Check for missing values.
sum(is.na (seatbelt))

sum(is.na (seatbelt$seatbelt))

# Replace missing values with 0.
seatbelt[is.na(seatbelt)] = 0

sum(is.na (seatbelt$seatbelt))

# View gist of data set.
summary(seatbelt)

DataExplorer::create_report(seatbelt)

# Drop column X.
seatbelt_df <- subset(seatbelt, select = -c(1))
head(seatbelt_df)

# Keep only numeric columns.
seatbelt_clean <- seatbelt_df %>% keep(is.numeric)
head(seatbelt_clean)

# Round all the columns to two decimal places.
seatbelt_clean <- round(seatbelt_clean, 2)
head(seatbelt_clean)

## 3. visualise the data set

# Create visualisation to check the distribution.
boxplot(seatbelt_clean$miles)
boxplot(seatbelt_clean$fatalities)
boxplot(seatbelt_clean$income)
boxplot(seatbelt_clean$age)
boxplot(seatbelt_clean$seatbelt)

## 4. Perform data manipulation

# Calculate the sum, max, and min of all the columns.
seatbelt_a <- apply(seatbelt_clean, 2, sum)
seatbelt_a <- round(seatbelt_a, 2)
seatbelt_a

seatbelt_l <- lapply(seatbelt_clean, min)
seatbelt_l

seatbelt_clean <- seatbelt_df %>% keep(is.numeric)
seatbelt_s <- sapply(seatbelt_clean, min)
seatbelt_s <- round(seatbelt_s, 2)
seatbelt_s

seatbelt_s <- sapply(seatbelt_clean, max)
seatbelt_s <- round(seatbelt_s, 2)
seatbelt_s

## 5. Further drill down with aggregate functions.

# Compare columns state, year and miles.
seatbelt_agg <- select(seatbelt, c("state", "year", "miles"))
as_tibble(seatbelt_agg)

agg1 = seatbelt_agg %>% group_by(state) %>% 
  summarize(cnt_rows = n(), min_year = min(year), max_year = max(year), 
            Avg_miles = mean(miles)) %>% arrange(desc(Avg_miles))

as_tibble(agg1)

# Compare columns drinkage, year and miles.
seatbelt_agg <- select(seatbelt, c("drinkage", "year", "miles"))
agg1 = seatbelt_agg %>%  group_by(drinkage) %>% 
  summarize(cnt_rows = n(), min_year = min(year), max_year = max(year),
            Avg_miles = mean(miles))  %>% arrange(desc(Avg_miles))
as_tibble(agg1)
