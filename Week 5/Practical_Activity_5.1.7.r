### Practical Activity 5.1.7

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
# Useful for visualisation.
library(ggplot2)

# Import data file.
police <- read.csv(file.choose(), header = T)

# Sense-check the data set.
as_tibble(police)
View(police)
dim(police)

## 2. Understand the data 

# Determine missing values.
police[is.na(police)] 
sum(is.na(police))

# Delete all missing values' rows.
police_new <- na.omit(police)
head(police_new)

dim(police_new)
sum(is.na(police_new))

# Compile a data profile report.
summary(police_new)
DataExplorer::create_report(police_new)

## 3. Data manipulation

# Drop unnecessary columns.
police_df <- select(police_new, -c('X', 'idNum', 'date', 'MDC', 'preRace',
                                   'race', 'lat', 'long', 'policePrecinct',
                                   'citationIssued', 'personSearch', 
                                   'vehicleSearch'))

colnames(police_df)
dim(police_df)

# Rename the column names with first letter to uppercase.
names(police_df) <- str_to_title(names(police_df))
colnames(police_df)

head(police_df)

# Determine the unique values in each column.
unique(police_df$Problem)
unique(police_df$Gender)
unique(police_df$Neighborhood)

## 4. Visualise data

# Visualise the neighbourhood with the most reported cases.
barplot(table(police_df$Neighborhood),
        main = "Police reports",
        xlab = "Neighbourhood",
        ylab = "Count",
        col = "green")

# Visualise the police reports: suspicious or traffic offences.
barplot(table(police_df$Problem),
main = "Police reports",
xlab = "Offense",
ylab = "Count",
col = "red")

# Visualise the gender ratio of traffic offences.
barplot(table(police_df$Gender),
        main = "Police reports",
        xlab = "Gender",
        ylab = "Count",
        col = "blue")

# Determine the number of offences for gender and types of problems.
table(police_df$Gender)            
table(police_df$Problem)
table(police_df$Neighborhood)

# Determine only females with traffic offences.
nrow(subset(police_df, Gender == "Female" & Problem == "traffic"))

# Determine only males with traffic offences.
nrow(subset (police_df, Gender == "Male" & Problem == "traffic"))

# Determine the neighbourhood with the most problems.
police_df %>% count(Neighborhood, sort = TRUE)
