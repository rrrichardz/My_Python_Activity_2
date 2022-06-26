############################################################################

# Updated packages

install.packages("tidyverse")
install.packages("skimr")
install.packages("DataExplorer") 

############################################################################

# Install the libraries
library(tidyverse)

library(readr)

library(dplyr)

library(tidyr)

library(skimr)

library(DataExplorer)

# Import your csv 
travelmode <- read.csv(file.choose(), header = TRUE)

# View the first six lines of the data frame
head(travelmode)

# View the last six lines of the data frame
tail(travelmode)

# View the data set in a new window in RStudio as an Excel-style sheet
View(travelmode)

# View the dimensions of the data set i.e. the number of rows and columns
dim(travelmode)

# View the titles or names of the columns in the data set
colnames(travelmode)

# These functions provide different views of the data set:
# str()
# glimpse()
# as_tibble(

str(travelmode)

glimpse(travelmode)

as_tibble(travelmode)

# To search for missing values in a data set
travelmode[is.na(travelmode)]

# To search for missing values in a specific column of a data set
is.na(travelmode$size)

# To search for missing values in a data set
sum(is.na(travelmode))

# To search for missing values in a specific column of a data set
sum(is.na(travelmode$size))

# These functions provide summary statistics of the data set
# summary()
# skim()
# DataExplorer()

summary(travelmode)

skim(travelmode)

# This creates a downloadable HTML file containing summary stats of the data set
DataExplorer::create_report(travelmode)
