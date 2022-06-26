##############################################################################

# Data manipulation with R

# Family = principle member + dependents (up to 4)
# Single = only 1 principle member

# Method for data analysis
# (i) subset travel mode into 4 tables based on mode of transport: air, train, bus and car

#############################################################################
# Install libraries and import data set

install.packages("tidyverse")

library(tidyverse)

# Import your csv 
travelmode <- read.csv(file.choose(), header = TRUE)

# Review/sense-check the data set
as_tibble (travelmode)
str(travelmode)

# Delete column X and gender
travelmode <- subset (travelmode, select = -c(X, gender))
# View the column names
names (travelmode)

# Change the names of the columns
travelmode <- travelmode %>%
  rename(waiting_time = wait, 
         vehicle_cost = vcost, 
         travel_time = travel, 
         general_cost = gcost, 
         family_size = size)

head(travelmode)

# Finds total costs for car clients only in a data frame

car_costs <- subset(travelmode, mode == 'car')

# Add a column with total costs
car_costs <- car_costs %>%
  mutate(total_cost = vehicle_cost + general_cost) 

head(car_costs)

# Add total_cost column to travelmode df
joined_travelmode <- left_join(travelmode, car_costs)

head(joined_travelmode)

# Confirm number of rows
dim(joined_travelmode)

# Subset using the filter() and select() functions from the dplyr package

# create new data frame as travelmode_air
air_family <- select (filter(travelmode, mode == 'air',
                                         family_size >= '2'), 
                          c(individual, choice:family_size))
head(air_family)

# subset for train, bus and car as well
train_family <- select (filter (travelmode, mode == 'train',
                             family_size >= '2'), 
                            c(individual, choice:family_size))
head(train_family)

bus_family <- select (filter (travelmode, mode == 'bus',
                              family_size >= '2'), 
                          individual, choice:family_size)
head(bus_family)


car_family <- select (filter (travelmode, mode == 'car',
                              family_size >= '2'), 
                          individual, choice:family_size)
head(car_family)

# Find preferred modes of travel
air_family %>% 
  count(choice)

train_family %>% 
  count(choice)

bus_family %>% 
  count(choice)

car_family %>% 
  count(choice)

# Average vehicle cost and general for family's in cars, trains, and buses.
mean_car_costs <- summarise(car_family, mean_VC = mean(vehicle_cost),
          mean_GC = mean(general_cost))
mean_car_costs

mean_bus_costs <- summarise(bus_family, mean_VC = mean(vehicle_cost),
                           mean_GC = mean(general_cost))
mean_bus_costs

mean_train_costs <- summarise(train_family, mean_VC = mean(vehicle_cost),
                              mean_GC = mean(general_cost))
mean_train_costs

# Create a data frame to hold these values
mean_costs <- rbind(mean_car_costs, mean_bus_costs,
                         mean_train_costs)
mean_costs

# Add a column containing the vehicle type
vehicle <- c('car', 'bus', 'train')
mean_costs <- cbind(vehicle, mean_costs)
mean_costs

# Set in descending order from the highest mean general cost down
arrange(mean_costs, desc(mean_GC))
