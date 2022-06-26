### lapply() function: List

## 1. Create a new list

# Assign a new list and create a new list:
sales.number <- list(t1 = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                     t2 = c(10, 30, 40, 50, 70, 85, 95, 110, 120, 125),
                     t3 = c(15, 20, 25, 35, 45, 55, 65, 70, 85, 95),
                     t4 = c(12, 20, 28, 35, 49, 60, 71, 80, 95, 105),
                     t5 = c(9, 15, 26, 38, 45, 59, 75, 85, 99, 110))

# Print the new list.
sales.number

# Confirm list creation.
class(sales.number)

## 2. Interrogate the list

# To find mean sales; specify data list, the function (mean), and determine
# the object class:
lapply(sales.number, mean)



### lapply() function: Data frame

## 1. Create a new data frame

# Assign a new data frame to the object and create a data frame:
computers.df <- data.frame(t1 = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                           t2 = c(10, 30, 40, 50, 70, 85, 95, 110, 120, 125),
                           t3 = c(15, 20, 25, 35, 45, 55, 65, 70, 85, 95),
                           t4 = c(12, 20, 28, 35, 49, 60, 71, 80, 95, 105),
                           t5 = c(9, 15, 26, 38, 45, 59, 75, 85, 99, 110))

# Print the new data frame.
computers.df

# Check data frame.
class(computers.df)

## 2. Interrogate the data frame

# Calculate the sd for each store (object as data frame, function as sd):
lapply(computers.df, sd)
