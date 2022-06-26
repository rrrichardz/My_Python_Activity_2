### apply() function: Matrix

## 1. Create a matrix

# Name and build the matrix:
sales <- matrix(c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                  10, 30, 40, 50, 70, 85, 95, 110, 120, 125,
                  15, 20, 25, 35, 45, 55, 65, 70, 85, 95,
                  12, 20, 28, 35, 49, 60, 71, 80, 95, 105,
                  9, 15, 26, 38, 45, 59, 75, 85, 99, 110),
                nrow = 10, byrow = FALSE)

# Specify row names:
rownames(sales) <- c("w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8",
                    "w9", "w10")

# Specify column names:
colnames(sales) <- c("t1", "t2", "t3", "t4","t5")

# Print the matrix
sales

# Verify the data structure.
class(sales)

## 2. Interrogate the matrix

# Maximum sales for Week 1; specify data source, and set the margin for rows:
max(sales [1, ])

# Maximum sales per week across 10 weeks; specify data source, set the margin
# for rows, add max function:
apply(sales, 1, max)

# Calculate mean per store; specify data source, set the margin for columns,
# add mean function:
apply(sales, 2, mean)



### apply() function: Data frame

## 1. Build a new data frame

# Create/build a new data frame and assign it to sales:
sales.df <- data.frame(t1 = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
                      t2 = c(10, 30, 40, 50, 70, 85, 95, 110, 120, 125),
                      t3 = c(15, 20, 25, 35, 45, 55, 65, 70, 85, 95),
                      t4 = c(12, 20, 28, 35, 49, 60, 71, 80, 95, 105),
                      t5 = c(9, 15, 26, 38, 45, 59, 75, 85, 99, 110))

# Print the data frame.
sales.df

# Confirm creation of data frame.
class(sales.df)

## 2. Interrogate the data frame

# Find the mean sales per week; specify data source, set the margin for rows,
# add mean function:
apply(sales.df, 1, mean)

# Find the minimum, maximum, and mean sales per store across 10 weeks.
apply(sales.df, 2, min)
apply(sales.df, 2, max)
apply(sales.df, 2, mean)



### apply() function: Other options

## 1. Exclude columns

# Calculate mean sales for all stores except t2; specify data source and the
# excluded column, set the margin for columns, and add mean function (all
# columns):
apply(sales.df[, -2], 2,mean)

# Calculate mean cutting length for all stores except t2 and w10:
apply(sales.df[-10, -2], 2, mean)

## 2. Change the values

# Assign NA to row and column in sales.df.
sales.df[1, 5] <- NA

# Print the data frame.
sales.df

# Calculate the mean without the NA; specify data source, the margin for
# columns, the function (min), and indicate to ignore NA values:
apply(sales.df, 2, min, na.rm = TRUE)
