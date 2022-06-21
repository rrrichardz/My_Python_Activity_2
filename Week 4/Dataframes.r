### Data frame in R
## 1. Create a list object in R
# Create a list object containing the numeric values
# 10, 20, 30 and 40:
a <- c (10, 20, 30, 40)

# Create another list object containing the string values
# of names for books:
b <- c ("R_for_Data_Science", "R_for_Dummies",
        "The_philosophy_of_R", "R_in_a_Nutshell")

# Create 3rd list with logical vectors to check stock:
c <- c (TRUE, FALSE, FALSE, TRUE)

# Create 4th list with price of the books:
d <- c (11.5, 18, 22.8, 15)

# Combines the 4 lists to create a data frame:
df <- data.frame (a, b, c, d)

# Print the data frame.
df

## 2. Customise a data frame
# Change names for columns.
names (df) <- c("ID", "books", "loan_status", "price")
df

# Examine the structure of df.
str (df)

## 3. Slice data from a data frame
# Extract the element in the 1st row of the 2nd column:
df [1, 2]

# Extract the 1st and 2nd row.
df [1:2, ]

df [1:2, 1:2]

# Extract the columns headed "ID" and "price":
df [, c ("ID", "price")]

## 4. Manipulate a data frame
# [1] Create a new object called "age".
age <- c (10, 35, 40, 5)
# [2] Add the new object to the data frame.
df$age <- age
# [3] Print the data frame.
df

# Place the $ operator between the data frame name
# and the column to extract:
df$ID

# Use subset() function to find specific values.
subset(df, ID == 10)
