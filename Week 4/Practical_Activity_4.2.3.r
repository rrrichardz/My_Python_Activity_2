### 1. Prepare your workstation

# Determine working directory
getwd()

### 2. Import the data set

# Import a CSV file.
data <- read.csv(file.choose(), header = T)

# Print the data frame.
data
View(data)

# Sense-check the data.
str(data)
dim(data)
typeof(data)
class(data)

### 3. Export the data set

# Export CSV file
write_csv (data, file = "data_output/Practical_Activity_4.2.3_data.csv")


