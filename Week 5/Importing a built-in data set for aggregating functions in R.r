### 1. Import and sense-check the data

# Import data set.
data("CO2")

# View the data set.
View(CO2)

# Dimensions of the data set.
dim(CO2)

### 2. Determine the mean CO2 uptake (one subset)

# Specify the function as aggregate(), and specify the numeric variable (uptake),
# the grouping variable (conc), the data source (as CO2), and the mean:
aggregate(uptake ~ conc, CO2, mean)

# Determine the sum and standard deviation.
aggregate(uptake ~ conc, CO2, sum)

aggregate(uptake ~ conc, CO2, sd)

### 3. Determine the mean CO2 uptake (multiple subsets)

# Specify the function as aggregate(), and specify the numeric variable (uptake),
# the grouping variable (conc), the additional grouping variable (treatment),
# the data source (as CO2), and the mean:
aggregate(uptake ~ conc + Treatment, CO2, mean)
