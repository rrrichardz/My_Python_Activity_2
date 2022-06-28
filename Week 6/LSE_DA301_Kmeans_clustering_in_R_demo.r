#############################################################################

## K-means Clustering In R


############################################################################

## First install the packages and libraries necessary for k-means 
install.packages('factoextra')

library(dplyr)
library(cluster)
library(factoextra)

## Import the CSV file
df_fb <- read.csv(file.choose(), header = TRUE)

## View the top six rows
head(df_fb)

## View summary statistics, make note of the nature of the first few columns
summary(df_fb)

## Remove the ID columns
df_fb_nec <- dplyr::select(df_fb, -c(1:3))

## Confirm column names
colnames(df_fb_nec)

## Remove NA values as these will cause errors in the model
df_fb_nec = na.omit(df_fb_nec)

## Scale the variables for acccurace
df_fb_nec = as.matrix(scale(df_fb_nec))

## Select the number of clusters (k)
fviz_nbclust(df_fb_nec, kmeans, method = "wss")

## Set seed to one so that we can reproduce these results
set.seed(1)

## Create the K-means model
model = kmeans(df_fb_nec, 4, nstart = 25)

## View the output
print(model)

## Create a plot to show the clusters
fviz_cluster(model, data = df_fb_nec)

## View the mean values of each cluster
aggregate(df_fb, by = list(cluster = model$cluster), mean)

