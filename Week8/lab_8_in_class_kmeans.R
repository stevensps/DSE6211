## set the random number seed to obtain reproducible results
set.seed(1234)

## load any libraries that will be used
library(NbClust)

## view the data
data(wine, package = "rattle")
head(wine)

## exclude the first column, since it contains the actual wine type
## we are going to try to identify the wine types using k-means clustering
## scale each of the variables to have a mean of zero and a standard deviation of one
wine_scaled <- scale(wine[, -1])
head(wine_scaled)

## use the NBClust function to search for the best number of clusters (between 2 and 15) for the kmeans algorithm
nc <- NbClust(wine_scaled, min.nc = 2, max.nc = 15, method = "kmeans")

barplot(table(nc$Best.n[1, ]), xlab = "Number of Clusters", ylab = "Number of Criteria",
        main = "Number of Clusters Chosen by 26 Criteria")

fit.km <- kmeans(wine_scaled, centers = 3, nstart = 25)

## print the cluster sizes and cluster centers in a centroid table
fit.km$size
fit.km$centers

## create a data frame that contains the original data, as well as the cluster number for each observation
cluster_number <- data.frame(cluster_number = fit.km$cluster)
wine_clusters <- cbind(wine, cluster_number)

## explore the clusters to see if we can classify them based on the mean value for each variable within each cluster
aggregate(wine_clusters[, -1], by=list(wine_clusters$cluster_number), mean)

## compare the actual wine types with the clusters we found in the data - the clusters look very good!
ct.km <- table(wine$Type, fit.km$cluster)
ct.km