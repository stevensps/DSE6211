## set the random number seed to obtain reproducible results
set.seed(1234)

## load any libraries that will be used
library(NbClust)

## view the data
data(nutrient, package = "flexclust")
head(nutrient)

## scale each of the variables to have a mean of zero and a standard deviation of one
nutrient_scaled <- scale(nutrient)
head(nutrient_scaled)

## use the NBClust function to search for the best number of clusters (between 2 and 15) for the hierarchical algorithm
nc <- NbClust(nutrient_scaled, distance = "euclidean", min.nc = 2, max.nc = 15, method = "average")

barplot(table(nc$Best.n[1, ]), xlab = "Number of Clusters", ylab = "Number of Criteria",
        main = "Number of Clusters Chosen by 26 Criteria")

## apply the hierarchical clustering algorithm
d <- dist(nutrient_scaled)
fit.hc <- hclust(d, method = "average")
plot(fit.hc, hang = -1, cex = 0.8, main = "Average Linkage Clustering")

## cut the dendrogram to get the number of clusters selected using the NbClust function
nutrient_clusters <- cutree(fit.hc, k = 5)
nutrient_clusters

## explore the clusters to see if we can classify them based on the mean value for each variable within each cluster
aggregate(nutrient, by=list(nutrient_clusters), mean)
