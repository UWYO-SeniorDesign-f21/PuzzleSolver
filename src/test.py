import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster

# generate 3 clusters of each around 100 points and one orphan point
N=100
data = numpy.random.randn(10*N,2)
for i in range(10):
    data[N*i : N*(i+1)] += 50*i

# clustering
thresh = 1.5
clusters = hcluster.fclusterdata(data, thresh, criterion="distance")

centers = []
for x in numpy.unique(clusters):
    center = numpy.mean(data[numpy.where(clusters == x)], axis=0)
    centers.append(center)

centers = numpy.array(centers)
ids = numpy.argsort(centers[:,0])
lookup = numpy.zeros_like(ids)
lookup[ids] = numpy.arange(len(centers))
print(lookup)
clusters = lookup[clusters - 1]

# plotting
plt.scatter(*numpy.transpose(data), c=clusters, cmap=plt.cm.rainbow)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()