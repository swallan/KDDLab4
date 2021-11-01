import sys
import pandas as pd
import numpy as np

def main(args):
    filename = args[1]
    k = int(args[2])
    D = parseFile(filename)
    alpha = 0.0003
    clusters, centroids = kMeans(D, k, alpha)
    printClusters(clusters, centroids)

def kMeans(D, k, alpha):
    # call relevant functions here
    centroids = selectInitialCentroids(D, k)
    s = {}  # stores the sum of points in each cluster
    num = {}  # stores the number of points in each cluster
    clusters = {}  # stores the points in each cluster
    while True:
        for j in range(k):
            s[j] = [0] * D.shape[1]  # stores the sum of the data points in each cluster
            num[j] = 0
            clusters[j] = []

        for d in D.itertuples(index=False, name=None):
            cl_dist = [(i, euclideanDistance(d, centroids[i])) for i in range(k)]
            cl = min(cl_dist, key=lambda t: t[1])[0]
            clusters[cl].append(d)
            s[cl] = [float(s[cl][i]) + float(d[i]) for i in range(D.shape[1])]
            num[cl] += 1

        centroids2 = centroidRecomputation(s, num, k)
        result = sumSquaredError(clusters, centroids, centroids2)
        if result < alpha:
            break
        centroids = centroids2

    return clusters, centroids

def selectInitialCentroids(D, k):
    # select initial centroids
    random = D.sample(n=k)
    return random.values.tolist()

def sumSquaredError(clusters, centroids1, centroids2):
    # compute sum of squared error for two points
    # insignificant decrease in the sum of squared errors
    SSE1 = 0
    SSE2 = 0
    for j in range(len(clusters)):
        sumCluster1 = sum([(euclideanDistance(clusters[j][i], centroids1[j]))**2 for i in range(len(clusters[j]))])
        SSE1 += sumCluster1

        sumCluster2 = sum([(euclideanDistance(clusters[j][i], centroids2[j]))**2 for i in range(len(clusters[j]))])
        SSE2 += sumCluster2

    return abs(SSE1 - SSE2)

def euclideanDistance(x, y):
    # compute distance between two points
    squares = [(float(x[i]) - float(y[i]))**2 for i in range(len(x))]
    dist = np.sqrt(sum(squares))
    return dist

def centroidRecomputation(s, num, k):
    # move centroids around
    newCentroids = [[s[j][i] / num[j] for i in range(len(s[j]))] for j in range(k)]
    return newCentroids

def sumSquaredClusterError(cluster, centroid):
    SSE = sum([(euclideanDistance(j, centroid))**2 for j in cluster])
    return SSE

def printClusters(clusters, centroids):
    for j in clusters:
        center = tuple(centroids[j])
        SSE = sumSquaredClusterError(clusters[j], center)
        print("Number of points in cluster %d: %d" % (j, len(clusters[j])))
        print("Center %d: %s" % (j, str(center)))
        dists = [euclideanDistance(i, center) for i in clusters[j]]
        maxDist = max(dists)
        minDist = min(dists)
        avgDist = sum(dists) / len(dists)
        print("Max Dist. to Center: ", maxDist)
        print("Min Dist. to Center: ", minDist)
        print("Avg Dist. to Center: ", avgDist)
        print("SSE: ", SSE)
        print("Points: ")
        for i in clusters[j]:
            print(i)
        print("----------------------------------------")

def parseFile(filename):
    df = pd.read_csv(filename, header=None)
    toDrop = [i for i in range(len(df.loc[0])) if df.loc[0][i] == '0']
    df = df.drop(toDrop, axis=1)
    df = df.drop(0)
    return df

if __name__ == '__main__':
    main(sys.argv)