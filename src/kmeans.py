import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def main(args):
    filename = args[1]
    k = int(args[2])
    rowId = True
    D, rowIds, dropColumn = parseFile(filename, rowId)
    alpha = 0.0003
    clusters, centroids, rowObjs = kMeans(D, k, alpha, rowIds, dropColumn)
    printClusters(clusters, centroids, rowObjs)
    # print2dScatter(clusters)
    # print3dScatter(clusters)
    # validateClusters(rowObjs)

def kMeans(D, k, alpha, rowIds, dropColumn):
    centroids = selectInitialCentroids(D, k)
    s = {}  # stores the sum of points in each cluster
    num = {}  # stores the number of points in each cluster
    clusters = {}  # stores the points in each cluster
    rowObjs = {}
    while True:
        for j in range(k):
            s[j] = [0] * D.shape[1]  # stores the sum of the data points in each cluster
            num[j] = 0
            clusters[j] = []
            rowObjs[j] = []

        rowCt = -1
        for d in D.itertuples(index=False, name=None):
            cl_dist = [(i, euclideanDistance(d, centroids[i])) for i in range(k)]
            cl = min(cl_dist, key=lambda t: t[1])[0]
            clusters[cl].append(d)
            if rowIds is not None:
                rowCt += 1
                rowObj = rowIds.iloc[rowCt][dropColumn]
                rowObjs[cl].append(rowObj)
            s[cl] = [float(s[cl][i]) + float(d[i]) for i in range(D.shape[1])]
            num[cl] += 1

        centroids2 = centroidRecomputation(s, num, k)
        result = sumSquaredError(clusters, centroids, centroids2)
        if result < alpha:
            break
        centroids = centroids2

    if rowIds is None:
        rowObjs = None
    return clusters, centroids, rowObjs

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

def validateClusters(rowObjs):
    # calculates purity of each cluster
    for i in rowObjs:
        pluralityClass = max(set(rowObjs[i]), key=rowObjs[i].count)
        purity = rowObjs[i].count(pluralityClass) / len(rowObjs[i]) * 100
        print("Cluster %d purity = %.3f %%" % (i, purity))

def printClusters(clusters, centroids, rowObjs):
    clusterNums = ["cluster"]
    centers = ["center"]
    maxDists = ["max dist"]
    minDists = ["min dist"]
    avgDists = ["avg dist"]
    SSEs = ["sse"]
    for j in clusters:
        clusterNums.append(j)
        center = tuple(centroids[j])
        centers.append(tuple(map(lambda x: round(x, 3), center)))
        SSE = sumSquaredClusterError(clusters[j], center)
        SSEs.append(SSE)
        print("Number of points in cluster %d: %d" % (j, len(clusters[j])))
        print("Center %d: %s" % (j, str(center)))
        dists = [euclideanDistance(i, center) for i in clusters[j]]
        maxDist = round(max(dists), 3)
        maxDists.append(maxDist)
        minDist = round(min(dists), 3)
        minDists.append(minDist)
        if len(dists) != 0:
            avgDist = round(sum(dists) / len(dists), 3)
        else:
            avgDist = 0
        avgDists.append(avgDist)
        print("Max Dist. to Center: ", maxDist)
        print("Min Dist. to Center: ", minDist)
        print("Avg Dist. to Center: ", avgDist)
        print("SSE: ", SSE)
        print("Points: ")
        for i in range(len(clusters[j])):
            if rowObjs:
                print(rowObjs[j][i], end=" ")
            print(clusters[j][i])
        print("----------------------------------------")
    data = [clusterNums, centers, minDists, maxDists, avgDists, SSEs]
    print(
        tabulate(
            data,
            tablefmt="fancy_grid"
        )
    )

    sumDists = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            sumDists += euclideanDistance(centroids[i], centroids[j])

    total = len(centroids) * (len(centroids) - 1) / 2
    avgIntercluster = sumDists/total
    print("Avg Intercluster Dist: %.3f" % avgIntercluster)
    avgRadius = sum(maxDists[1:])/(len(maxDists) - 1)
    print("Ratio: %.3f" % (avgRadius / avgIntercluster))

def print2dScatter(clusters):
    idx = 0
    for i in clusters:
        colors = ['red', 'blue', 'green', 'black', 'purple', 'yellow', 'orange', 'cyan', 'magenta', 'pink', 'maroon', 'aquamarine', 'brown', 'olive']
        for j in clusters[i]:
            plt.scatter(j[0], j[1], color=colors[idx])
        idx += 1
    plt.show()

def print3dScatter(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    idx = 0
    for i in clusters:
        colors = ['red', 'blue', 'green', 'black', 'purple', 'yellow', 'orange', 'cyan', 'magenta', 'pink', 'maroon', 'aquamarine', 'brown', 'olive']
        for j in clusters[i]:
            ax.scatter(j[0], j[1], j[2], c=colors[idx])
        idx += 1
    plt.show()

def parseFile(filename, rowId):
    df = pd.read_csv(filename, header=None)
    toDrop = [i for i in range(len(df.loc[0])) if df.loc[0][i] == '0']
    df = df.drop(0)
    rowIds = None
    dropColumn = None
    if rowId and len(toDrop) == 1:
        rowIds = df[toDrop]
        dropColumn = toDrop[0]
    df = df.drop(toDrop, axis=1)
    return df, rowIds, dropColumn

if __name__ == '__main__':
    main(sys.argv)