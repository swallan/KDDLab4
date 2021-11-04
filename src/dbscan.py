from matplotlib import pyplot as plt
from tabulate import tabulate

from hclustering import Point
import numpy as np

class DBSCAN:
    MAX_VAL = 10

    def __init__(self):
        self.epsilon = None

    def parseData(self, fileName: str):
        """
        Takes in filename and parses non-classifier data into numpy array
        of data.
        :param fileName:
        """
        self.fileName = fileName
        dataWithoutClassifier = []
        rawData = []
        dataAsPoints = []
        with open(fileName, "r") as f:
            lines = f.readlines()
            restrictions = np.asarray(np.asarray(lines[0].strip().split(','), dtype=int), dtype=bool)
            for li in lines[1:300]:
                if li.strip() != "":
                    di = li.split(',')
                    p = Point(*(np.asarray(di)[restrictions]))
                    dataWithoutClassifier.append((np.asarray(di)[restrictions]))
                    rawData.append(di)
        self.restrictions = restrictions
        self.dataWithClass = np.asarray(rawData, dtype=object)
        self.rawData = np.asarray(dataWithoutClassifier, dtype=float)
        self.non_normalized = self.rawData
        self.rawData = (self.rawData - np.min(self.rawData, axis=0)) / (np.max(self.rawData, axis=0) - np.min(self.rawData, axis=0))
        self.dataAsPoints = np.asarray([Point(d) for d in self.rawData])
        # self.dataAsPoints = np.asarray(dataAsPoints)

    def calculateDistances(self):
        # resultMatrix = np.zeros((pointList.shape[0], pointList.shape[0]))
        t = (self.dataAsPoints.reshape((self.dataAsPoints.shape[0], 1))
                           - self.dataAsPoints)
        for ti in range(len(t)):
            for tx in range(len(t[ti])):
                t[ti, tx] = np.sum(np.abs(t[ti, tx]))

        self.distMatrix = np.asarray(t, dtype=float)
        for i in range(len(self.distMatrix)):
            for j in range(len(self.distMatrix[i])):
                if i == j:
                    self.distMatrix[i, j] = self.MAX_VAL
        return self.distMatrix

    def decide_groupings(self):
        # create a set of labels for the data (0 - n)
        labels = list(range(len(self.distMatrix)))
        # store boundry and core points
        self.boundryPoints = set()
        self.corepoints = set()
        # for each datapoint
        for i in labels:
            # select the distances to all other points
            dists = dbscan.distMatrix[i]
            # select the neighbors that are within distance of epsilon
            good_neighbors = dists[dists < self.epsilon]
            # if the number of neighbors within reach exceeds minPts,
            # designate it as a core point.
            if len(good_neighbors) >= self.minPts:
                self.corepoints.add(i)
                # add every neighbor within reach to the boundry points set.
                # core points will be removed later.
                for ni in np.where(dists < self.epsilon)[0]:
                    self.boundryPoints.add(ni)
        # remove core points from boundry point set
        self.boundryPoints -= self.corepoints
        # set noise points to all points minus (core + boundry)
        self.noise = set(labels) - (self.corepoints | self.boundryPoints)
        
    def form_clusters(self):
        clusters = []
        corePoints = self.corepoints
        # while there are still core points,
        while len(corePoints) > 0:
            # establish the top core point as the beginning of the cluster.
            centroid = corePoints.pop()
            # create set with this point as the cluster
            newC = {centroid}
            # add it to the master list of clusters.
            clusters.append(newC)
            # store a temporary variable for points to check when extending the
            # cluster. start with the current only point in the cluster.
            pointsToCheck = [centroid]
            while len(pointsToCheck) > 0:
                # pull out top point.
                currPoint = pointsToCheck.pop(0)
                # grab distance to all other points for this point
                currRow = self.distMatrix[currPoint]
                # find indices where the current point has close neighbors
                add_idc = np.where(currRow < self.epsilon)[0]
                for i in add_idc:
                    # for each nearby neighbor, verify that it is a core point
                    # and that it is not already in the cluster
                    if i in self.corepoints and i not in newC:
                        # if this satisfies those conditions, add it to the
                        # cluster and add it to the points to check neighbors
                        # for
                        pointsToCheck.append(i)
                        newC.add(i)
                    # if it is just a boundry point, add it to the cluster.
                    if i in self.boundryPoints:
                        newC.add(i)
            # remove the core points from the current cluster from the master
            # set of core points.
            corePoints -= newC
        return clusters
    
    

import sys
if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    OUT = ""
    dbscan = DBSCAN()
    # fp = "../data/iris.csv"
    # fn = fp
    fn = sys.argv[1].split("/")[-1]
    dbscan.parseData(sys.argv[1])
    dbscan.calculateDistances()
    dbscan.minPts = int(sys.argv[2])
    dbscan.epsilon = float(sys.argv[3])
    dbscan.decide_groupings()
    clusters = dbscan.form_clusters()
    all_categorized = set()
    OUT += f"""
dbscan for {sys.argv[1]} with:
- minPts: {sys.argv[2]}
- epsilon: {sys.argv[3]}
    """

    enumerates = ["cluster"]
    centers = ["center"]
    mind = ["min dist"]
    maxd = ["max dist"]
    means = ["mean dist"]
    sse = ['mean square error']

    for i, c in enumerate(clusters):
        OUT += f'\nCluster {i}:\n'
        OUT += str(dbscan.dataWithClass[list(c)])
        data = dbscan.rawData[list(c)]
        center = np.mean(data, axis=0)
        distances = np.sum(np.abs(data - center), axis=1)
        enumerates.append(i)
        centers.append([round(c, ndigits=2) for c in center])
        mind.append(round(np.min(distances), ndigits=2))
        maxd.append(round(np.max(distances), ndigits=2))
        means.append(round(np.mean(distances), ndigits=2))
        sse.append(np.mean(np.sum(np.abs(data - center) ** 2, axis=1) ** .5))

        for ci in c:
            all_categorized.add(ci)
    OUT += '\n'
    OUT += tabulate([enumerates, centers, mind, maxd, means, sse], tablefmt="fancy_grid")
    all_points = set(list(range(len(dbscan.rawData))))

    uncategorized_points = all_points - all_categorized
    OUT += f"\n{len(uncategorized_points)} datapoints were found to be noise out of {len(all_points)}"
    if dbscan.rawData.shape[1] <=3:
        if dbscan.non_normalized.shape[1] == 3:
            ax = plt.axes(projection='3d')

        #%%
        plt.title(f"{fn}, minPts = {dbscan.minPts}\ndbscan.epsilon = {dbscan.epsilon}")
        l = list(uncategorized_points)
        cdata = dbscan.non_normalized[l]
        if dbscan.rawData.shape[1] == 2:
            ax.scatter(cdata[..., 0], cdata[..., 1], c='black', marker="X", alpha=.5)
            # for i in l:
            #     plt.annotate(f"c{i}", dbscan.non_normalized[i],
            #                  size=7)

        else:

            ax.scatter(cdata[..., 0], cdata[..., 1], cdata[..., 2], c='black', marker="X",
                        alpha=.5)

        colors = iter(['darkorange', 'lime', 'darkblue', 'hotpink', 'red', 'gold', 'indigo', 'dodgerblue', 'wheat'])

        for c in clusters:
            colour = next(colors)
            boundry = [ci for ci in c if ci in dbscan.boundryPoints]
            corp = list(c - set(boundry))
            corePoints = dbscan.non_normalized[corp]

            boundryPoints = dbscan.non_normalized[list(boundry)]

            if dbscan.rawData.shape[1] == 2:
                ax.scatter(corePoints[..., 0], corePoints[..., 1], c=colour,
                            alpha=.5)
                ax.scatter(boundryPoints[..., 0], boundryPoints[..., 1],
                            marker="X", c=colour)

                # for i in corp:
                #     plt.annotate(f"c{i}", dbscan.non_normalized[i],
                #                 size=7, c=colour)

            else:
                ax.scatter(corePoints[..., 0], corePoints[..., 1], corePoints[..., 2], c=colour,
                            alpha=.5)
                ax.scatter(boundryPoints[..., 0], boundryPoints[..., 1], boundryPoints[..., 2],
                            marker="X", c=colour)



        if dbscan.non_normalized.shape[1] <= 3:
            plt.title(f"{sys.argv[1].split('/')[-1]} dbscan minPts={sys.argv[2]} epsilon={sys.argv[3]}")
            plt.savefig(f"out/dbscan_{fn}.jpg")
            plt.show()


    OUT = OUT.replace("\\n", "")
    with open(f"out/dbscan_{fn}.out", 'w') as f:
        f.write(OUT)
    print(OUT)
