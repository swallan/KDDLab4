from matplotlib import pyplot as plt

from hclustering import Point
import numpy as np

class DBSCAN:
    MAX_VAL = 10
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
                    dataAsPoints.append(p)
                    dataWithoutClassifier.append((np.asarray(di)[restrictions]))
                    rawData.append(di)
        self.restrictions = restrictions
        self.dataWithClass = np.asarray(rawData, dtype=object)
        self.rawData = np.asarray(dataWithoutClassifier, dtype=float)
        self.non_normalized = self.rawData
        # self.rawData = (self.rawData - np.min(self.rawData, axis=0)) / (np.max(self.rawData, axis=0) - np.min(self.rawData, axis=0))
        self.dataAsPoints = np.asarray(dataAsPoints)

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
        labels = list(range(len(self.distMatrix)))
        self.boundryPoints = set()
        self.corepoints = set()
        for i in labels:
            dists = np.delete(dbscan.distMatrix[i], i)
            good_neighbors = dists[dists < self.epsilon]
            if len(good_neighbors) >= self.minPts:
                self.corepoints.add(i)
            for n in good_neighbors:
                self.boundryPoints.add(n)
        self.boundryPoints = self.boundryPoints - self.corepoints
        self.noise = set(labels) - (self.corepoints | self.boundryPoints)
        
    def _form_clusters_h(self, clusters, corePoints):
        # if len(corePoints) == 0:
        #     return clusters

        
        while(len(corePoints) > 0):
            centroid = corePoints.pop()
            newC = set([centroid])
            clusters.append(newC)
            pointsToCheck = [centroid]
            while len(pointsToCheck) > 0:
                currPoint = pointsToCheck.pop(0)
                currRow = self.distMatrix[currPoint]
                add_idc = np.where(currRow < self.epsilon)[0]
                for i in add_idc:
                    if i in self.corepoints and i not in newC:
                        pointsToCheck.append(i)
                        newC.add(i)
                    if i in self.boundryPoints:
                        newC.add(i)
            corePoints = corePoints - newC
        return clusters
    
    
    def form_clusters(self):
        clusters = []
        return self._form_clusters_h(clusters, self.corepoints)
        
        


if __name__ == '__main__':

    # from sklearn.datasets import make_circles
    #
    # # moons_X: Data, moon_y: Labels
    # np.random.seed(32142135)
    # moons_X, moon_y = make_circles(n_samples=15000, noise=.05, factor=.5)
    #
    # with open("../data/ringData.csv", "w+") as f:
    #     f.write("1,1\n")
    #     for i, j in moons_X:
    #         f.write(f"{i:.04f},{j:.04f}\n")


    dbscan = DBSCAN()
    
    dbscan.parseData("../data/moonData.csv")
    dbscan.calculateDistances()
    dbscan.minPts = 6
    dbscan.epsilon = .3
    dbscan.decide_groupings()
    print(dbscan.noise, dbscan.corepoints, dbscan.boundryPoints)
    clusters = dbscan.form_clusters()
    print(clusters)
    all_categorized = set()
    for c in clusters:
        for ci in c:
            all_categorized.add(ci)
    
    all_points = set(list(range(len(dbscan.rawData))))
    
    uncategorized_points = all_points - all_categorized  
    #%%
    cdata = dbscan.rawData[list(uncategorized_points)]
    plt.scatter(cdata[...,0], cdata[...,1], c='black' ) 
    for c in clusters:
        cdata = dbscan.rawData[list(c)]
        plt.scatter(cdata[...,0], cdata[...,1] ) 
    #%%
    plt.show()
    
    # plt.scatter(dbscan.rawData[...,0], dbscan.rawData[...,1] ) 
    # for i in range(len(dbscan.rawData[...,0])):
    #     plt.annotate(str(i), (dbscan.rawData[...,0][i], dbscan.rawData[...,1][i]))
    
    #%%

    print('dbscan')