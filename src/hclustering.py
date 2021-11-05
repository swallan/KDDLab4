import itertools
from tabulate import tabulate
import numpy as np
import scipy.stats as stats

p1, p2, height = 0, 1, 2


class Point:
    def __init__(self, *args):
        self.args = np.asarray(args, dtype=float)

    def __sub__(self, other):
        argsOut = []
        for i, a, in enumerate(self.args):
            argsOut.append(a - other.args[i])
        return np.asarray(argsOut)

    def __repr__(self):
        return f'Point({" ".join([str(x) for x in self.args])}'

class Hlustering:
    def __init__(self):
        self.methodDist = 'min'
        self.thresh = .5
        self.MAX_VAL = 1e11
        self.minClusters = 0

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

    def calculateDistances(self):
        # make a copy of the data, reshape into columns, then broadcast
        # into matrix of difference between each element.
        t = (self.dataAsPoints.reshape((self.dataAsPoints.shape[0], 1))
             - self.dataAsPoints)

        # calculate euclidean distance for each point.
        for ti in range(len(t)):
            for tx in range(len(t[ti])):
                t[ti, tx] = (np.sum(np.abs(t[ti, tx])**2))**.5

        self.distMatrix = np.asarray(t, dtype=float)

        # set the diagonal to be the max value so it is never selected as a
        # join in the cluster.
        for i in range(len(self.distMatrix)):
            for j in range(len(self.distMatrix[i])):
                if i == j:
                    self.distMatrix[i, j] = self.MAX_VAL
        return self.distMatrix


    def method(self, d1, d2):
        if self.methodDist == 'min':
            return min(d1, d2)
        if self.methodDist == 'max':
            return max(d1, d2)
        if self.methodDist == 'mean':
            return (d1+d2)/2
    
    def make_cluster(self, d, clusters, out_of_commission, labels, all_clusters):

        # termination condition reached, only one cluster remains.
        if len(all_clusters) == 1:
            return all_clusters

        # find the shortest distance between two clusters in the matrix.
        amin = np.amin(d)
        # find the index of this distance.
        where = np.where(d == amin)
        cluster_idx = where[0][0], where[1][0]

        # add to the 'labels' the new cluster and what height it was joined on.
        labels.append((labels[cluster_idx[0]], labels[cluster_idx[1]], amin))

        # add this new cluster (the joining of the two indices) to the clusters array.
        all_clusters.append((labels[cluster_idx[0]], labels[cluster_idx[1]], amin))

        # remove two clusters that were just joined, mark their indices as no
        # longer used.
        all_clusters.remove(labels[cluster_idx[0]])
        all_clusters.remove(labels[cluster_idx[1]])
        out_of_commission.append(cluster_idx[0])
        out_of_commission.append(cluster_idx[1])
    
        # add the index of this cluster to the clusters.
        clusters.append(cluster_idx)

        # for every new cluster, we add an identical row and column to the end
        # of the matrix representing the new distances.
        add_row = np.zeros(d.shape[0])    

        # calculate the new distances between existing clusters and the newly
        # formed cluster.
        for i, c in enumerate(clusters):
            if i < len(d):
                add_row[i] = self.method(d[i][cluster_idx[0]], d[i][cluster_idx[1]])

        # append the column to the matrix. (reshape st it is in column form)
        conc = np.concatenate((d, add_row.reshape(add_row.shape[0], 1)), axis=1)
        # append the row to the matrix
        row_w_z = np.concatenate((add_row, [self.MAX_VAL]))
        res = np.append(conc, [row_w_z]).reshape(row_w_z.shape[0], row_w_z.shape[0])

        # mark all indices belonging to the previous two clusters as the max
        # val
        # so that they are not re-selected.
        for ii in range(len(res)):
            for jj in range(len(res[ii])):
                if ii == cluster_idx[0] or jj == cluster_idx[1]:
                    res[ii, jj] = self.MAX_VAL
                    res[jj, ii] = self.MAX_VAL

        # make recursive call to again create another cluster.
        return self.make_cluster(res, clusters, out_of_commission, labels,
                                 all_clusters)
                    
 

    def _form_jsonH(self, tuples):
        
        if type(tuples) == int:
            return {
                'type': 'leaf',
                'data': f"{self.rawData[tuples][0]:.02f},{self.rawData[tuples][1]:.02f} ",
                'height': 0
                }
        else:
            return {
                'type': 'node',
                'data': [self._form_jsonH(tuples[p1]), self._form_jsonH(tuples[p2])],
                'height': tuples[height]
                }
        
    def form_json(self, tuples):
        d = dict()
        d['type'] = 'root'
        d['height'] = tuples[height]
        d['nodes'] = [self._form_jsonH(tuples[p1]), self._form_jsonH(tuples[p2])]
        return d
    
    def get_all_points_from_cluster(self, cluster, s):
        if len(cluster) == 0:
            return s
        cpy=[]
        for c in cluster:
            if type(c) == int:
                s.add(c)
            else:
                # print(type(c))
                cpy.append(c[0])
                cpy.append(c[1])
        return self.get_all_points_from_cluster(cpy, s)
    
    
    def calculate_center(self, pts):
        points = self.rawData[pts]
        center = np.mean(points, axis=0)
        return center
        
    
    def get_clusters_thresh(self, clst, threshold):
        cpy = []
        for c in clst:
            # one of the clusters is just a point
            if type(c) == int:
                cpy.append(c)
            else:
                # if the distance is still over the threshold,
                # keep splitting
                if c[2] > threshold:
                    cpy.append(c[0])
                    cpy.append(c[1])
                else:
                    cpy.append(c)
                    
        should_return = set([(x[2] <= threshold if not (type(x) == int) else True) for x in cpy])
        if should_return == set([True]) and len(should_return) == 1:
            return cpy
        else:
            return self.get_clusters_thresh(cpy, threshold)

import sys
if __name__ == '__main__':
    print('hclustering')
    cluster = Hlustering()
    cluster.parseData(f"{sys.argv[1]}")
    d = cluster.calculateDistances()
    d = np.asarray(d)
    cluster.minClusters = 0
    cluster.methodDist = 'mean'
    labels = list(range(len(d)))
    result = cluster.make_cluster(d, list(range(len(d))), [], labels, labels[:])

    OUT = ""

    root = result[0]
    import json
    OUT += f"DATASET: {sys.argv[1]}"
    OUT += json.dumps(cluster.form_json(root))

    # if a threshold was provided.
    if len(sys.argv) == 3:

        OUT += f"\n thresh: {sys.argv[2]}\n"
        final_clusters = cluster.get_clusters_thresh(result, float(sys.argv[2]))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)

        enumerates = ["cluster"]
        centers = ["center"]
        mind = ["min dist"]
        maxd = ["max dist"]
        means = ["mean dist"]
        sse = ['SSE']

        # for accidents 3d
        if cluster.non_normalized.shape[1] == 3:
            ax = plt.axes(projection='3d')
        colors = itertools.cycle(['darkorange', 'lime', 'darkblue', 'hotpink', 'red', 'gold', 'indigo', 'dodgerblue', 'wheat'])

        for i, c in enumerate(final_clusters):


            pts = cluster.get_all_points_from_cluster([c], set())
            datas = cluster.rawData[list(pts)]
            OUT += '\n'
            OUT += f"Data from cluster {i}: \n{cluster.dataWithClass[list(pts)]}"
            center = cluster.calculate_center(list(pts))
            distances = np.sum(np.abs(datas - center), axis=1)
            enumerates.append(i)
            centers.append([round(c, ndigits=2) for c in center])
            mind.append(round(np.min(distances), ndigits=2))
            maxd.append(round(np.max(distances), ndigits=2))
            means.append(round(np.mean(distances), ndigits=2))
            sse.append(np.sum(np.sum(np.abs(datas - center) ** 2, axis=1)))

            mode = stats.mode(cluster.dataWithClass[list(pts)][..., -1])[0][0]
            OUT += f"\nMost common classifier: {mode[:-2]}: this cluster is {len(np.where(cluster.dataWithClass[list(pts)][..., -1] == mode)[0]) / len((cluster.dataWithClass[list(pts)][..., -1])) * 100:.02f}% {mode[:-2]}"

            if cluster.non_normalized.shape[1] == 2:
                nonNonrmal_data = cluster.non_normalized[list(pts)]
                ax.scatter(nonNonrmal_data[..., 0], nonNonrmal_data[..., 1], c=next(colors))
                ax.annotate(f"c{i}", np.mean(nonNonrmal_data, axis=0), size=20, weight='bold',)
            if cluster.non_normalized.shape[1] == 3:
                nonNonrmal_data = cluster.non_normalized[list(pts)]
                ax.scatter(nonNonrmal_data[..., 0], nonNonrmal_data[..., 1], nonNonrmal_data[..., 2], c=next(colors))

        if cluster.non_normalized.shape[1] <= 3:
            plt.title(f"{sys.argv[1].split('/')[-1]} hcluster t={sys.argv[2]}")
            plt.savefig(f"out/hclust_{sys.argv[1].split('/')[-1]}.jpg")
            plt.show()



        OUT += '\n'
        OUT += tabulate([enumerates, centers, mind, maxd, means, sse], tablefmt="fancy_grid")
    OUT = OUT.replace("\\n", "")
    print(OUT)
    with open(f"out/hclust_{sys.argv[1].split('/')[-1]}.out", 'w') as f:
        f.write(OUT)




    # # for datasets that can be graphed
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # for x in result:
    #     print("new cluster")
    #     # fig, ax = plt.subplots(i, i)
    #     x = str(x).replace("(", "").replace(")", "").replace(",", "").replace("[", "").replace("]", "").split()
    #     x = [int(xi) for xi in x]
    #     curData = cluster.rawData[x]
    #     ax.scatter(curData[...,0], curData[...,1], curData[...,2] )

                
                
