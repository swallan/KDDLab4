import numpy as np
MAX_VAL = 1e11
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
        pass

    def parseData(self, fileName: str):
        """
        Takes in filename and parses non-classifier data into numpy array
        of data.
        :param fileName:
        """
        self.fileName = fileName
        dataWithoutClassifier = []
        rawData = []
        restrictions = []
        dataAsPoints = []
        with open(fileName, "r") as f:
            lines = f.readlines()
            restrictions = np.asarray(np.asarray(lines[0].strip().split(','), dtype=int), dtype=bool)
            for li in lines[1:]:
                if li.strip() != "":
                    di = li.split(',')
                    p = Point(*(np.asarray(di)[restrictions]))
                    dataAsPoints.append(p)
                    dataWithoutClassifier.append((np.asarray(di)[restrictions]))
                    rawData.append(di)
        self.restrictions = restrictions
        self.rawData = np.asarray(dataWithoutClassifier, dtype=float)
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
                    self.distMatrix[i, j] = MAX_VAL
        return self.distMatrix


def method(d1, d2, method='min'):
    if method == 'min':
        return min(d1, d2)
    if method == 'max':
        return max(d1, d2)
    if method=='mean': 
        return (d1+d2)/2

def make_cluster(d, clusters, out_of_commission, labels, all_clusters):

    if len(all_clusters) == 1:
        return all_clusters
    
    amin = np.amin(d)
    
    s = set()
    for i in all_clusters:
        s.add(type(i))
        
    if len(all_clusters) == 4:
            return all_clusters
    
    # if amin > 7:
    #     return all_clusters
    where = np.where(d==amin)
    cluster_idx = where[0][0], where[1][0]
    labels.append((labels[cluster_idx[0]], labels[cluster_idx[1]]))
    all_clusters.append((labels[cluster_idx[0]], labels[cluster_idx[1]]))
    all_clusters.remove(labels[cluster_idx[0]])
    all_clusters.remove(labels[cluster_idx[1]])
    # print(f"merging {labels[cluster_idx[0]]} and {labels[cluster_idx[1]]}")
    out_of_commission.append(cluster_idx[0])
    out_of_commission.append(cluster_idx[1])

    clusters.append(cluster_idx)

    add_row = np.zeros(d.shape[0])    
    
    for i, c in enumerate(clusters):
        if i < len(d):
            # print(r)
            r = d[i]
            add_row[i] = method(d[i][cluster_idx[0]], d[i][cluster_idx[1]],
                                'mean')
    
    conc = np.concatenate((d, add_row.reshape(add_row.shape[0], 1)), axis=1)
    row_w_z = np.concatenate((add_row, [MAX_VAL]))
    res = np.append(conc, [row_w_z]).reshape(row_w_z.shape[0],row_w_z.shape[0])
           # newD = np.zeros((d.shape[0]+1, d.shape[1]+1))
    for ii in range(len(res)):
        for jj in range(len(res[ii])):
            if ii == cluster_idx[0] or jj == cluster_idx[1]:
                res[ii, jj] = MAX_VAL   
                res[jj, ii] = MAX_VAL
    return make_cluster(res, clusters, out_of_commission, labels, all_clusters)
                
    
if __name__ == '__main__':
    print('hclustering')
    cluster = Hlustering()
    cluster.parseData("../data/4clusters.csv")
    d = cluster.calculateDistances()
    print(d)
    # d = [[50, 2, 25, 14, 23, 5],
    # [2,50, 22, 16, 23, 6],
    # [25, 22, 50, 9, 4, 13],
    # [14, 16, 9, 50, 6, 7],
    # [23, 23, 4, 6, 50, 8],
    # [5, 6, 13, 7, 8, 50]]
    d = np.asarray(d)
    labels = list(range(len(d))) #['a', 'b', 'c', 'd', 'e', 'f']
    result = make_cluster(d, list(range(len(d))), [], labels, labels[:])
    print(result)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    
    # ax.scatter(cluster.rawData[...,0], cluster.rawData[...,1])
    # ax.set_title("expected clustering")
    # fig.show()
    
    
    # i = 2
    for x in result:
        print("new cluster")
        # fig, ax = plt.subplots(i, i)
        x = str(x).replace("(", "").replace(")", "").replace(",", "").replace("[", "").replace("]", "").split()
        x = [int(xi) for xi in x]
        curData = cluster.rawData[x]
        ax.scatter(curData[...,0], curData[...,1])
        
    ax.legend(["cluster"] * 4)
        
        
        
    
    # ax.scatter(cluster.rawData[...,0], cluster.rawData[...,1])
    # ax.set_title("expected clustering")
    # fig.show()
    
    
    




                
                
    # for y in range(d.shape[0]):
    #     for x in range(d.shape[1]):
    #         if y != i and x != j:
    #             if y < i and x < j:
    #                 print(y, x)
    #                 newD[y, x] = d[y, x]
    #             else:
    #                 newD[y - 1, x - 1] = d[y, x]
                    
                    
                    
                
                
