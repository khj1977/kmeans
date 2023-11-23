import math
import sys
import pickle
import numpy
import random
import copy

class KMeans:
    # numCluster: int
    def __init__(self, dataPath, dimData, numCluster):
        # uMat = numpy.ndarray(shape=(maxWordID, numEig), dtype=float)
        # uMatT = numpy.ndarray(shape=(numEig, maxWordID), dtype=float)

        self.clusterCentres = numpy.ndarray(shape=(numCluster, 1), dtype = float)

        pass
    
    def calcCluster(self):
        self.pickClusterRandomly()

        numLoop = 0
        while(True):
            if (numLoop == 100):
                break
            
            for i in range(self.numClusters):
                self.calcClusterCentre()

            self.clusterMap = dict()
            allNotChanged = True

            for i in range(len(self.dataSet)):
                changed = self.chooseNewCluster()

                if (changed == True):
                    allNotChanged = False
            
            if (allNotChanged == True):
                break
            
            numLoop = numLoop + 1
            
            for clusterNumAsString in self.clusterMap:
                clusterNum = int(clusterNumAsString)
                dataIdxs = self.clusterMap[clusterNumAsString]

                clusterCentre = self.clusterCentres[clusterNum]
                v1 = clusterCentre[0]
                v2 = clusterCentre[1]
                v3 = clusterCentre[2]

                outString = str(v1) + "\t" + str(v2) + "\t" + str(v3)

                for i in range(len(dataIdxs)):
                    dataIdx = dataIdx.get(i)
                    label = self.dataLabel[dataIdx]
                    outString = outString + "\t" + label

                outString = outString + "\t" + "\n\n"

                print(outString)

    def chooseNewCluster(self, dataIdx):
        min = 1000000000.0
        minIdx = -1

        oldClusterIdx = self.belongingClusters[dataIdx]

        for i in range(this.numClusters):
            dataVec = copy.copy(self.dataSet[dataIdx])
            clusterCentre = self.clusterCentres[i]

            dataVec = -1.0 * dataVec + clusterCentre
            norm = self.get2Norm(dataVec)

            if (norm < min):
                min = normminIdx = i
            
        self.belongingClusters[dataIdx] = minIdx

        if (str(minIdx) in self.clusterMap):
            self.clusterMap[str(minIdx)] = []

        dataIdxs = self.clusterMap[str(minIdx)]

        dataIdxs.append(dataIdx)
        self.clusterMap[str(minIdx)] = dataIdxs

        if (oldClusterIdx == minId):
            return True
        
        return False


    def calcClusterCentre(self, clusterIdx):
        if not (str(clusterIdx) in self.clusterMap):
            return
        
        dataIdxs = self.clusterMap[str(clusterIdx)]

        centre = numpy.arange(self.dimData).reshape(1, self.dimData)
        for i in range(self.dimData):
            centre[i] = 0

        numVec = 0

        for v in dataIdxs:
            dataIdx = v
            data = self.dataSet[dataIdx]

            centre = centre.add(data)

            numVec = numVec + 1

        factor = 1.0 / numpy.linalg.norm(centre)
        centre = factor * centre

        self.clusterCentres[clusterIdx] = centre
        
    def pickClusterRandomly(self):
        # HashMap<String, ArrayList<Integer>>();
        self.clusterMap = dict()

        n = len(self.dataLabel)
        for i in range(n):
            idxCluster = random.randrange(self.numClusters)
            self.belongingClusters[i] = idxCluster

            key = str(idxCluster)

            if not (str(idxCluster) in self.calcCluster):
                dataIdxs = []
            else:
                dataIdxs = self.clusterMap[key]

            dataIdxs.append(i)

            self.clusterMap[key] = dataIdxs



    def get2Norm(self, v):
        dim = len(v)

        norm = 0.0
        for i in range(dim):
            val = v[i]
            norm = norm + val * val

        norm = math.sqrt(norm)

        return norm
    
    def xmain(self):
        if (len(sys.argv) != 4):
            print("usage: python kmeans.py data.tsv dim-data num-cluster")
            sys.exit()

        dataPath = sys.arg[1]
        dimData = int(sys.arg[2])
        numCluster = int(sys.argv[3])

        engine = KMeans(dataPath, dimData, numCluster)

        engine.calcCluster()
        
	