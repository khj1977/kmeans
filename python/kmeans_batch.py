import sys
import kmeans

class KMeansBatch:

    def run(self):
        self.xmain()

    def xmain(self):
        if (len(sys.argv) != 4):
            print("usage: python kmeans.py data.tsv dim-data num-cluster")
            sys.exit()

        dataPath = sys.arg[1]
        dimData = int(sys.arg[2])
        numCluster = int(sys.argv[3])

        engine = kmeans.KMeans(dataPath, dimData, numCluster)

        engine.calcCluster()


batch = KMeansBatch()
batch.run()