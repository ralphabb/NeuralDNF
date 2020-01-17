import matplotlib.pyplot as plt
import pickle, numpy as np
from GraphNeuralNet import GGNN
import os
''' BAD IDEA: x: Formula Index, y1: KLM, y2: 
plt.plot(xAxis, np.array(KLMMus)[xAxis])
plt.plot(xAxis, np.array(networkMus)[xAxis])
plt.show()'''
''' Plan B: Simple scatter plot 
plt.scatter(KLMMus, networkMus)
plt.show() '''
# Plan C: Heat-Map
# Step 1: Discretise the values into very small "buckets"
from matplotlib2tikz import save as tikz_save

def produce_heatmap(KLMMus,networkMus):
    nbBuckets = 600
    sizeOfGrain = 2 # in terms of discretisations
    occurrencesThreshold = 50
    nbOfTicks = 10
    KLMMeans = np.floor(np.array(KLMMus,dtype="float") * nbBuckets).astype(int)
    print(np.unique(KLMMeans,return_counts=True))
    networkMeans = np.floor(np.array(networkMus,dtype="float") * nbBuckets).astype(int)
    # Remove 400 values
    KLMMeans[KLMMeans >= nbBuckets] = nbBuckets - 1
    networkMeans[networkMeans >= nbBuckets] = nbBuckets -1
    cardinalitiesPerRange = np.zeros((nbBuckets, nbBuckets),dtype=np.int32)
    xIndex = len(KLMMus)
    for i in range(xIndex):
        x = KLMMeans[i]
        y = networkMeans[i]
        cardinalitiesPerRange[x,y]+=sizeOfGrain
        for j in range(x- sizeOfGrain + 1, x + sizeOfGrain):
            for k in range(y - sizeOfGrain + 1, y + sizeOfGrain):
                manhattanDistance = np.abs(x-j) + np.abs(y-k)
                if manhattanDistance < sizeOfGrain:
                    try:
                        cardinalitiesPerRange[j,k]+=sizeOfGrain - manhattanDistance
                    except:
                        continue # Best way to handle out of bounds
    #Normalise by max
    cardinalitiesPerRange[cardinalitiesPerRange > occurrencesThreshold] = occurrencesThreshold
    cardinalitiesPerRange = np.log(1+cardinalitiesPerRange)
    cardinalitiesPerRange /= np.amax(cardinalitiesPerRange)
    plt.gca().invert_yaxis()
    plt.imshow(cardinalitiesPerRange,cmap="binary", interpolation='gaussian')
    tickSize =int(nbBuckets/nbOfTicks)
    tickBase = range(0,nbBuckets+1,tickSize)
    xi = [x/ nbBuckets for x in tickBase]
    yi = [y/ nbBuckets for y in tickBase]
    plt.xticks(tickBase,xi)
    plt.yticks(tickBase,yi)
    plt.xlabel("KLM Approximations")
    plt.ylabel("GNN Approximations")
    #plt.show()
    tikz_save("test.tex")
    # plt.savefig('heatmap_8iter.png', bbox_inches="tight")

def identifyCandidatesForProbVis(batchDir, discreteStep = 20,tolerance = 0.02):
    bucketFilled = np.array([False]*(discreteStep+1))
    bucketLowerBound = [i / discreteStep - tolerance for i in range(discreteStep+1)]
    bucketUpperBound = [i / discreteStep + tolerance for i in range(discreteStep+1)]
    bucketBatchFile = [None]*(discreteStep+1)
    bucketBatchIndex = [None]*(discreteStep+1)
    bucketBatchFormulaIndex = [None]*(discreteStep+1)
    # Can shuffle this for randomisation
    batchFiles = [batchDir + "/"+ f for f in os.listdir(batchDir) if f.endswith(".p")]  # Prepare data for loading
    # Mother of for loops
    for batchFile in batchFiles:
        with open(batchFile, "rb") as file:
            batchSet = pickle.load(file)
            for index, batch in enumerate(batchSet):
                nbC,posLit,disjConj,conjLit,approxKL,approxRA = batch
                for j in range(approxKL.shape[0]):
                    formulaKL = np.exp(approxKL[j,0])
                    for k in range(discreteStep+1):
                        if not bucketFilled[k] and bucketUpperBound[k] >= formulaKL >= bucketLowerBound[k]:
                            bucketBatchFile[k] = batchFile
                            bucketBatchIndex[k] = index
                            bucketBatchFormulaIndex[k] = j
                            bucketFilled[k] = True
                            #print(formulaKL)
                # End of single batch check
                if bucketFilled.all():
                    return bucketBatchFile, bucketBatchIndex, bucketBatchFormulaIndex
def visualiseProbOverTime(bucketBatchFiles,batchBatchList,batchFormulaIndeces,nbIter = 32):
    GraphNet = GGNN(vizMode=True,communicationProtocol=2,nbIterations=nbIter) # Enable the new visualisation mode
    probSequences = []
    for index, file in enumerate(bucketBatchFiles):
        print(index)
        with open(file,"rb") as f:
            batchSet = pickle.load(f)
        batch = batchSet[batchBatchList[index]]
        nbC, posLit, disjConj, conjLit, approxKL, approxRA = batch
        print(np.exp(approxKL[batchFormulaIndeces[index],0]))
        # Hacky, but does the trick
        probs = np.array(GraphNet.produceVizProbs(nbC,posLit,disjConj,conjLit,batchFormulaIndeces[index],paramLocation="netParams_2_8Iter/values.ckpt"))
        probs = probs[:,:,0]
        probSequences.extend(probs)
        # Now write vis code
    probSequences = np.array(probSequences)
    print(probSequences.shape)

    plt.imshow(probSequences, cmap="Spectral", interpolation='nearest') # Play with color scheme
    xTickBase = range(0, probSequences.shape[1], 2)
    yTickBase = range(probSequences.shape[0])
    xi = [x + 1 for x in xTickBase]
    yi = [y + 1 for y in yTickBase]
    plt.xticks(xTickBase, xi)
    plt.yticks(yTickBase, yi)
    plt.xlabel("Message Passing Iteration")
    plt.ylabel("Formulas")
    #plt.show()
    tikz_save("VisualisingFormulaProbs.tex")
def MeanVsStDev(networkSigmas, networkMus):
    plt.scatter(networkMus, networkSigmas)
    plt.show()
def errorVsStDev(networkSigmas, absDiffs):
    plt.scatter(absDiffs, networkSigmas)
    plt.show()
if __name__=="__main__":
    # Part A: Scatter plot
    # Step 1: Load the visualisation data
    with open("8IterTestSet.p", "rb") as f:
        data = pickle.load(f)
    batchRunTimes, batchSizes, networkMus, networkSigmas, KLMMus, AbsDiffs, KLDivs = data  # Unpack Data
    print(len(networkMus))
    # produce_heatmap(KLMMus, networkMus)
    bF, BL, BFI = identifyCandidatesForProbVis("testData5K", tolerance=0.005)
    visualiseProbOverTime(bF,BL,BFI, nbIter=8)
    #MeanVsStDev(networkSigmas, networkMus)