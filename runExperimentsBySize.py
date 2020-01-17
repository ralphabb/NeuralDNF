from GraphNeuralNet import GGNN
import os, argparse, pickle, time, numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def runExperiments_SynthData():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    argParser = argparse.ArgumentParser(description='Test the Neural#DNF system')
    argParser.add_argument("dataDir",type=str,help="The directory from which labelled test data is loaded")
    argParser.add_argument("-communicationProtocol", type=int, default=2, metavar='',
                           help="Which Communication Protocol to use for testing (Default 2)")
    argParser.add_argument("-embeddingDim", type=int, default=128, metavar='',
                           help="The embedding size to use (Default 128)")
    argParser.add_argument('-numIter',type=int, metavar='',
                           help="The number of message passing iterations to run (Default 8)",default=8)
    argParser.add_argument("-weightsDir",type=str,default=None, metavar='',
                           help="The directory from which to load network parameters")
    argParser.add_argument("-measureRunTime",type=str2bool,default=True, metavar='',
                           help="Specify whether timing tests should be conducted (Default True)")
    argParser.add_argument("-outputFileName", type=str, default="Results.p", metavar='',
                           help="Set a custom output file name (Default Results.p)")
    # Parse the Arguments
    args = argParser.parse_args()
    nbIter = args.numIter
    if args.weightsDir is not None:
        paramLocation = args.weightsDir+"/values.ckpt"
    else:
        paramLocation = None
    dataDir = args.dataDir
    commProt = args.communicationProtocol
    embDim = args.embeddingDim
    measureRunTime = args.measureRunTime
    fileName = args.outputFileName
    thresholds = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2])
    GraphNet = GGNN(nbIterations=nbIter,communicationProtocol=commProt,embeddingDim=embDim) # Initialise the Graph Neural Net
    sizes = [50,100,250,500,750,1000,2500,5000]
    print(" Thresholds: " + str(thresholds))
    for size in sizes:
        dataFiles = [dataDir + f for f in os.listdir(dataDir) if f.endswith("_"+str(size)+"0.p")] # Load data by size
        thresholdsComparable = np.expand_dims(np.array(thresholds).T,axis=-1)
        nbThresholds = thresholds.shape[0]
        counts = np.zeros(nbThresholds)
        # Measured Entities
        batchRunTimes = []
        batchSizes = []
        networkMus = []
        networkSigmas = []
        KLMMus = []
        AbsDiffs = []
        KLDivs = []
        nbFormulas = 0
        #Start experiments
        print("Size "+str(size)+":")
        try:
            for dataFile in dataFiles:
                with open(dataFile, "rb" ) as file:
                    batchSet = pickle.load(file) # Load batch set
                    # New: Only load session once
                    GraphNet.loadParamsSession(paramLocation=paramLocation)
                for index, batch in enumerate(batchSet):
                    nbC, posLit, disjConj, conjLit, approxKL, approxRA = batch # Load individual batch
                    if measureRunTime: # Measure Time
                        tBefore = time.time()
                        logMeans, logVariances = GraphNet.forwardPass(nbConjunctions=nbC, posLitProbs=posLit,
                                                                      disjConj = disjConj, conjLit=conjLit,
                                                                      createSession=False)
                        runTime = time.time() - tBefore
                    else: # Don't measure time
                        runTime="Not Measured" # Won't be used, but to eliminate the pesky warning
                        logMeans, logVariances = GraphNet.forwardPass(nbConjunctions=nbC, posLitProbs=posLit,
                                                                      disjConj=disjConj, conjLit=conjLit,
                                                                      createSession=False)
                    batchSize = approxKL.shape[0]
                    batchSizes.append(batchSize)
                    #Now log the individual batcheS
                    if measureRunTime:
                        batchRunTimes.append(runTime)
                    # Convert entries to probabilities, as opposed to logs of probabilities
                    approxProbValues = np.exp(approxKL)
                    networkProbValues = np.exp(logMeans)
                    # Eliminate useless for loop
                    approxLogValues = approxKL[:,0]
                    approxStDevs = approxKL[:, 1]
                    networkLogValues = logMeans[:,0]
                    approxValues = approxProbValues[:,0]
                    networkValues = networkProbValues[:,0]
                    logStDevValues = logVariances[:,0]
                    absDifferences = computeAbsDiff(networkValues, approxValues)
                    networkMus.extend(networkValues)
                    networkSigmas.extend(logStDevValues)
                    KLMMus.extend(approxValues)
                    AbsDiffs.extend(absDifferences)
                    KLDivs.extend(computeKLDiv(networkLogValues,logStDevValues,approxLogValues,approxStDevs))

                    #Check how many match the thresholds
                    AbsDiffsRepeated = np.repeat(np.expand_dims(absDifferences, axis=0), nbThresholds, axis=0) # Repeat
                    threshCompArray = (AbsDiffsRepeated <= thresholdsComparable) * 1
                    nbSuccesses = np.sum(threshCompArray,axis =1)
                    counts+=nbSuccesses
                    nbFormulas += batchSize
            # End of testing
            with open(str(size)+"_"+fileName,"wb") as f: # Make it file-specific
                print(np.divide(counts, nbFormulas))
                statsObject = (batchRunTimes, batchSizes, networkMus, networkSigmas, KLMMus, AbsDiffs, KLDivs)
                pickle.dump(statsObject, f) # End of experiments
        except KeyboardInterrupt: # Make robust to interruption
            pass
def computeKLDiv(networkLogMean, networkLogStDev, approxMean, approxStDev):
    return np.log(approxStDev / networkLogStDev) - 0.5 + np.divide(networkLogStDev**2 +
            (approxMean - networkLogMean)**2, 2*approxStDev**2)
def computeAbsDiff(networkMean, approxMean):
    return np.abs(approxMean - networkMean)
def saveExcelSheet(book,fileName):
    print("Saving Results so far and quitting... ")
    book.save(fileName)
if __name__ == "__main__":
    runExperiments_SynthData()
