# By width, by size, by everything
import os
import pickle
import random
import numpy as np
logPath = "dataAnalytics.txt"
fileDirectory="data/"
dataFiles = [fileDirectory+f for f in os.listdir(fileDirectory) if f.endswith(".p")] # Get the files
totalProb = 0
totalFormulas = 0
logFile = open(logPath,"w")
logFile.write("Data Analytics: \r\n")
logFile.close()
filePath = "logKL.txt"
for file in dataFiles:
    with open(file,"rb") as f:
        batches = pickle.load(f)
    # Shuffle Batches
    shuffling = [i for i in range(len(batches))]
    random.shuffle(shuffling)
    fileSumProb = 0
    fileFormulas = 0
    batches = [batches[shuffling[i]] for i in range(len(batches))]
    for batchIndex, batch in enumerate(batches):
        # Loads a new batch
        nbC, posLit, disjConj, conjLit, approxKL, approxRA = batch
        batchSumProb = np.sum(np.exp(approxKL[:,0]))
        batchFormulas = approxKL.shape[0]
        fileSumProb += batchSumProb
        fileFormulas += batchFormulas
    # End of File
    logFile = open(logPath, "a+")
    logFile.write(file+" average probability: "+str(fileSumProb/fileFormulas) + "\r\n")
    logFile.write(file + " Number of Formulas: " + str(fileFormulas) + "\r\n\r\n")
    logFile.close()
    totalProb += fileSumProb
    totalFormulas += fileFormulas
logFile = open(logPath, "a+")
logFile.write("Overall Average: " + str(totalProb / totalFormulas) + "\r\n")
logFile.write("Total Number of Formulas: " + str(totalFormulas) + "\r\n")
logFile.close()
