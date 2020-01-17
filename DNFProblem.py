import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import time
import scipy.stats as st
from functools import reduce
class DNFProblem:
    def __init__(self, clauseList, nbVariables, variableProbs):
        self.nbVariables = nbVariables
        self.nbLiterals = 2*nbVariables
        assert (self.nbVariables == len(variableProbs)) # make sure shape is consistent
        negatedProbs = 1 - np.array(variableProbs)
        self.varProbs = variableProbs # This is expected to be a NumPy array
        self.litProbs = np.concatenate((variableProbs, negatedProbs), axis=0)  # variableProbs (pos lit probs)
        self.logLitProbs = np.log(self.litProbs)
        self.varProbs = np.append(self.varProbs,-0.1) # Add a dummy variable that's always zero for optimisation
        self.clauseList = clauseList
        self.nbClauses = len(clauseList)
        self.computeClauseProbs()  # Compute clause probs
        # Convert probabilities
    def computeClauseProbs(self):  # Trying to build a numerically stable version of the Linear Time Coverage Alg.
        logProbs = []
        for clause in self.clauseList:  # Outside LTC loop, fine
            logProb = 0  # Initially 1
            for lit in clause:
                logProb += self.logLitProbs[lit]
            logProbs.append(logProb)
        self.unlikeliestClause = np.argmin(logProbs) # We now know the "unlikeliest clause"
        self.ratios = np.exp(logProbs - logProbs[self.unlikeliestClause])  # Get the prob ratios relative to min clause
        # Now that log probabilities are computed, find the "ratio"
        self.denomRatio = np.sum(self.ratios)
        self.clauseProbs = np.array(self.ratios / self.denomRatio)
        # Do some housekeeping
        self.uniformiseClauseWidths()  # New Dec 6
        self.universeDisjointProbSum = np.exp(logProbs[self.unlikeliestClause] + np.log(self.denomRatio))
        #print(self.universeDisjointProbSum)

        #Overhauled Extremely Memory Hungry Clause Representation (nbC * nbV). To check it, view old GitHub commits
    def uniformiseClauseWidths(self):  #Takes unequally long
        self.clauseVarList = [np.array(clause) % self.nbVariables for clause in self.clauseList]
        self.clauseVarValueList = [1 - np.array(clause) // self.nbVariables for clause in self.clauseList]
        self.clauseWidths = np.array([len(x) for x in self.clauseList])
        self.maxWidth = np.max(self.clauseWidths)  # Compute maximum clause width
        # Introduce a "dummy" extra variable at pos nbVars (actual pos nbVars + 1), which is always 0 ... for now
        self.clauseVarIndices = np.full((self.nbClauses, self.maxWidth), self.nbVariables)
        self.clauseVarValues = np.zeros((self.nbClauses, self.maxWidth))
        for i in range(self.nbClauses): # Only must do this once (pre-processing)
            self.clauseVarIndices[i, 0:self.clauseWidths[i]] = self.clauseVarList[i] # Add indices to representation
            self.clauseVarValues[i, 0:self.clauseWidths[i]] = self.clauseVarValueList[i] #And their values
            # New representation is (2 * nbC * maxWidth)
    def generateRandomAssignmentBatch(self, batchSize):  # Output is shaped [batchSize, nbVars]
        assignments = (np.random.uniform(0,1,size=(batchSize, self.nbVariables+1)) < self.varProbs)*1 #Introduce dummy
        return assignments
    def generateAssignmentBatch(self, batchClauses, batchSize):  # Passing batch size to avoid re-computing it
        assignments = self.generateRandomAssignmentBatch(batchSize) # Can compute size from length of clause list, but not really any gain
        indices = self.clauseVarIndices[batchClauses,:] # Get the var indices corresponding to the clauses
        values = self.clauseVarValues[batchClauses,:]
        assignments[self.advancedIndexingRows[0:batchSize,:],indices] = values #New: Advanced Indexing
        return assignments

    def checkClauseSATBatch(self, assignments,mask,clauses):
        representationIndices = self.clauseVarIndices[clauses,:] #
        representationValues = self.clauseVarValues[clauses,:]
        relevantValues = assignments[mask,representationIndices] # New: Advanced Indexing
        intermediate = np.abs(relevantValues - representationValues) != 1  # Also check notes
        SATs = np.all(intermediate, axis = 1)
        return SATs
    def LTCWithLogAns(self,epsilon,delta, batchSize,returnTSN=False,userNbSectors=0): # Can I optimise this?
        # Returns a distribution over the log of the answer
        mean = self.linearTimeCoverage(epsilon, delta, batchSize,returnTSN=returnTSN,userNbSectors=userNbSectors)
        if returnTSN:
            return mean
        else:
            logMean = np.log(np.minimum(1, mean))
            logError = np.log(1 + epsilon)
            Zscore = st.norm.ppf(1 - delta / 2)
            sigma = logError/Zscore
            return [logMean,sigma],[logMean,epsilon,delta]


    def linearTimeCoverageParaTime(self, epsilon, delta, batchSize):
        nbCores =  multiprocessing.cpu_count()
        result = Parallel(n_jobs=nbCores)(delayed(self.linearTimeCoverage)(epsilon=epsilon, delta=delta, batchSize = batchSize
                                , nbCores = nbCores) for i in range(nbCores))
        aggResult = np.mean(result)
        return aggResult
    def linearTimeCoverageParaBatch(self, epsilon, delta, batchSize):
        nbCores =  multiprocessing.cpu_count()
        results = np.array(Parallel(n_jobs=nbCores)(delayed(self.linearTimeCoverage)(epsilon=epsilon, delta=delta,
                                    batchSize = batchSize // nbCores, returnTSN = True) for i in range(nbCores)))
        overAllPs = np.sum(results,axis = 0)
        aggResult = np.mean(overAllPs[0]*self.universeDisjointProbSum / (self.nbClauses * overAllPs[1]))
        return aggResult

    def linearTimeCoverage(self, epsilon,delta,batchSize, nbCores = 1, returnTSN = False,userNbSectors = 0):
        #PLEASE DO NOT CHOOSE A PRIME NUMBER BATCH SIZE!
        #-STATISTICS-#
        totalTrialLength = 0
        totalTrials = 0
        totalSATChecks = 0
        iterationCounter = 0
        # Number of time steps: Karp, Luby, Madras paper
        clauseProbList = np.ndarray.tolist(self.clauseProbs)
        T = np.ceil((8*(1+epsilon)*self.nbClauses*np.log(2/delta))/(epsilon**2 * nbCores)) #Split the load (paraTime only)
        numberOfTrials = 0
        # December 30: Ideally, batch size be a strict multiple (or divisor) of the number of clauses,
        # such that every clause is tried nbC times (Recall that expectation for trial is lower bounded by 1/nbC)
        if userNbSectors == 0: # User did not specify sector size, so use my heuristic
            # Note: E[NbT] <= nbC, REMEMBER THAT
            heuristicSectorSize = (1.0/4)*(self.nbClauses/self.universeDisjointProbSum)
            if heuristicSectorSize >= (1.0/4) * self.nbClauses:
                heuristicSectorSize = (1.0/4) * self.nbClauses
            batchSizeFactors = np.array(list(factors(batchSize)))
            differenceToFactors = np.absolute(batchSizeFactors - heuristicSectorSize)
            sectorSize = batchSizeFactors[np.argmin(differenceToFactors)] # Get the factor that's closest
            nbSectors = batchSize //sectorSize
            # Now make sure number of sectors is divisible by target

        else:
            nbSectors = userNbSectors # Jan 1: Enable User to specify Number of Sectors
        #print("NB SECTORS: "+str(nbSectors))
            sectorSize = int(batchSize/nbSectors)
        sectorIndices = np.arange(sectorSize*nbSectors).reshape((nbSectors,sectorSize))
        littleT = np.zeros(nbSectors) # And set little T array size accordingly
        sectorsToUpdate = range(nbSectors) # All positions in batch should be replaced for init.
        howMany = nbSectors
        selectedClauseIndices = np.array([0]*batchSize)
        #---INITIALISATIONS---#
        assignments = np.zeros((batchSize,self.nbVariables+1))  # Initialise to empty, with dummy
        self.advancedIndexingRows = np.array([range(batchSize)]*self.maxWidth).T
        effectiveTime = 0 # Dec 29-30: New Batching Algorithm that guarantees correctness
        currentSector = 0 # Go through batch as if one at a time. This eliminates minimum bias.
        latestCompletedTrialLengths = np.zeros(nbSectors)
        trialInProgress = np.full(nbSectors,True)  # Formerly the barely descriptive usable trials
        individualIndicesArray = np.arange(batchSize)
        sectorsArray = np.arange(nbSectors)
        # Beautiful Result: When nbSectors is 1, this reduces to pure batching over the same assignment
        while True:  # Run until T total "effective" iterations have been made
            iterationCounter+=1
            # Step 1: Select #Batch clauses
            #print("Iteration Nb: " + str(iterationCounter))
            #print("New Assignments to Generate: " + str(howMany)+"/"+str(sectorsToUpdate))
            newSelectedClauseIndices = np.random.choice(self.nbClauses, size=howMany, p=clauseProbList)
            selectedClauseIndices[sectorsToUpdate] = newSelectedClauseIndices
            # Step 2: Generate Batch Assignments
            newAssignments = self.generateAssignmentBatch(newSelectedClauseIndices, howMany)
            # Step 3: Now Duplicate them across the sectors (NEW!)
            newAssignmentsByIndividualIndex = np.repeat(np.arange(howMany), sectorSize)
            sectorsToUpdateByIndividualIndex = sectorIndices[sectorsToUpdate].flatten()
            assignments[sectorsToUpdateByIndividualIndex,:] = newAssignments[newAssignmentsByIndividualIndex,:]
            # Step 4: Update Little T across the batch (Reset to 0 for new assignments)
            littleT[sectorsToUpdate] = 0
            # January 1: Reduce useless computations by only running trials over the incomplete assignments
            nbToTry = np.sum(trialInProgress)*sectorSize
            randomClauseIndices = np.random.randint(self.nbClauses, size=nbToTry)
            totalSATChecks+=nbToTry # Statistics
            trialInProgressByIndex = np.repeat(trialInProgress,sectorSize)
            mask = self.advancedIndexingRows[trialInProgressByIndex,:]
            SATs = np.full(batchSize, False)
            SATs[trialInProgressByIndex] = self.checkClauseSATBatch(assignments,mask,randomClauseIndices) # Check SAT as normal
            # (NEW! Sector SATs)
            SATIndices = individualIndicesArray[SATs] # Useful for last success (if any) and little T
            SATIndicesReversed = np.flip(SATIndices,0) # Necessary to get first success ... if any
            #print("--------------------------")
            #print("SAT Indices:"+str(SATIndices))
            SATSectors = SATIndices // sectorSize
            SATSectorsBool = np.full(nbSectors, True)
            SATSectorsBool[SATSectors] = False # Used to increment the failed trials
            SATSectorsBoolReversed = np.logical_not(SATSectorsBool) # True when successful, useful for later
            SATSectorsReversed = SATIndicesReversed // sectorSize
            SATSectorIndices = SATIndices % sectorSize
            SATSectorIndicesReversed = SATIndicesReversed % sectorSize
            #print("SATSectorIndices: "+str(SATSectorIndices))
            #print("SATSectors: " + str(SATSectors))
            #print("SATSectorIndicesReversed: " + str(SATSectorIndicesReversed))
            #print("SATSectorsReversed: " + str(SATSectorsReversed))
            #Dec 31: Little T/ Latest Lengths Update Bug Fix
            firstSuccess = np.full(nbSectors, None) # Ah, the None pointer
            firstSuccess[SATSectorsReversed] = SATSectorIndicesReversed + 1 # Jan 2 Bug Fix: Increment Index By 1
            #print("First Success:"+str(firstSuccess))
            lastSuccess = np.full(nbSectors, None)
            lastSuccess[SATSectors] = SATSectorIndices + 1 # Jan 2 Bug Fix
            #print("Last Success:"+str(lastSuccess))
            # Update Latest Lengths ONLY IF IN PROGRESS AND SUCCESSFUL
            latestToUpdate = np.logical_and(SATSectorsBoolReversed,trialInProgress)
            latestCompletedTrialLengths[latestToUpdate] = littleT[latestToUpdate]+firstSuccess[latestToUpdate]  # Up To First Success
            # Carry residual from last trial (which is 0 if just updated)
            littleT[SATSectorsBoolReversed] = sectorSize - lastSuccess[SATSectorsBoolReversed] # Count from last success
            trialInProgress[SATSectorsBoolReversed] = False # Update usability flags
            littleT[SATSectorsBool] += sectorSize # Update the trials that have yet to finish
            # Use the successful ones to do as needed

            totalTrialLength+=np.sum(latestCompletedTrialLengths[np.logical_not(SATSectorsBool)])
            totalTrials+= SATSectors.shape[0]# STATS
            sectorsToUpdate = np.array([],dtype=np.int32) # Initially nothing to update
            nbSectorsToUpdate = 0
            #print("Little Ts:" + str(littleT))
            #print("Latest Lengths:" + str(latestCompletedTrialLengths))
            #print("Usable Trials:" + str(trialInProgress))
            #print("Effective Time:" + str(effectiveTime))
            #print("Current Index:" + str(currentSector))
            #print(latestCompletedTrialLengths)
            if not trialInProgress[currentSector]:  # Trial we are waiting on in our "semi-serial" run has succeeded
                sectorsInProgress = sectorsArray[trialInProgress]  # No Loops
                if sectorsInProgress.shape[0] == 0:  # All the sector assignments are successful. Unlikely but possible
                    firstInProgressSector = currentSector  # Take all the assignments
                    runOverBound = True
                else:
                    sectorsInProgressAhead = sectorsInProgress[sectorsInProgress > currentSector]
                    if sectorsInProgressAhead.shape[0] == 0:  # There are no in progress sectors ahead of the current index,
                        firstInProgressSector = sectorsInProgress[0]  # Take advantage of the fact that it's already sorted.
                        # This is sure to exist because of the previous check
                        runOverBound = True
                    else:  # Next in Progress sector is ahead without cycling back
                        firstInProgressSector = sectorsInProgressAhead[0]
                        runOverBound = False
                # Now Update everything
                if runOverBound:
                    sectorsToUpdate = sectorsArray[np.r_[:firstInProgressSector, currentSector:nbSectors]]
                    newlyIncorporatedTrials = np.sum(latestCompletedTrialLengths[sectorsToUpdate])
                else:
                    sectorsToUpdate = sectorsArray[currentSector:firstInProgressSector]
                    newlyIncorporatedTrials = np.sum(latestCompletedTrialLengths[sectorsToUpdate])
                # We now have the trial times for completed sectors which we incorporate into the effective time
                if effectiveTime + newlyIncorporatedTrials > T:  # We have finished our experiment, T iterations reached
                    threshold = T - effectiveTime  # Compute the needed number of trials to "cap off"
                    cumulativeTrials = np.cumsum(latestCompletedTrialLengths[sectorsToUpdate]) <= threshold  # Cumulative sum and comparison with threshold
                    lengthOfAddedSuccesses = np.sum(cumulativeTrials)  # Compute how many of these trials to add
                    '''
                    # TOTAL estimator
                    effectiveTOTAL = effectiveTime + cumulativeTrials[lengthOfAddedSuccesses - 1]  # Useful to have all KLM paper estimators
                    # Standard Y estimator
                    fPrimes = latestCompletedTrialLengths[sectorsToUpdate][:lengthOfAddedSuccesses] / self.nbClauses
                    fMean += np.sum(fPrimes)  # Get all the f Primes added on top'''
                    numberOfTrials += lengthOfAddedSuccesses  # Add the successful trials here
                    effectiveTime = effectiveTime + cumulativeTrials[lengthOfAddedSuccesses]
                    break  # No need to update batch indices, just break. Experiment finished
                else:  # Still below threshold, so incorporate all the selected trials
                    effectiveTime += newlyIncorporatedTrials
                    nbSectorsToUpdate = sectorsToUpdate.shape[0]
                    numberOfTrials += nbSectorsToUpdate  # Add these successes
                # Now update the in progress flags, current sector index, and the trial lengths
                currentSector = (currentSector + nbSectorsToUpdate) % nbSectors
                trialInProgress[sectorsToUpdate] = True
                latestCompletedTrialLengths[sectorsToUpdate] = 0  # Not necessary, but still good to keep in mind
            # Now, Update the three arrays (littleT, assignments,selectedClauseIndices) that reflect the batch state to only keep those that failed.
            # Regardless of success of trial at index, continue sampling across the batch as before
            # Update the three arrays that reflect the batch state to only keep those that failed
            howMany = nbSectorsToUpdate
            #batchIndicesToUpdate = trialIndices  # Get the indices where the random trial was successful. These will be updated later
            #howMany = batchIndicesToUpdate.shape[0] # Continue updating the finished trials as before
            #print(effectiveTime / (iterationCounter * batchSize))
            #print(effectiveTime)
            #print(latestCompletedTrialLengths)
        #print(iterationCounter)
        #print(numberOfTrials)
        #print(totalTrials)
        #print(effectiveTime / numberOfTrials)
        #print("SAT Efficiency: "+str(effectiveTime/totalSATChecks))
        #print("Absolute Efficiency: "+str(effectiveTime/(iterationCounter*batchSize)))
        #print(totalTrialLength/ totalTrials)
        if returnTSN:
            return [T, numberOfTrials]
        else:
            # return fMean * self.universeDisjointProbSum/numberOfTrials
            #print((T * self.universeDisjointProbSum)/ (numberOfTrials * self.nbClauses))
            return (T * self.universeDisjointProbSum)/ (numberOfTrials * self.nbClauses)

# For Heuristic
#https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))