import numpy as np
import math
from math import ceil
import tensorflow as tf  # TensorFlow is needed for sparse matrix definition
import pickle
from DNFProblem import DNFProblem
from joblib import Parallel, delayed
import multiprocessing


def generate(nbVars, nbClauses, minCWidth, maxCWidth, R=0, Q=1,uniformisePrivilegedVars=True):
    # Min appearances ensures that all variables are used
    # Step 1: Generate widths of all clauses
    # Removed hard width requirement
    nbClauseSlots = 0
    clauseWidths = []
    failedAttempts = 0
    while nbClauseSlots < nbVars:
        clauseWidths = np.random.randint(maxCWidth - minCWidth + 1, size = nbClauses) + minCWidth
        # Minimum Appearances is Hard-Coded as 1. Balls (appearances) in boxes (variables)
        nbClauseSlots = np.sum(clauseWidths)
        if nbClauseSlots < nbVars:
            failedAttempts+=1
            if failedAttempts >=5:
                raise TypeError(" Could not generate sufficient slots in 5 different attempts. Try a more generous allocation")
    clauseVacancies = np.copy(clauseWidths)
    # New: Introduction of R and Q
    excess = nbClauseSlots - nbVars  # How many extra slots do we have?
    excessAllocation = int(R * excess)
    if Q > 0 and R > 0:
        standardAllocation = nbClauseSlots - excessAllocation # Equivalent to nbVars + (1 - R)*excess
    else:
        standardAllocation = nbClauseSlots  # As before
    nbSeparators = standardAllocation - 1
    separators = np.random.choice(nbSeparators, size=nbVars - 1, replace=False) + 1 # Allocate such that all variables get at least one spot
    separators.sort()
    separators = np.append(separators, standardAllocation)
    separators = np.insert(separators,0, 0)
    varAllocations = np.diff(separators)
    if Q > 0 and R > 0:
        nbPrivilegedVars = ceil(Q * nbVars)  # These are the variables that will be allocated additional appearances.
        #  Made Ceil to be more fair if Q is randomised
        privilegedVars = np.random.choice(nbVars, size=nbPrivilegedVars, replace=False)  # Choose "privileged" vars
        # Now allocate balls in boxes (with empty boxes allowed)
        privilegedSeparators = np.random.choice(excessAllocation + nbPrivilegedVars - 1, size=nbPrivilegedVars - 1, replace=False)
        privilegedSeparators.sort()
        privilegedSeparators = np.append(privilegedSeparators, excessAllocation + nbPrivilegedVars - 1) # Bug fix: Forgot - 1
        privilegedSeparators = np.insert(privilegedSeparators, 0, -1)  # To avoid need for an indexed call
        privilegedVarAllocations = np.diff(privilegedSeparators) - 1  # Almost similar to the non-empty case, except differences have to be subbed by one
        varAllocations[privilegedVars] += privilegedVarAllocations
    # End of new part
    clauseIndices = np.arange(nbClauses)
    runningSum = 0
    lastIndex = 0
    variableOrder = np.argsort(varAllocations)[::-1]  # Decreasing order
    # To deal with non-uniformities... fill them first
    maxVacancy = np.max(clauseVacancies) # Initialise the counting of "maximally vacant" clauses. Initially = maxWidth
    clausesExactlyAtMaxVacancy = clauseVacancies == maxVacancy
    clausesOverMaxVacancy = clauseVacancies > maxVacancy
    clausesGEMaxVacancy = np.logical_or(clausesExactlyAtMaxVacancy,clausesOverMaxVacancy)
    maxVacancyClauseCount = np.count_nonzero(clausesGEMaxVacancy)
    clauses = np.full((nbClauses, maxVacancy), 2 * nbVars)  # Use a uniformised representation
    for oIndex,index in enumerate(variableOrder):
        rerun = True # This is to handle the exceptional case where we exceed the clauseCount at the final index (Mutex break)
        while rerun:
            allocation = varAllocations[index]
            if runningSum + allocation > maxVacancyClauseCount or oIndex == nbVars - 1: # We have reached the threshold or end of list
                if oIndex == nbVars - 1: # Jan 29 Note: These are not Mutex --> This is a bug. What if both conditions are met at the same time?
                    relevantAllocations = varAllocations[variableOrder[lastIndex:]] # Step 1: Extract the relevant variable allocations
                    relevantVariables = variableOrder[np.arange(lastIndex,nbVars)] # 1.1 get the indices
                    if runningSum+allocation <= maxVacancyClauseCount:
                        runningSum += allocation # Use the last variable
                        rerun = False
                    else: # only use this to disable rerun
                        rerun = True # This is where we need a rerun
                        relevantAllocations = varAllocations[variableOrder[lastIndex:oIndex]]  # 1
                        relevantVariables = variableOrder[np.arange(lastIndex, oIndex)]  # 1.1
                else:
                    rerun = False
                    relevantAllocations = varAllocations[variableOrder[lastIndex:oIndex]] # 1
                    relevantVariables = variableOrder[np.arange(lastIndex, oIndex)] # 1.1
                    # Slight Bug Fix to allow more randomised distribution across variables
                variableAllocsSeparate = np.repeat(relevantVariables,relevantAllocations) # 1.2 translate from multiset to list
                exactClauses = clauseIndices[clausesExactlyAtMaxVacancy] # Step 2: Compute Clauses at and/or over max vacancy
                overClauses = clauseIndices[clausesOverMaxVacancy]
                nbOverClauses = overClauses.shape[0]
                selectedClauses = np.zeros(runningSum,int)
                # Heuristic : Always select clauses STRICTLY OVER max vacancy benchmark
                dropMaxVacancy = True #Default is False, becomes True if more > max variables ,in which case next iter should have same max
                if nbOverClauses > runningSum:
                    selectedClauses = np.random.choice(overClauses, size=runningSum, replace=False) # Randomly pick from the overs
                    dropMaxVacancy = False
                    #This will reduce their numbers but some will survive. In this case, max Vacancy must not drop
                else:
                    selectedClauses[:nbOverClauses] = overClauses
                    '''print(len(exactClauses))
                    print(nbOverClauses)
                    print(runningSum)
                    print(allocation)
                    print("------------")'''
                    selectedClauses[nbOverClauses:] = np.random.choice(exactClauses, size=runningSum - nbOverClauses, replace=False) # Step 3: Select "runningSum" clauses
                '''  This is not a perfect process. Some maximally vacant clauses will not be filled. So, to avoid having 
                the next iteration trying to fill the max again, which would be 1 or 2 clauses, which could end up thrashing
                indefinitely, we loosen the def of maxVacancy to a decrementing running count, which decrements at every 
                draw and then sampling is done at >= to max, so that omitted clauses are caught up with later '''
                # Feb 1 Addition: Not uniform literal selection when Q, R > 0
                if uniformisePrivilegedVars and Q > 0 and R > 0: # New February 1
                    # Make all literals corresponding to a privileged variable all positive or negative
                    privilegedVariables = np.isin(variableAllocsSeparate,privilegedVars) # Identify privileged vars in current alloc
                    # Because of condition on Q and R, privilegedVars will be initialised, so no worry there...
                    nonPrivilegedVariables = np.logical_not(privilegedVariables) # And the non-privileged
                    nbOfNonPrivVar = np.sum(1*nonPrivilegedVariables)
                    # To separate privileged and non-privileged in the original alloc array, "negate" the privileged.
                    separationVarAlloc = nbVars*privilegedVariables+variableAllocsSeparate
                    uniqueValues,indices = np.unique(separationVarAlloc,return_inverse=True)
                    uniquePrivMask = uniqueValues >= nbVars
                    uniquePrivVars = uniqueValues[uniquePrivMask] # Now identify the unique privileged variables
                    nbPrivVarsInAlloc = uniquePrivVars.shape[0]
                    if nbPrivVarsInAlloc > 0:
                        uniformRandomisation = np.random.randint(2,size=nbPrivVarsInAlloc)*nbVars
                        # And the punch line ... update the new unique values with the random values
                        uniqueValues[uniquePrivMask] = uniqueValues[uniquePrivMask]%nbVars + uniformRandomisation
                        literalAllocsSeparate = uniqueValues[indices] # Privileged vars treated
                        # And now allocate the non-privileged variables as usual
                        literalAllocsSeparate[nonPrivilegedVariables]+=np.random.randint(2,size=nbOfNonPrivVar)*nbVars
                    else: # No Privileged Variables. Treat as before
                        literalAllocsSeparate = variableAllocsSeparate + np.random.randint(2, size=runningSum) * nbVars
                else:
                    literalAllocsSeparate = variableAllocsSeparate + np.random.randint(2,size=runningSum)*nbVars # Step 4: Compute Literal Negation
                clauses[selectedClauses,clauseVacancies[selectedClauses] - 1] = literalAllocsSeparate# Step 5: Use Advanced Indexing to update clauses
                ''' Writing into the clause array is happening in the reverse order to avoid unnecessary subtraction of 
                clause widths at every time step '''
                lastIndex = oIndex # Step 6: Update starting point
                clauseVacancies[selectedClauses] -= 1  # Step 7: Update the selected clauses' fullness
                if not rerun:
                    runningSum = allocation # Step 8: Reset the running sum to the starting value
                else:
                    runningSum = 0 # So as not to count the last element twice.
                    # Normally it increments to the one after, but in the rerun case it's the same number that would be counted twice
                # Step 9: Update the max vacancy and the rest now
                if dropMaxVacancy: # Only do this if nb > max <= runningSum, otherwise we'll need to run again at the same max
                    maxVacancy = maxVacancy - 1 if maxVacancy > 1 else maxVacancy #Loose update based on comment above. Don't allow 0 max vacancy otherwise can fill full clauses
                clausesExactlyAtMaxVacancy = clauseVacancies == maxVacancy
                clausesOverMaxVacancy = clauseVacancies > maxVacancy
                clausesGEMaxVacancy = np.logical_or(clausesExactlyAtMaxVacancy, clausesOverMaxVacancy)
                maxVacancyClauseCount = np.count_nonzero(clausesGEMaxVacancy)
                # Step 10: If the new max is still smaller than the new running sum, shrink further
                while runningSum > maxVacancyClauseCount and dropMaxVacancy and not oIndex == nbVars - 1:  # Only matters if you're not at the end
                    if maxVacancy > 1:
                        maxVacancy-=1
                        # This is a bit more tedious ... but is worth it considering the algorithm is less likely to crash
                        clausesExactlyAtMaxVacancy = clauseVacancies == maxVacancy
                        clausesOverMaxVacancy = clauseVacancies > maxVacancy
                        clausesGEMaxVacancy = np.logical_or(clausesExactlyAtMaxVacancy, clausesOverMaxVacancy)
                        maxVacancyClauseCount = np.count_nonzero(clausesGEMaxVacancy)
                    else:
                        raise TypeError("Generation Failed. Please Try again") # You've reached a point where a single var has
                        # more allocations than the whole number of remaining clauses. Declare failure.
            else:
                rerun = False
                runningSum += allocation
    clausesPost = clauses[clauses < 2 * nbVars]
    splitIndeces = np.cumsum(clauseWidths)
    clausesNonUnif = np.split(clausesPost,splitIndeces)[:-1] # Result of last op is empty (This is a very slow operation)
    return clausesNonUnif # The individual clauses have to be numpy arrays

def prepareNetworkData(formulas,nbVariableList, nbClauseList):  # Where input is an array of the lists of conjunctions of every formula
    # Output is sparse matrix used by TensorFlow
    batchSize = len(formulas)
    variableSum = np.insert(np.cumsum(nbVariableList),0,0) # nb var and nb clauses are given to reduce redundant computation
    totalNbVariables = variableSum[-1]
    conjunctSum = np.insert(np.cumsum(nbClauseList),0,0)
    totalNbConjunctions = conjunctSum[-1]
    indicesConjToLiteral = []
    indicesDisjToConj = []
    for index, formula in enumerate(formulas):
        indicesDisjToConj.extend(([index, conjunctSum[index]+i] for i in range(nbClauseList[index])))  # Can I do better?
        for cIndex, conjunction in enumerate(formula):
            # Convert negation into batch-wide negation
            indicesConjToLiteral.extend(([cIndex+conjunctSum[index],
            (lit//nbVariableList[index])*totalNbVariables + (lit % nbVariableList[index]) + variableSum[index]] for lit in conjunction))
    indicesDisjToConj = np.array(indicesDisjToConj)
    indicesConjToLiteral = np.array(indicesConjToLiteral)
    CLValues = np.ones(indicesConjToLiteral.shape[0])
    DCValues = np.ones(indicesDisjToConj.shape[0])

    # Now define outputs
    disjConj = tf.SparseTensorValue(indices=indicesDisjToConj, values=DCValues, dense_shape=[batchSize,totalNbConjunctions])
    conjLit = tf.SparseTensorValue(indices=indicesConjToLiteral, values=CLValues, dense_shape=[totalNbConjunctions, totalNbVariables*2])
    return disjConj, conjLit, totalNbConjunctions

def createWeightedDNFProblem(nbVars, nbClauses,minCWidth, maxCWidth): #Static method
    try:
        conjunctions = generate(nbVars, nbClauses, minCWidth, maxCWidth)
    except:
        return
    probDist = np.random.uniform(0,1,size=(nbVars))
    # In case generation is successful
    DNFProb = DNFProblem(conjunctions, nbVars, probDist)
    return DNFProb

def prepareNetworkBatchData(listOfDNFProbs):
    conjunctionLists = [prob.clauseList for prob in listOfDNFProbs]
    nbVarList = [prob.nbVariables for prob in listOfDNFProbs]
    nbClauseList = [prob.nbClauses for prob in listOfDNFProbs]
    aggregatePosLitProbDist = np.concatenate([prob.varProbs[:prob.nbVariables] for prob in listOfDNFProbs]) # Dec 20: Remove Dummy Variable
    disjConj, conjLit , nbConjunctions = prepareNetworkData(conjunctionLists, nbVarList, nbClauseList)
    return nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit

def createBatch(nbDistinctFormulas, nbVars, nbClauses, minWidth, maxWidth, distbsPerFormula = 1,
                uniform=False,epsilon=0.1, delta=0.05,R=0,Q=0,uniformisePrivilegedVars = False):
    DNFProblems = []
    solutionsKL = []
    solutionsRA = []
    for i in range(nbDistinctFormulas):
        formula = []
        while True:
            try:
                formula = generate(nbVars=nbVars,nbClauses=nbClauses,minCWidth=minWidth,maxCWidth=maxWidth,R=R,Q=Q,
                                   uniformisePrivilegedVars=uniformisePrivilegedVars)
                break
            except:
                #print(str(i)+")")
                #print("Generation failed, try again")
                pass
            # The while loop is to make the generator retry until success

        for j in range(distbsPerFormula):
            if uniform:
                probDist = np.array([0.5] * nbVars)
            else:
                probDist = np.random.uniform(0, 1, size=nbVars)
            DNFProb = DNFProblem(formula, nbVars, probDist)
            if nbVars < 1000:
                solverBatchSize = 512  # Heuristic (Design Choice)
            elif nbVars < 10000:
                solverBatchSize = 256
            else:
                solverBatchSize = 128
            solutionKL,solutionRA = DNFProb.LTCWithLogAns(epsilon,delta,solverBatchSize)  # Won't use parallel version for now because no correctness guarantee and a bit hacky, also not much faster
            solutionsKL.append(solutionKL)
            solutionsRA.append(solutionRA)
            DNFProblems.append(DNFProb)
    nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit = prepareNetworkBatchData(DNFProblems)
    approxKL = np.array(solutionsKL)
    approxRA = np.array(solutionsRA)
    return nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit, approxKL,approxRA

# Old and practically useless code, should delete at some point
def wholesaleHelper(nbBatches, nbBatchesPerFile,nbDistForm,nbVars,fileName="data"): # To simplify running on servers
    distbsPerFormula = 1 # Won't change these
    clauseNumsStr = ["0.25","0.375","0.5","0.625","0.75"]
    clauseNums = [nbVars // 4, 3 * nbVars // 8, nbVars //2 , 5 * nbVars //8, 3 * nbVars // 4]
    clauseWidthsPre = [[(x,x),(x,x+4),(x,x+9),(x,x+19)] if x!=3 else [(x,x+4),(x,x+9),(x,x+19)] for x in [3,5,8,13,21]]
    clauseWidthsFlat = [x for sublist in clauseWidthsPre for x in sublist]
    denominator = len(clauseWidthsFlat) * len(clauseNums)  # This is uniform. If not, can add distbs implemetation later
    nbBatchesPerDenom = math.ceil(nbBatches / denominator)
    print(nbBatchesPerDenom)
    for index, clauseNum in enumerate(clauseNums):
        for min,max in clauseWidthsFlat:
            fileN = fileName+"_"+str(nbVars)+"_("+str(min)+","+str(max)+")_"+clauseNumsStr[index]
            # Using Default settings for epsilon, delta, R, and Q.
            print("Configuration ("+str(min)+","+str(max)+") - "+clauseNumsStr[index])
            wholesaleCreateBatch(nbBatchesPerDenom,nbBatchesPerFile,nbDistForm,nbVars,clauseNum,min,max
                                 ,distbsPerFormula,fileName=fileN)

def wholesaleHelperPara(nbBatches, nbBatchesPerFile,nbDistForm,nbVars,fileName="data",uniform=False): # To simplify running on servers
    distbsPerFormula = 2 # Won't change these
    clauseNumsStr = ["0.25","0.375","0.5","0.625","0.75"]
    clauseNums = [nbVars // 4, 3 * nbVars // 8, nbVars //2 , 5 * nbVars //8, 3 * nbVars // 4]
    clauseWidthsPre = [[(x,x),(x,x+4),(x,x+9),(x,x+19)] if x!=3 else [(x,x+4),(x,x+9),(x,x+19)] for x in [3,5,8,13,21]]
    clauseWidthsFlat = [x for sublist in clauseWidthsPre for x in sublist]
    denominator = len(clauseWidthsFlat) * len(clauseNums)  # This is uniform. If not, can add distbs implemetation later
    nbBatchesPerDenom = math.ceil(nbBatches / denominator)
    print(nbBatchesPerDenom)
    for index, clauseNum in enumerate(clauseNums):
        for min,max in clauseWidthsFlat:
            fileN = fileName+"_"+str(nbVars)+"_("+str(min)+","+str(max)+")_"+clauseNumsStr[index]
            # Using Default settings for epsilon, delta, R, and Q.
            print("Configuration ("+str(min)+","+str(max)+") - "+clauseNumsStr[index])
            wholesaleCreateBatchPara(nbBatchesPerDenom,nbBatchesPerFile,nbDistForm,nbVars,clauseNum,min,max
                                 ,distbsPerFormula,fileName=fileN,uniform=uniform)

def wholesaleCreateBatchPara(nbBatches, nbBatchesPerFile, nbDistForm, nbVars, nbClauses, minWidth, maxWidth,
                         distbsPerFormula = 1, uniform=False, epsilon=0.1, delta=0.05,R=0,Q=0,fileName="data"):
    nbFiles = math.ceil(nbBatches / nbBatchesPerFile)  # Compute number of files to return
    num_cores = multiprocessing.cpu_count()
    for fileIndex in range(nbFiles):  # Create the necessary number of batches for every file
        fileBatches = []
        print("File "+str(fileIndex))
        innerFor = nbBatchesPerFile
        if fileIndex == nbFiles - 1:
            innerFor = nbBatches % nbBatchesPerFile
        fileBatches = Parallel(n_jobs=num_cores)(delayed(createBatch)(nbDistinctFormulas=nbDistForm,nbVars=nbVars,nbClauses=nbClauses,
                                           minWidth=minWidth,maxWidth=maxWidth,distbsPerFormula=distbsPerFormula,
                                           uniform=uniform,epsilon=epsilon,delta=delta,R=R,Q=Q) for i in range(innerFor))
        with open(fileName+str(fileIndex)+".p","wb") as file:
            pickle.dump(fileBatches,file)#
# Feb 2: Add Automated Chernoff Viability Test (For Q = 0 , R = 0) given a guarantee bound
def chernoffViabilityTest(nbV,nbC,minW,maxW,guarantee): # Return Boolean Value
    # Made for the case R = 0 and Q = 0. Can account for R, but since this is for unpriv vars and R inc. makes exp dec.
    # It's best to do worst-case analysis
    expectedNbSlots = nbC*(minW+maxW)/2 # Assume uniform distb over clause widths
    mu = expectedNbSlots / nbV
    delta = (nbC / mu) - 1 # For informative purposes only
    chernoffBoundExponent = (-(nbC**2)/mu + 2*nbC - mu)/3
    chernoffBound = np.exp(chernoffBoundExponent)
    if chernoffBound * nbV < (1 - guarantee): # Union Bound
        return True
    else:
        return False
# Jan 29: New method to create training data that is more randomised
def createTrainingData(distinctFormulaSizes, numClauseDistribution,clauseWidthDistribution,genUniform = False,
                       epsilon=0.1,delta = 0.05,batchesPerFile = 200,fileName = "TrainingFile",saveDestination = "",
                       logSaveDestination="", distinctDistributionsPerFormula = 4, nodesPerBatch = 2500,uniformisePrivilegedVars=True,
                       chernoffGuarantee = 0.01): # 2500 to account for replication x4 within batch
    # Input parameters
    # New: Upper Bound on number of nodes in a batch
    # Recall that clause width generation is uniform between min and max
    # Step 1: Compute cardinality dict
    # First computational step: Compute Distribution over aggregate and automatically remove unlikely patterns
    clauseNumWidths = {(key, width[0], width[1]): probability1 * probability2 for width, probability2 in
                       clauseWidthDistribution.items() for key, probability1 in numClauseDistribution.items() if
                       key * (width[0] + width[1]) > 2}
    probSum = np.sum(list(clauseNumWidths.values()))
    clauseNumWidthsNorm = {x: y / probSum for x, y in
                           clauseNumWidths.items()}  # Compute normalised probability distribution
    # Now Compute cardinalities: Convention is size, nbClauses, minW, maxW
    preNormCardinalityDict = {
    (size, ceil(size * clauseProperties[0]), clauseProperties[1], clauseProperties[2]): ceil(cardinality * probability)
    for size, cardinality in distinctFormulaSizes.items() for clauseProperties, probability in
    clauseNumWidthsNorm.items() if chernoffViabilityTest(size,ceil(size * clauseProperties[0]),
                                                        clauseProperties[1], clauseProperties[2],chernoffGuarantee)}
    # New (Feb 2): Chernoff viability test
    # Will accept a For Loop, since will only use once:
    preNormSizeCardinalities = {key:0 for key in distinctFormulaSizes.keys()}
    for key, value in preNormCardinalityDict.items():
        preNormSizeCardinalities[key[0]]+=value
    # Normalise by the necessary value when Chernoff Viability Test does some trimming
    cardinalityDict = {key: ceil(value*distinctFormulaSizes[key[0]]/preNormSizeCardinalities[key[0]])
    if preNormSizeCardinalities[key[0]] < distinctFormulaSizes[key[0]] else value for key, value
    in preNormCardinalityDict.items()}
    #  New Part: Node-based batch creation and non-uniform batch generation
    nodeNumList = [properties[0] + properties[1] for properties in cardinalityDict.keys()]  #  Trick to allow duplicates
    nodeNumFullList = np.array(list(cardinalityDict.keys())) # Trick to get full properties structure
    numKeys = len(cardinalityDict)
    cardinalityList = list(cardinalityDict.values())
    indexBridge = np.repeat(np.arange(numKeys,dtype=np.int32),cardinalityList) # Now use index "bridge" to produce our target
    arrayToShuffle = np.repeat(nodeNumList, cardinalityList) # Copy to represent the proportions and then shuffle
    # Do the shuffling in the same manner across bridge and actual indices
    shufflingIndices = np.arange(len(arrayToShuffle))
    np.random.shuffle(shufflingIndices)
    # Retrieve Shuffled
    shuffledNodes = arrayToShuffle[shufflingIndices]
    indexBridgeShuffled = indexBridge[shufflingIndices]
    shuffledFullNodes = nodeNumFullList[indexBridgeShuffled]
    # Split up specification into batches such that no batch has more than nodesPerBatch nodes
    runningSum = 0
    splitIndices = []
    for index, value in enumerate(shuffledNodes): # Inefficient, but only needs to run once
        runningSum += value
        if runningSum >= nodesPerBatch:
            splitIndices.append(index)
            runningSum = value # This is what is preventing me from making this more efficient
    # Now retrieve the elements corresponding to the batches.
    # Batch node size checked :)
    #batchLists = np.split(shuffledNodes, splitIndices) this matches the full nodes. Job Done
    fullPropListsWithEmpty = np.split(shuffledFullNodes, splitIndices) # Batch details now ready
    fullPropLists = [x for x in fullPropListsWithEmpty if len(x) > 0] # February 3 Bug Fix
    # Now use Parallel Computing to create batches
    numBatches = len(fullPropLists) # Count the number of generated batches
    numFiles = ceil(numBatches / batchesPerFile) # Count the number of needed files given the user's file size spec
    numCores = multiprocessing.cpu_count() # Get the number of cores
    for fileIndex in range(numFiles):
        # Create files one by one
        startingParamIndex = fileIndex * batchesPerFile
        innerFor = batchesPerFile if fileIndex < numFiles - 1 else (numBatches % batchesPerFile)
        fileBatches = Parallel(n_jobs=numCores)(
            delayed(createBatchNew)(distbsPerFormula = distinctDistributionsPerFormula,
                                 uniform=genUniform, epsilon=epsilon, delta=delta,
                                formulaProperties = fullPropLists[startingParamIndex+i],
                                uniformisePrivilegedVars=uniformisePrivilegedVars) for i in range(innerFor))
        # Now needed due to change ... flatten the list of lists into a single list
        fileBatches = [batch for batchSet in fileBatches for batch in batchSet]
        with open(saveDestination + fileName + str(fileIndex) + ".p", "wb") as file:
            pickle.dump(fileBatches, file)
        with open(logSaveDestination + "Log for " + fileName + str(fileIndex) + ".txt", "w") as file:
            file.write(str(fullPropLists[startingParamIndex:startingParamIndex+innerFor]))
            file.close()

# Jan 29: Create non-uniform batch generation method
def createBatchNew(formulaProperties, distbsPerFormula = 4, uniform = False, epsilon = 0.1, delta = 0.05,zeroQProb = 0.5, guaranteeBound = 0.2,uniformisePrivilegedVars=True): # Heuristic
    '''No need to specify R and Q from outside for the time being, we will figure out how to randomise these
     (and the probability distribution) in due course '''
    np.random.seed(None) # Very important for parallelisation to avoid the same randomness across all threads
    DNFProblems = []
    solutionsKL = []
    solutionsRA = []
    if uniform and distbsPerFormula > 1:
        print("It is redundant to have duplication for uniform generation. Setting distbsPerFormula to 1")
        distbsPerFormula = 1
    # Now proceed to generate formulas according to user-specified properties
    for indivFormulaProps in formulaProperties:
        nbVars, nbClauses, minWidth, maxWidth = indivFormulaProps
        # Control randomisation to minimise the risk of failure
        expectedExcess = (nbClauses * (minWidth + maxWidth) / 2) - nbVars # Bear in mind that for non-uniform clauses, knowing the exact excess is necessary to guarantee computation
        # Define variables as in derivation
        Q = np.random.uniform(0,1)
        if Q <= zeroQProb:
            Q = 0 # Set Q to value 0 w.p. zeroQProb
        else:
            # Heuristic Lambda
            Q = np.random.exponential(1) % np.log(nbVars)/nbVars # Heuristic: Exponential Distribution with modulus, equivalent to scaled and bounded exponential distribution
            if Q < 1/nbVars:
                Q = 1/nbVars # Avoid problems later in polynomial computations
        nbPrivilegedVars = ceil(Q*nbVars)  # Compute the number of privileged variables
        if Q > 0:
            targetProbPerVariable = (1 - guaranteeBound) / nbPrivilegedVars # Assuming every variable's alloc. Check derivations
            # CANNOT Assume Independence. Can Only Use Union Bound
            coefficient = 1 / targetProbPerVariable - 1
            # Test Chebyshev Upper (One-Sided) Bound
            # Define my solution variables
            alpha = coefficient * np.sqrt(expectedExcess * (nbVars - 1)) / nbVars
            beta = coefficient * np.sqrt(expectedExcess * (Q * nbVars - 1)) / (Q * nbVars)
            gamma = 1 + (expectedExcess / nbVars) - nbClauses + alpha
            tau = expectedExcess * (1 - Q) / (Q * nbVars)
            # Solve the derived quadratic equation (check derivations)
            if beta ** 2 - 4 * gamma * tau > 0: # Only if real root exists
                x_1_max = (- beta + np.sqrt(beta ** 2 - 4 * gamma * tau)) / (2 * tau)
                maxR = x_1_max ** 2 if x_1_max > 0 else 0  # Check notebook for derivations
                if maxR > 1:
                    maxR = 1 # Make sure to keep R within the bound
            else:
                maxR = 0
            if maxR > 0:
                R = maxR  # np.random.uniform(0,maxR) # Should I always pick the max R? but what if we want less repetitive formulae
            else:
                R = 0
        else:
            R = 0
        '''expectation = (1 - R) * expectedExcess / nbVariables + R * expectedExcess / (Q * nbVariables) + 1
        variance = (1 - R) * expectedExcess * (nbVariables - 1) / (nbVariables ** 2) + R * expectedExcess * (
                    Q * nbVariables - 1) / (Q ** 2 * nbVariables ** 2)
        if a > 0:
            print("Probability:" + str(variance / (variance + a ** 2)))
        print(targetProbPerVariable)
        '''
        # Now control randomisation to ensure we have high likelihood of generation success (By Chebyshev One-sided bound)
        formula = []
        while True:
            attempts = 0
            try:
                # How to randomise over R and Q? This has to be calculated
                #print("Attempt")
                formula = generate(nbVars=nbVars, nbClauses=nbClauses, minCWidth=minWidth, maxCWidth=maxWidth,
                                   R=R, Q=Q,uniformisePrivilegedVars=uniformisePrivilegedVars)
                break
            except:
                # print(str(i)+")")
                # print("Generation failed, try again")
                pass
            # The while loop is to make the generator retry until success
        # Now randomise
        subBatches = [] # New Feb 15: Allow Division into Sub-batches
        distributions = []
        if uniform:
            distributions.append(np.array([0.5] * nbVars))
        else:
            distributions.append(np.random.uniform(0, 1, size=nbVars))
        for i in range(1,distbsPerFormula):
            distributions.append((distributions[0]+(i/distbsPerFormula)) % 1)

        for j in range(distbsPerFormula):
            DNFProb = DNFProblem(formula, nbVars, distributions[j])
            if nbVars < 1000:
                solverBatchSize = 512  # Heuristic (Design Choice)
            elif nbVars < 10000:
                solverBatchSize = 256
            else:
                solverBatchSize = 128
            solutionKL, solutionRA = DNFProb.LTCWithLogAns(epsilon, delta,
                                                           solverBatchSize)
            if nbVars > 1000: # New: February 15 --> Separate Batches into sub-batches:
                nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit = prepareNetworkBatchData([DNFProb]) # List of Size 1
                subBatches.append((nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit,
                                   np.array([solutionKL]), np.array([solutionRA])))
            else: # Standard Generation as before
                solutionsKL.append(solutionKL)
                solutionsRA.append(solutionRA)
                DNFProblems.append(DNFProb)
    if nbVars <=1000:  # Bad coding practice ... single element of size 1000 or more is detected, but this isn't clean
        nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit = prepareNetworkBatchData(DNFProblems)
        approxKL = np.array(solutionsKL)
        approxRA = np.array(solutionsRA)
        return [(nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit, approxKL, approxRA)] # List of Size 1
    else:
        # New Part: Return Multiple SubBatches
        return subBatches

def wholesaleCreateBatch(nbBatches, nbBatchesPerFile, nbDistForm, nbVars, nbClauses, minWidth, maxWidth,
                         distbsPerFormula = 1, uniform=False, epsilon=0.1, delta=0.05,R=0,Q=0,fileName="data"):
    nbFiles = math.ceil(nbBatches / nbBatchesPerFile)  # Compute number of files to return
    for fileIndex in range(nbFiles):  # Create the necessary number of batches for every file
        fileBatches = []
        print("File "+str(fileIndex))
        innerFor = nbBatchesPerFile
        if fileIndex == nbFiles - 1:
            innerFor = nbBatches % nbBatchesPerFile
        for batch in range(innerFor):
            if batch % 2 == 0:
                print(str(batch) + " batches of "+str(innerFor)+" generated")
            fileBatches.append(createBatch(nbDistinctFormulas=nbDistForm,nbVars=nbVars,nbClauses=nbClauses,
                                           minWidth=minWidth,maxWidth=maxWidth,distbsPerFormula=distbsPerFormula,
                                           uniform=uniform,epsilon=epsilon,delta=delta,R=R,Q=Q))
        with open(fileName+str(fileIndex)+".p","wb") as file:
            pickle.dump(fileBatches,file)

def printDNF(conjunctions,nbVars):
    for conjunction in conjunctions:
        print("(", end=" ")
        for index, value in enumerate(conjunction):
           negated = (value / nbVars) >= 1
           if (negated):
               print("~", end=" ")
           print("x_"+str(value % nbVars),end = " ")
           if index < len(conjunction) - 1:
               print("^", end=" ")
        print(")OR", end="")
    print()
