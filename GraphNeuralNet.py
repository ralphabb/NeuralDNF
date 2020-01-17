import tensorflow as tf
import pickle
import time
import random
import numpy as np
def load_batches(filePath = "data.p"):
    with open( filePath, "rb" ) as f:
        batch = pickle.load(f)
    return batch
def deltaToTimeString(delta):
    sec = int(delta) % 60
    min = int(delta/60) % 60
    hr = int(delta/3600)
    return str(hr)+":"+str(min).zfill(2)+":"+str(sec).zfill(2)
class GGNN: # Gated Graph Neural Network
    def __init__(self, embeddingDim = 128, nbIterations = 32, learningRate = 10**-5, gradientClip = 0.5, KLDirection = False,
                 communicationProtocol = 1, valuesSaveLocation = None, vizMode = False, widthExtract = False):
        # A) Property and Structure Definitions
        self.embeddingDim = embeddingDim  # Embedding Size
        self.nbIterations = nbIterations  # Number of message passing iterations
        # Added Code to visualise evolution over number of iterations
        self.vizMode = vizMode
        # (I) Literals
        self.posLitProbs = tf.placeholder(tf.float32, shape=[None], name='posLitProbs')  # Convention: positive lit probabilities passed by user
        self.nbVars = tf.shape(self.posLitProbs)[0]
        self.nbLits = self.nbVars*2  # Extract number of literals
        self.negLitProbs = 1 - self.posLitProbs  # Compute negative lit weights
        self.allLitProbs = tf.concat([self.posLitProbs,self.negLitProbs],axis = 0)
        self.litProbsWithDepth = tf.expand_dims(self.allLitProbs, axis = 1)  # Shape [2n,1] ... First n are positive literals
        self.communicationProtocol = communicationProtocol # Jan 28: Added Second Comm Protocol
        # Design Choice: MLP layer number and sizes:
        self.probEmbedLayerSizes = [8,32,self.embeddingDim]
        self.embeddedProbs = createMLP(widths = self.probEmbedLayerSizes, inputs=
        self.litProbsWithDepth, name="litProbEmbed")
        # The embedded probabilities are the initial literal embeddings (equivalent to L_init)
        # Conjunction and Disjunction Initial Embeddings
        self.nbConjunctions = tf.placeholder(tf.int32, shape=[], name='nbConjuncts')
        self.conjunctInitial = tf.get_variable(name="AND_0", initializer=tf.random_normal([1, self.embeddingDim]))
        self.disjunctInitial = tf.get_variable(name="OR_0", initializer=tf.random_normal([1, self.embeddingDim]))
        # Update Mechanism: 3 Cells, for Literal, Conjunct, and Disjunct
        # Design Choice: Layer-norm LSTM (As done in NeuroSAT and TSP)
        self.litUpdate = tf.contrib.rnn.LayerNormBasicLSTMCell(self.embeddingDim,reuse=tf.AUTO_REUSE)  # Tanh activation
        self.conjunctUpdateL = tf.contrib.rnn.LayerNormBasicLSTMCell(self.embeddingDim, reuse=tf.AUTO_REUSE)
        self.disjunctUpdate = tf.contrib.rnn.LayerNormBasicLSTMCell(self.embeddingDim, reuse=tf.AUTO_REUSE)
        if self.communicationProtocol == 2: # In case protocol 2 is used, we need an update based on disjunctions
            self.conjunctUpdateD = tf.contrib.rnn.LayerNormBasicLSTMCell(self.embeddingDim, reuse=tf.AUTO_REUSE)
        # For Width-Based Testing
        self.WidthExtractionSetUp = widthExtract
        # For training:
        self.ConjunctLitAdjacencyMatrix = tf.sparse_placeholder(tf.float32, shape=[None, None], name='ConjLitAdjMat')  # [Conjuncts,Literals]
        self.DisjunctConjunctAdjacencyMatrix = tf.sparse_placeholder(tf.float32, shape=[None, None], name='DisjConjAdjMat') #[Disjuncts, Conjuncts]
        self.CLAdjT = tf.sparse_transpose(self.ConjunctLitAdjacencyMatrix)
        self.DCAdjT = tf.sparse_transpose(self.DisjunctConjunctAdjacencyMatrix)
        self.approximations = tf.placeholder(tf.float32, shape=[None, 2], name='approxSol')  # Mean/StDev of every graph
        self.nbFormulae = tf.shape(self.DisjunctConjunctAdjacencyMatrix)[0]
        if self.WidthExtractionSetUp:  # Call this after all placeholders are defined
            self.setUpWidthExtractionGraph()  # Set up the necessary mechanics for tests if requested
        # [Literals, Conjuncts (equal to number of graphs)
        # Design Choice: structure of message-passing MLPs
        # Misc Declarations
        if valuesSaveLocation is None:
            if self.communicationProtocol == 2:
                self.valuesSaveLocation = "netParams_2/values.ckpt"
            else:
                self.valuesSaveLocation = "netParams/values.ckpt"
        else:
            self.valuesSaveLocation = valuesSaveLocation+"/values.ckpt"
        self.saver = None  # This is initialised later during training or testing
        self.messageLayerSizes = [self.embeddingDim]*4  # Design Choice: Review this later
        self.graphIterate() #Build graph message passing structure
        if not self.vizMode: # Only build the rest of the network if the need arises
            self.computeMeanAndVariance()
            self.groundTruthLogMeans = tf.expand_dims(self.approximations[:, 0], axis=1)
            self.groundTruthStDev = tf.expand_dims(self.approximations[:, 1], axis=1)
            self.computeKLDivergence(KLDirection)  # Just left both as options. Default is approx --> Real,
                # True makes it Real --> approx, which is the standard, but which isn't good for our purposes
            self.optimiser = tf.train.AdamOptimizer(learning_rate=learningRate)
            gradients, variables = zip(*self.optimiser.compute_gradients(self.loss)) #Added gradient clipping following observations
            gradients, _ = tf.clip_by_global_norm(gradients, gradientClip)
            self.train_op = self.optimiser.apply_gradients(zip(gradients, variables), name='minimize')

    def flip(self, litHiddenState):
        return tf.concat([litHiddenState[self.nbVars:(2 * self.nbVars), :], litHiddenState[0:self.nbVars, :]], axis=0)

    def fullMessagePassingProtocol1(self,iterNum, litState, conjState):
        '''
        Message passing in this approach is as follows
        1) Conjunctions update based on literals
        2) Literal update based on conjunctions and their negated literal
        '''
        sumLen = 3
        pleasePrint = False # For diagnostics purposes
        # 1 : L -> C
        LtoC = createMLP(self.messageLayerSizes, litState.h, "literalMessage")
        lc = tf.Print(LtoC,[LtoC],"L to C",summarize= sumLen) if pleasePrint else LtoC
        litMessagesToConj = tf.sparse_tensor_dense_matmul(self.CLAdjT, lc, adjoint_a=True)
        with tf.variable_scope('conjUpdL') as scope:
            _, newConjState = self.conjunctUpdateL(inputs= litMessagesToConj,state=conjState)
        #newH = tf.Print(newConjState.h, [newConjState.h])
        # 2 : C -> L
        CtoL = createMLP(self.messageLayerSizes, newConjState.h, "conjunctMessage")  # Create
        cl = tf.Print(CtoL, [CtoL], "C to L",summarize= sumLen) if pleasePrint else CtoL
        conjMessagesToLits = tf.sparse_tensor_dense_matmul(self.ConjunctLitAdjacencyMatrix,
                                                           cl, adjoint_a=True)
        msgsFromNegatedLiterals = self.flip(litState.h)
        with tf.variable_scope('litUpd') as scope:
            _, newLitState = self.litUpdate(inputs=tf.concat([conjMessagesToLits, msgsFromNegatedLiterals], axis=1),
                                         state=litState)
        return iterNum+1,newLitState,newConjState

    def fullMessagePassingProtocol2(self, iterNum, litState, conjState, disjState,disjStateList):
        '''
        Message passing in this approach is as follows
        1) Conjunctions update based on literals
        2) Disjunctions update based on conjunction
        3) Conjunctions update again based on disjunction
        4) Literals update based on conjunctions
        '''
        sumLen = 3
        pleasePrint = False
        # 1 : L -> C
        LtoC = createMLP(self.messageLayerSizes, litState.h, "literalMessage")
        lc = tf.Print(LtoC, [LtoC], "L to C", summarize=sumLen) if pleasePrint else LtoC
        litMessagesToConj = tf.sparse_tensor_dense_matmul(self.CLAdjT, lc, adjoint_a=True)
        with tf.variable_scope('conjUpdL') as scope:
            _, newConjState = self.conjunctUpdateL(inputs=litMessagesToConj, state=conjState)

        # 2 : C -> D
        CtoD = createMLP(self.messageLayerSizes, conjState.h, "conjunctMessage")
        cd = tf.Print(CtoD, [CtoD], "C to D", summarize=sumLen) if pleasePrint else CtoD
        ConjMessagesToDisj = tf.sparse_tensor_dense_matmul(self.DCAdjT, cd, adjoint_a=True)
        with tf.variable_scope('disjUpd') as scope:
            _, newDisjState = self.disjunctUpdate(inputs=ConjMessagesToDisj, state=disjState)
        if self.vizMode:
            # Keep a running list of disjunction state lists
            disjStateList = disjStateList.write(iterNum, tf.identity(newDisjState.h))
        # 3 : D -> C
        DtoC = createMLP(self.messageLayerSizes, newDisjState.h, "disjunctMessage")
        dc = tf.Print(DtoC, [DtoC], "D to C", summarize=sumLen) if pleasePrint else DtoC
        DisjMessagesToConj = tf.sparse_tensor_dense_matmul(self.DisjunctConjunctAdjacencyMatrix, dc, adjoint_a=True)
        with tf.variable_scope('conjUpdD') as scope:
            _, newnewConjState = self.conjunctUpdateD(inputs=DisjMessagesToConj, state=newConjState)

        # 4 : C -> L
        CtoL = createMLP(self.messageLayerSizes, newnewConjState.h, "conjunctMessage")  # Create
        cl = tf.Print(CtoL, [CtoL], "C to L", summarize=sumLen) if pleasePrint else CtoL
        conjMessagesToLits = tf.sparse_tensor_dense_matmul(self.ConjunctLitAdjacencyMatrix,
                                                           cl, adjoint_a=True)
        msgsFromNegatedLiterals = self.flip(litState.h)
        with tf.variable_scope('litUpd') as scope:
            _, newLitState = self.litUpdate(inputs=tf.concat([conjMessagesToLits, msgsFromNegatedLiterals], axis=1),
                                         state=litState)
        return iterNum + 1, newLitState, newnewConjState, newDisjState, disjStateList

    def while_condProtocol1(self, i, L_state, C_state): #Same as NeuroSAT
        return tf.less(i, self.nbIterations)
    def while_condProtocol2(self, i, L_state, C_state, D_state,disjList):  # Same function, different signature
        return tf.less(i, self.nbIterations)
    def graphIterate(self):
        # Initiate hidden states for all three LSTMs
        self.nbVars = tf.shape(self.posLitProbs)[0]
        self.nbLits = self.nbVars * 2  # Update NbLit to accommodate a changing number of variables
        denom = tf.sqrt(tf.cast(self.embeddingDim, tf.float32)) #NeuroSAT
        litLSTMState = tf.contrib.rnn.LSTMStateTuple(h=tf.div(self.embeddedProbs,denom), c=tf.zeros([self.nbLits, self.embeddingDim]))
        conjH = tf.div(tf.tile(self.conjunctInitial, [self.nbConjunctions, 1]),denom)
        conjLSTMState = tf.contrib.rnn.LSTMStateTuple(h=conjH, c=tf.zeros([self.nbConjunctions, self.embeddingDim]))
        disjH = tf.div(tf.tile(self.disjunctInitial, [self.nbFormulae, 1]),denom)
        disjLSTMState = tf.contrib.rnn.LSTMStateTuple(h=disjH, c=tf.zeros([self.nbFormulae, self.embeddingDim]))

        if self.communicationProtocol == 1:
            # Now compute the update
            _, litLSTMState, conjLSTMState = tf.while_loop(self.while_condProtocol1, self.fullMessagePassingProtocol1,
            [0, litLSTMState, conjLSTMState])
            # Inspired by NeuroSAT code (credit to them)
            self.finalConjState = conjLSTMState.h
            # 3 : C -> D (Think of other iteration structures)
            CtoD = createMLP(self.messageLayerSizes, self.finalConjState, "conjunctMessage")
            ConjMessagesToDisj = tf.sparse_tensor_dense_matmul(self.DCAdjT, CtoD, adjoint_a=True)
            with tf.variable_scope('disjUpd') as scope:
                _, disjLSTMState = self.disjunctUpdate(inputs=ConjMessagesToDisj, state=disjLSTMState)
            # Now compute the final disj state
            self.finalDisjState = disjLSTMState.h  # We use this to compute losses later on

        elif self.communicationProtocol == 2: # Jan 28: New
            # Now compute the update
            disjStateList = tf.TensorArray(size=self.nbIterations, dtype=tf.float32, infer_shape=True)
            _, litLSTMState, conjLSTMState, disjLSTMState, disjStateList = tf.while_loop(self.while_condProtocol2,
                                            self.fullMessagePassingProtocol2, [0, litLSTMState, conjLSTMState, disjLSTMState, disjStateList])
            if self.vizMode: # Should have used scopes when I had the chance
                self.formulaInBatch = tf.placeholder(tf.int32, shape=[], name='formulasInBatch')
                self.disjOverTime = disjStateList.stack()[:,self.formulaInBatch,:] # Define the overall tensor of values we need
                # And an analogous probability value computation network
                preResult = createMLP([32, 8], self.disjOverTime, "predictorMLP")
                # For computing normalised mean
                mean_weights = tf.get_variable(name="meanW", shape=[8, 1],
                                                            initializer=tf.contrib.layers.xavier_initializer())
                mean_biases = tf.get_variable(name="meanB", shape=[], initializer=tf.zeros_initializer())
                        # For computing >=0 standard dev
                stdev_weights = tf.get_variable(name="stdW", shape=[8, 1],
                                                             initializer=tf.contrib.layers.xavier_initializer()),
                stdev_biases = tf.get_variable(name="stdB", shape=[], initializer=tf.zeros_initializer())
                    # Feb 11: Introducing ELU activation functions
                predictedVizMeans = - (
                            tf.nn.elu(tf.matmul(preResult, mean_weights) + mean_biases) + 1)
                    # Mean is negated to ensure that the value returned is always negative, since mu is less than 1 -> log(mu) < 0
                #predictedVizStDev = tf.nn.elu(tf.matmul(preResult, stdev_weights) + stdev_biases) + 1
                self.vizProbs = tf.exp(predictedVizMeans)
            self.finalConjState = conjLSTMState.h
            self.finalDisjState = disjLSTMState.h

    def computeMeanAndVariance(self):
        #We need an MLP
        #Design Choice: widths of MLP layers
        self.preResult = createMLP([32,8],self.finalDisjState, "predictorMLP")
        # For computing normalised mean
        self.mean_weights = tf.get_variable(name="meanW", shape=[8, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.mean_biases = tf.get_variable(name="meanB", shape=[], initializer=tf.zeros_initializer())
            # For computing >=0 standard dev
        self.stdev_weights = tf.get_variable(name="stdW", shape=[8, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.stdev_biases = tf.get_variable(name="stdB", shape=[], initializer=tf.zeros_initializer())
        # Feb 11: Introducing ELU activation functions
        self.predictedLogMeans = - (tf.nn.elu(tf.matmul(self.preResult, self.mean_weights) + self.mean_biases) + 1)
        # Mean is negated to ensure that the value returned is always negative, since mu is less than 1 -> log(mu) < 0
        self.predictedStDev = tf.nn.elu(tf.matmul(self.preResult, self.stdev_weights) + self.stdev_biases) + 1
        #self.predictedLogMeans = tf.log(self.predictedMeans)

    def computeKLDivergence(self,approxIsDenom = False):
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        #Design Choice: We'll compute KL div s.t. sigma 2 is answer by default so that network does not control quadratic denom
        self.meanDiffSquared = tf.square(self.groundTruthLogMeans - self.predictedLogMeans)
        if not approxIsDenom:
            self.numerator = self.meanDiffSquared + tf.square(self.predictedStDev)
            self.denominator = 2*tf.square(self.groundTruthStDev)
            self.KLDiv = tf.log(tf.div(self.groundTruthStDev,self.predictedStDev)) + tf.div(self.numerator,self.denominator) -0.5
        else:  # This is the standard approach in the literature
            self.numerator = self.meanDiffSquared + tf.square(self.groundTruthStDev)
            self.denominator = 2 * tf.square(self.predictedStDev)
            self.KLDiv = tf.log(tf.div(self.predictedStDev, self.groundTruthStDev)) + tf.div(self.numerator,
                                                                                             self.denominator) - 0.5
        self.loss = tf.reduce_mean(self.KLDiv, axis=0)

    def setUpFeedDict(self, nbConjunctions = None, posLitProbs = None, disjConj = None, conjLit = None, approx = None):
        feed_dict1 = {}
        if nbConjunctions is not None:
            feed_dict1[self.nbConjunctions] = nbConjunctions.item()
        if posLitProbs is not None:
            feed_dict1[self.posLitProbs] = posLitProbs
        if disjConj is not None:
            feed_dict1[self.DisjunctConjunctAdjacencyMatrix] = disjConj
        if conjLit is not None:
            feed_dict1[self.ConjunctLitAdjacencyMatrix] = conjLit
        if approx is not None:
            feed_dict1[self.approximations] = approx
        return feed_dict1

    # Feb 14: Define Session and parameter loading separately
    def loadParamsSession(self,paramLocation=None):
        if self.saver is None:
            self.saver = tf.train.Saver(name="Saver")
        self.sess = tf.Session()
        if paramLocation is None:  # If user doesn't specify the parameters to use
            paramLocation = self.valuesSaveLocation
        try:
            self.saver.restore(self.sess, paramLocation)
        except Exception as e:
            print(e)
            self.sess.run(tf.global_variables_initializer())

    def setUpWidthExtractionGraph(self):
        # Necessary Placeholders:
        self.conjLitWidths = tf.sparse.reduce_sum(self.ConjunctLitAdjacencyMatrix , axis=1)
        # Use the fact that all widths are uniform. This is a necessary condition
        self.conjunctionIndices_pre = tf.sparse.reduce_sum(self.DisjunctConjunctAdjacencyMatrix, axis=1)
        self.conjunctionIndices = tf.cumsum(self.conjunctionIndices_pre)
        self.WidthExtractionSetUp = True

    def extractWidthFromBatch(self, disjConj, conjLit, paramLocation=None,createSession=False):
        if self.WidthExtractionSetUp:
            feed_dict1 = self.setUpFeedDict(disjConj=disjConj, conjLit=conjLit)
            if createSession: # Load the parameters only once and separately
                self.loadParamsSession(paramLocation)
            conjLitWidths, conjunctionIndices = self.sess.run([self.conjLitWidths, self.conjunctionIndices],
                                                              feed_dict=feed_dict1)
            width = conjLitWidths.astype(np.int)
            indices_pre = conjunctionIndices[:-1].astype(np.int)
            indices = np.insert(indices_pre, 0, 0)
            widths = width[indices]
            return widths
        else:
            print("Width Extraction Graph not set up. Calling setUpWidthExtractionGraph() ")
            self.setUpWidthExtractionGraph()
            return self.extractWidthFromBatch(disjConj, conjLit, paramLocation, createSession)
    # Feb 14: This is slow because it's reloading parameters, every time
    def forwardPass(self, nbConjunctions, posLitProbs, disjConj, conjLit, paramLocation=None, createSession=True):
        feed_dict1 = self.setUpFeedDict(nbConjunctions = nbConjunctions, posLitProbs = posLitProbs,
                    disjConj = disjConj, conjLit = conjLit)
        if createSession: # Load the parameters only once and separately
            self.loadParamsSession(paramLocation)
        means, variances = self.sess.run([self.predictedLogMeans, self.predictedStDev], feed_dict=feed_dict1)
        return means, variances
    def produceVizProbs(self,nbConjunctions, posLitProbs, disjConj, conjLit, formulaInBatch=0, paramLocation=None, createSession=True):
        if self.vizMode and self.communicationProtocol == 2:
            feed_dict1 = self.setUpFeedDict(nbConjunctions = nbConjunctions, posLitProbs = posLitProbs,
                    disjConj = disjConj, conjLit = conjLit)
            if createSession: # Load the parameters only once and separately
                self.loadParamsSession(paramLocation)
            feed_dict1[self.formulaInBatch] = formulaInBatch
            vizProbs = self.sess.run([self.vizProbs], feed_dict=feed_dict1)
            return vizProbs
        else:
            print("Network not initialised to support visualisation of intermediate states in Comm"
                  " Prot 2. Please Initialise Appropriately")
    def train(self,batch_paths,feedback_period = 50, nbEpochs = 1, resetWeights = True, lossName="losses",logPrints=True):
        # Dec 29: Initialise Log file
        if logPrints:
            if self.communicationProtocol == 1:
                logFile = open("log12.txt","w")
                filePath = "log12.txt"
            else:
                logFile = open("log1234.txt","w")
                filePath = "log1234.txt"
            logFile.write("Log File Started: \r\n")
            logFile.close()
        else:
            filePath = "log.txt"
        losses = []
        self.sess = tf.Session()
        if resetWeights:
            self.sess.run(tf.global_variables_initializer())
        else:
            if self.saver is None:
                self.saver = tf.train.Saver(name="Saver")
            try:
                self.saver.restore(self.sess, self.valuesSaveLocation)  # Restore learned weights
            except:
                self.sess.run(tf.global_variables_initializer())  # If no weights are there, just initialise and run
        batch_total_count = 0
        try:
            tim = time.time()
            if self.saver is None:  # Save At Every feedback period
                self.saver = tf.train.Saver(name="Saver")
            for nbEpochs in range(nbEpochs):
                # Dec 29:
                printOrLog("Epoch " + str(nbEpochs + 1),logPrints,filePath)
                for path in batch_paths:
                    batches = load_batches(filePath=path)  # Load batch with the following path
                    printOrLog("Reading batches from file:" + path,logPrints,filePath)
                    # Shuffle Batches
                    shuffling = [i for i in range(len(batches))]
                    random.shuffle(shuffling)
                    batches = [batches[shuffling[i]] for i in range(len(batches))]
                    for batchIndex, batch in enumerate(batches):
                        # Prepares a new batch
                        nbC, posLit, disjConj, conjLit, approxKL, approxRA = batch
                        approx = approxKL
                        feed_dict1 = self.setUpFeedDict(nbConjunctions=nbC, posLitProbs=posLit, disjConj=disjConj,
                                                        conjLit=conjLit, approx=approx)
                        _, loss= self.sess.run([self.train_op,self.loss], feed_dict1)
                        losses.append(loss)  # Adds the computed loss at the batch to the log being kept
                        if (batchIndex + batch_total_count) % feedback_period == 0:
                            tim2 = time.time()
                            delta = tim2 - tim
                            deltaString = deltaToTimeString(delta)
                            printOrLog(deltaString + "\Loss @Batch " + str(batchIndex + batch_total_count) + ":" + str(
                                loss),logPrints,filePath)  # Prints loss value every specified interval
                            # Dec 29: Write print log
                            self.saver.save(self.sess, self.valuesSaveLocation)  # save intermediate weights
                    batch_total_count += len(batches)
        except KeyboardInterrupt:  # Allows for user to stop training at any time without losing progress
            printOrLog("Training Stopped",logPrints,filePath)
        # Save Network weights at the end of training
        self.saver.save(self.sess, self.valuesSaveLocation)  # save to location
        printOrLog("Weights saved to " + str(self.valuesSaveLocation),logPrints,filePath)
        with open(lossName + ".p", "wb") as f:
            pickle.dump(losses, f)
        return losses
def createMLP(widths, inputs, name):
    nbLayers = len(widths)
    layerOuts = []
    layerOuts.append(inputs) #input is 0th layer, must have at least two dimensions (i.e. Rank 2)
    #Design Choice: Activation will always be ReLU except at the last layer (like NeuroSAT)
    # Dense implementation does not help, because it needs a static size, had to rewrite things myself
    for layerNum in range(nbLayers):
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE): #Allow variable re-use
            layerWeights = tf.get_variable(name=name+"_W_"+str(layerNum+1), shape=[layerOuts[-1].shape[-1],widths[layerNum]],
            initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
            layerBiases = tf.get_variable(name=name+"_b_"+str(layerNum + 1), shape=[widths[layerNum]],
            initializer=tf.zeros_initializer(),dtype=tf.float32)
            preActivationOut = tf.matmul(layerOuts[-1],layerWeights) + layerBiases
            if layerNum < nbLayers - 1:
                postActivationOut = tf.nn.relu(preActivationOut)
            else:
                postActivationOut = preActivationOut
            layerOuts.append(postActivationOut)
    return layerOuts[-1]
def printOrLog(str1,logPrints,filePath="log.txt"):
    if not logPrints:
        print(str1)
    else:
        logFile = open(filePath, "a+")
        logFile.write(str1 + "\r\n")
        logFile.close()  # Keep the file available throughout execution

