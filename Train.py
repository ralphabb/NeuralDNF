from GraphNeuralNet import GGNN
import os, argparse
from tensorflow.python.client import device_lib
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set up the network. Use the default values for now. Now uses communication architecture as opposed to RA loss,
# which we proved has bad side effects
def runTraining():
    argParser = argparse.ArgumentParser(description='Train the Neural#DNF system')
    argParser.add_argument("dataDir", type=str, help="The directory to read training data from")
    argParser.add_argument("-communicationProtocol", type=int, default=2, metavar='',
                           help="Which Communication Protocol to use within the neural network (Default 2)")
    argParser.add_argument("-nbTrainingEpochs", type=int, default=1, metavar='',
                           help="The number of times to run over the data when training (Default 1)")
    argParser.add_argument("-embeddingDimension", type=int, default=128, metavar='',
                           help="The Embedding Dimensionality to use (Default 128)")
    argParser.add_argument('-numIter', type=int, help="The number of message passing iterations to run (Default 8)",
                           default=8, metavar='')
    argParser.add_argument('-feedbackPeriod', type=int, help="The frequency at which the system prints"
                                                             " a progress value (Default once every 50 batches)",
                           default=50, metavar='')
    argParser.add_argument("-lossFileName", type=str, default="losses", metavar='',
                           help="Set a custom losses file name (Default Losses)")
    argParser.add_argument("-resetWeights",type=str2bool,default=True,help="Specify whether to "
        "initialise weights (default) or train starting from existing values", metavar='')
    argParser.add_argument("-logPrints", type=str2bool, default=True, help="Specify whether to "
                        "log the network outputs in a text file (default) or print them out to console", metavar='')
    # Parse the Arguments
    args = argParser.parse_args()
    nbIter = args.numIter
    embDim = args.embeddingDimension
    lossFileName = args.lossFileName
    commProt = args.communicationProtocol
    directory = args.dataDir
    feedPer = args.feedbackPeriod
    resetW = args.resetWeights
    logPr = args.logPrints
    nbEp = args.nbTrainingEpochs
    data = [directory+f for f in os.listdir(directory) if f.endswith(".p") and not f.startswith(lossFileName)]
    # Initialise the Neural Net as specified
    GraphNet = GGNN(nbIterations = nbIter, embeddingDim=embDim,communicationProtocol=commProt)
    GraphNet.train(data,nbEpochs = nbEp, feedback_period=feedPer, logPrints=logPr, resetWeights=resetW)
#print(data)
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if __name__=="__main__":
    runTraining()