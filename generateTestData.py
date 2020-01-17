import DNFGen,  argparse
from math import ceil

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
argParser = argparse.ArgumentParser(description='Generate Training Data for Neural#DNF')
argParser.add_argument("-divisionNb",type=int,default=1,help="The inverse of the fraction of the generation task this process should take on. Useful for running multiple concurrent generation tasks", metavar='')
argParser.add_argument("-genUniform",type=str2bool, metavar='', default= False, help="Option to generate uniform distributions. Default False")
argParser.add_argument('-epsilon',type=float,default = 0.1, help="The KLM Epsilon value used for computing scores (Default 0.1)", metavar='')
argParser.add_argument("-delta",type=float,default=0.05, help="The KLM Delta value used for computing scores (Default 0.05)", metavar='')
argParser.add_argument("-batchesPerFile",type=int,default=200, help="The number of batches per file", metavar='')
argParser.add_argument("-fileName", type=str, default="Data", help="Set a custom output file name", metavar='')
argParser.add_argument("-logSaveDestination", type=str, default="./", help="Save Location for log files", metavar='')
argParser.add_argument("-saveDestination", type=str, default="./", help="Save Location for output files", metavar='')
argParser.add_argument("-distinctDistbPerFormula",type=int,default=4,help=" Number of Distinct Distributions per formula", metavar='')
argParser.add_argument("-nodesPerBatch", type=int,default=2500,help="Number of Nodes (conjunction and literal) to have per batch", metavar='')
argParser.add_argument("-uniformisePrivilegedVars", type=str2bool,default=True,help="The option to uniformise the privileged variables during generation (Default True)", metavar='')
argParser.add_argument('-chernoffGuarantee',type=float,default = 0.01, help="The Maximum probability provided by the Chernoff Bound for which a data gen instance could fail (Default 0.01)", metavar='')
# Parse the Arguments
args = argParser.parse_args()
divisionNb = args.divisionNb
genUniform = args.genUniform
epsilon = args.epsilon
delta = args.delta
batchesPerFile = args.batchesPerFile
fileName = args.fileName
saveDest = args.saveDestination
logSaveDest = args.logSaveDestination
distinctDistPerForm = args.distinctDistbPerFormula
nodesPerBatch = args.nodesPerBatch
unifPV = args.uniformisePrivilegedVars
chernoffG = args.chernoffGuarantee
# Prepare the proportions, but as a list. This will enable separation of the test set
dFSs = [{50:3000}, {100:2000}, {250:1600}, {500: 1200}, {750:1000}, {1000:800}, {2500: 600}, {5000: 300}]
nCD = {0.25: 0.2, 0.375: 0.2, 0.5: 0.2, 0.625: 0.2, 0.75: 0.2}
cWD = {(3, 3): 1.0 / 6, (5, 5): 1.0 / 6, (8, 8): 1.0 / 6, (13, 13): 1.0 / 6, (21, 21): 1.0 / 6, (34, 34): 1.0 / 6}
for dFS in dFSs:
    fileNameD = fileName+"_"+str(list(dFS.keys())[0])
    print(fileNameD)
    dFSdiv = {key:ceil(val/divisionNb) for key,val in dFS.items()}
    print(dFSdiv)
    #And now build the command
    DNFGen.createTrainingData(distinctFormulaSizes = dFSdiv,numClauseDistribution=nCD,clauseWidthDistribution=cWD,
                            genUniform = genUniform, epsilon=epsilon,delta = delta,batchesPerFile = batchesPerFile,
                            fileName = fileNameD, saveDestination = saveDest, logSaveDestination=logSaveDest,
                            distinctDistributionsPerFormula = distinctDistPerForm, nodesPerBatch = nodesPerBatch,
                              uniformisePrivilegedVars=unifPV,chernoffGuarantee=chernoffG)

