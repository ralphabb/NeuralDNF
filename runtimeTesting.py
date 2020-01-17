import DNFGen
from DNFProblem import DNFProblem
import numpy as np
import time
from GraphNeuralNet import GGNN
import pickle
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

argParser = argparse.ArgumentParser(description='Conduct Runtime Tests')
argParser.add_argument("-load_mode", type=str2bool,default=False, help="Load or generate formulas")
argParser.add_argument("-run_klm", type=str2bool, default=False,
                       help="Whether to run KLM")
argParser.add_argument("-run_gnn",type=str2bool, default=False, help="Whether to run GNN")
args = argParser.parse_args()
load_mode = args.load_mode  # False -> generate on the fly
run_klm = args.run_klm
run_gnn = args.run_gnn
epsilon = 0.1
delta = 0.05
runs_per_formula = 50  # How many times
clause_widths = [3,5,8,13,21,34]
nb_variables = [1000,2000,5000,10000,15000]
nb_clauses = [0.25,0.375,0.5,0.625,0.75]
test_DNFs = {}
KLM_time_dict = {}
GNN_time_dict = {}
if run_gnn:
    GraphNet = GGNN(nbIterations=32,communicationProtocol=2,embeddingDim=128)
    GraphNet.loadParamsSession(paramLocation=None)
if load_mode:
    with open("runtimeFormulas.p",'rb') as f:
        test_DNFs = pickle.load(f)
for nb_var in nb_variables:
    for nb_cl in nb_clauses:
        for cl_width in clause_widths:
            if nb_cl == 0.25 and cl_width == 3:
                continue  # Ignore this case
            print(str(nb_var)+","+str(nb_cl)+","+str(cl_width))
            if not load_mode:
                while True:
                    try:
                        formula = DNFGen.generate(nbVars=nb_var, nbClauses=int(nb_cl*nb_var),
                                              minCWidth=cl_width, maxCWidth=cl_width)
                        break
                    except:
                        pass

                probDist = np.random.uniform(0, 1, size=nb_var)  # This is a runtime. This doesn't matter here
                DNFProb = DNFProblem(formula, nb_var, probDist)
            else:
                DNFProb = test_DNFs[(nb_var, nb_cl, cl_width)]
            if nb_var < 1000:
                solverBatchSize = 512  # Heuristic (Design Choice)
            elif nb_var < 10000:
                solverBatchSize = 256
            else:
                solverBatchSize = 128
            if run_klm:
                time_before_KLM = time.time()  # KLM Measurement
                for i in range(runs_per_formula):
                    solutionKL, solutionRA = DNFProb.LTCWithLogAns(epsilon, delta,
                                                                   solverBatchSize)  # Won't use parallel version for now because no correctness guarantee and a bit hacky, also not much faster
                time_after_KLM = time.time()
                KLM_time_dict[(nb_var, nb_cl, cl_width)] = ((time_after_KLM - time_before_KLM) / runs_per_formula)
            # Now build the GNN version
            if run_gnn:
                nbConjunctions, aggregatePosLitProbDist, disjConj, conjLit = DNFGen.prepareNetworkBatchData([DNFProb])
                time_before_GNN = time.time()
                for i in range(runs_per_formula):
                    GraphNet.forwardPass(nbConjunctions=nbConjunctions, posLitProbs=aggregatePosLitProbDist,
                                     disjConj=disjConj, conjLit=conjLit,createSession=False)
                time_after_GNN = time.time()
                GNN_time_dict[(nb_var, nb_cl, cl_width)] = ((time_after_GNN - time_before_GNN) / runs_per_formula)
            if not load_mode:
                test_DNFs[(nb_var, nb_cl, cl_width)] = DNFProb
if not load_mode:
    with open("runtimeFormulas.p","wb") as f:
        pickle.dump(test_DNFs, f)
if run_klm:
    with open("runtimesKLM.p","wb") as f:
        pickle.dump(KLM_time_dict, f)
if run_gnn:
    with open("runtimesGNN.p","wb") as f:
        pickle.dump(GNN_time_dict, f)
