import time
import pandas as pd
from main_lockdown import mainCall
from main_lockdown_SEIR import mainCall_SEIR
from cProfile import Profile
from pstats import SortKey, Stats
from prob_utils import readProbFile, setupFamilies
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # Profiling:
    # with Profile() as profile:
    #     model_results_sim = mainCall(20,acum, numIDs,numFam,idToCounty,idToFamily,
    # idToAge,familyToIDs,countyToIDs)
    #     (
    #         Stats(profile)
    #         .strip_dirs()
    #         .sort_stats(SortKey.TIME)
    #         .print_stats()
    #     )

    numSim = 1
    T = 100

    # Set up
    
    # S-E-(O,U,H)-R model
    probFILE = './config_data/prob.xlsx'

    # S-E-I-R model
    probFILE = './config_data/prob_small.xlsx'

    print('Reading probability file')
    acum = readProbFile(probFILE)

    print('Creating household network')
    familyFILE = './config_data/setupFamily_reduced.csv'
    [numIDs,numFam,idToCounty,idToFamily,
        idToAge,familyToIDs,countyToIDs] = setupFamilies(familyFILE)


    for sim in range(numSim):

        start_time = time.time()
        prefix = 'SEIR_Adaptive'

        if prefix == '':
            model_results_sim = mainCall(T, acum, numIDs, numFam, idToCounty, idToFamily,
                                        idToAge, familyToIDs, countyToIDs)
        else:
            model_results_sim, avg_cts_s, avg_cts_e, avg_cts_i, avg_cts_r = mainCall_SEIR(T, acum, numIDs, numFam, idToCounty, idToFamily,
                                        idToAge, familyToIDs, countyToIDs)
        end_time = time.time()
        print(f'This iteration of {T} days lasted {(end_time - start_time)/3600} hours.')

        model_results_sim.to_csv(f'./results/model_results_sim_{sim}{prefix}.csv', index=False)
        avg_cts_s.to_csv(f'./results/model_results_sim_{sim}{prefix}_avg_cts_s.csv', index=False)
        avg_cts_e.to_csv(f'./results/model_results_sim_{sim}{prefix}_avg_cts_e.csv', index=False)
        avg_cts_i.to_csv(f'./results/model_results_sim_{sim}{prefix}_avg_cts_i.csv', index=False)
        avg_cts_r.to_csv(f'./results/model_results_sim_{sim}{prefix}_avg_cts_r.csv', index=False)
