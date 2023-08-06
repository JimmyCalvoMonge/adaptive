import time
import pandas as pd
from main_lockdown import mainCall
from main_lockdown_SEIR import mainCall_SEIR
from cProfile import Profile
from pstats import SortKey, Stats
from prob_utils import readProbFile, setupFamilies


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
        prefix = 'SEIR'

        if prefix == '':
            model_results_sim = mainCall(T, acum, numIDs, numFam, idToCounty, idToFamily,
                                        idToAge, familyToIDs, countyToIDs)
        else:
            model_results_sim = mainCall_SEIR(T, acum, numIDs, numFam, idToCounty, idToFamily,
                                        idToAge, familyToIDs, countyToIDs)
        end_time = time.time()
        print(f'This iteration of {T} days lasted {end_time - start_time} seconds.')
        model_results_sim.to_csv(f'./results/model_results_sim_{sim}{prefix}.csv', index=False)
