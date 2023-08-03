import time
import pandas as pd
from main_lockdown import mainCall
from cProfile import Profile
from pstats import SortKey, Stats
from prob_utils import readProbFile, setupFamilies
probFILE = './config_data/prob.xlsx'
familyFILE = './config_data/setupFamily_reduced.csv'

print('Reading probability file')
acum = readProbFile(probFILE)
print('Creating household network')
[numIDs,numFam,idToCounty,idToFamily,
    idToAge,familyToIDs,countyToIDs] = setupFamilies(familyFILE)

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
    for sim in range(numSim):
        start_time = time.time()
        model_results_sim = mainCall(T, acum, numIDs, numFam, idToCounty, idToFamily,
                                     idToAge, familyToIDs, countyToIDs)
        end_time = time.time()
        print(f'This iteration of {T} days lasted {end_time - start_time} seconds.')
        model_results_sim.to_csv(f'./results/model_results_sim_{sim}.csv', index=False)
        # for val in dict_return_sim.items():
        #     pd.DataFrame(val[1]).to_csv(f'{val[0]}_{sim}.csv')