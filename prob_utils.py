import pandas as pd
import numpy as np
from matlab_utils import randperm, cumsum

# Custom Functions:

def readProbFile(file):
    xl = pd.ExcelFile(file)
    sheet_names = xl.sheet_names
    result = {}
    for sheet in sheet_names:
        df = pd.read_excel(file,
                        sheet_name=sheet, engine='openpyxl')
        df = df[[col for col in df.columns if not col.startswith('Unnamed')]]
        df = df.drop(columns=['Dias/E'])
        df['start'] = 0
        col_ = df.pop('start')
        df.insert(0, 'start', col_)
        result[sheet] = df.cumsum(axis=1).values
    return result

def setupFamilies(fileName):
    
    # read demographic data (located in the same directory)
    data = pd.read_csv(fileName)
    houses_county = data['Households'].tolist()
    age_group_01 = data['AG1'].tolist()
    people_county_0 = data['Population'].tolist()
    people_county = [0]*len(people_county_0)
    age_group_02 = data['AG2'].tolist()
    numCounties = data.shape[0] # number of counties
    numFam = sum(houses_county) # number of households

    # allocate arrays
    people_families = [0]*numFam # people per family
    people_age = []
    # create families based on Poisson distribution
    index = 0
    for j in range(numCounties):

        local_houses = houses_county[j]
        local_population = people_county_0[j]
        avrg_per_house = local_population/local_houses
        # create distribution for number of individuals per family
        number_people_per_house = list(
            np.random.poisson(avrg_per_house - 1, local_houses) + 1)
        # store number of people per family
        adv = len(number_people_per_house)
        people_families[index: index + adv] = number_people_per_house
        index = index + adv
        # correct number of people per county (due to the random distribution)
        people_county[j] = sum(number_people_per_house)

        # include age groups
        n1 = int(np.floor(age_group_01[j]*people_county[j]))
        n2 = int(np.floor(age_group_02[j]*people_county[j]))
        n3 = people_county[j] - n1 - n2
        # create number of people per age labeled as 1, 2, 3
        age_groups = [1]*n1 + [2]*n2 + [3]*n3
        # sort age groups
        age_groups_perm = []
        perm_index = randperm(people_county[j])
        for index_2 in perm_index:
            age_groups_perm.append(age_groups[index_2])
        people_age.append(age_groups_perm)

    # accumulate sums for families and counties
    cumsum_people_per_family = [0] + cumsum(people_families)
    cumsum_people_per_county = [0] + cumsum(people_county)
    numIDs = int(cumsum_people_per_county[-1]) # total population
    # allocate arrays
    idToAge     = [0]*cumsum_people_per_family[-1]
    idToCounty  = [0]*cumsum_people_per_county[-1]
    countyToIDs = []
    idToFamily  = [0]*cumsum_people_per_family[-1]
    familyToIDs = []
    # fill idToCounty, idToAge, countyToIDs
    index = 0
    adv_total = 0
    for j in range(numCounties):
        range_ = range(cumsum_people_per_county[j], cumsum_people_per_county[j+1])
        for t in range_:
            idToCounty[t] = j
        countyToIDs.append(list(range_))
        adv = len(people_age[j])
        adv_total = adv_total + adv
        for k in range(index, index+adv):
            idToAge[k] = people_age[j][k - index]
        index = index + adv
    # fill idToFamily, familyToIDs
    for j in range(numFam):
        range_ = range(cumsum_people_per_family[j], cumsum_people_per_family[j+1])
        for t in range_:
            idToFamily[t] = j
        familyToIDs.append(list(range_))

    return numIDs, numFam, idToCounty, idToFamily, idToAge, familyToIDs, countyToIDs
