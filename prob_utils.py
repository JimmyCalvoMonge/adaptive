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


def setup_optimal_contacts(minmaxBubble, countyToIDs, numIDs):

    idToOptimal, idToCurvature = [0]*numIDs, [0]*numIDs

    """
    TODO:
    This creates an assignment of contact behaviour specs for each node.
    id | optimal_coworker_contacts | willingness_to_change
    .  | ....                      | ....

    willingness_to_change: parameter that will control the utility function curvature.
    optimal_coworkwer_contacts: utility function optimal point (decision purely economical).

    function returns two vectors:

    idToOptimalCts  <- an optimal contact number for each individual in the population. 
    idToCurvature <- a utility function curvature parameter for each individual in the population.

    Algorithm proposed:
    - Contacts selected by individuals are restricted to county limits.
    We use three types of populations within a county:
        1. People with high optimal contact requirements ---> They have a optimal contacts in the upper quartile of the county allowed contact range.
        2. People with middle optimal contact requirements ---> They have a optimal contacts between q25 and q75 of the county allowed contact range.
        3. People with low optimal contact requirements ---> They have a optimal contacts in the lower quartile of the county allowed contact range.

    We set a proportion of high contact individuals:  p_high_contact
    We set a proportion of middle contact individuals:  p_middle_contact
    We set a proportion of low contact individuals:  p_low_contact

    We also must set a utility function curvature for each individual.
    There are two types of individuals with this specs:
        1. Individuals with high curvature (less willing to change contacts in view of disease).
        2. Individuals with low curvature (more willing to change contacts in view of disease).
    
    For each county:

        contact_range_county = range(minmaxBubble[county][0], minmaxBubble[county][1])
        
        # High contacts:
        random sample p_high_contact individuals of the county assign them a contact number in the upper q75 of contact_range_county
        for those, assign a proportion p_high_curvature_high_contact with high curvature and a proportion p_low_curvature_high_contact with low curvature.

        # Middle contacts:
        random sample p_middle_contact individuals of the county that are left and assign them a contact number in the upper q75 of contact_range_county
        for those, assign a proportion p_middle_curvature_middle_contact with middle curvature and a proportion p_low_curvature_middle_contact with low curvature.

        # Low contacts:
        random sample p_low_contact individuals of the county that are left and assign them a contact number in the upper q75 of contact_range_county
        for those, assign a proportion p_low_curvature_low_contact with low curvature and a proportion p_low_curvature_low_contact with low curvature.

        Add these specs to idToOptimal and idToCurvature

    return idToOptimal, idToCurvature

    Note that only the susceptible individuals make an adaptive decision process here.
    (we are back to the non-relapse case, for simplicity). This will be only for susceptibles.

    """

    # Examples (decide):
    p_high_contact = 0.2
    p_high_curvature_high_contact = 0
    p_low_curvature_high_contact = 1
    
    p_middle_contact = 0.5
    p_high_curvature_middle_contact = 0
    p_low_curvature_middle_contact = 1

    p_low_contact = 0.3
    p_high_curvature_low_contact = 0
    p_low_curvature_low_contact = 1

    for county_idx, countyPop in enumerate(countyToIDs):
        
        contact_range_county = range(minmaxBubble[county_idx][0], minmaxBubble[county_idx][1] + 1)

        # High contact individuals:
        # Sample p_high_contact out of the countyPop:
        samp_high_contact = []

        # Of those, assign high_curvature and low curvature:

        # Middle contact individuals:

        # Low contact individuals:

    return idToOptimal, idToCurvature