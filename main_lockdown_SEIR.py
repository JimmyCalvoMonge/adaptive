import pandas as pd
import numpy as np
from itertools import chain, compress, combinations
from tqdm import tqdm
from matlab_utils import cell, randi, randsample, rand, sort, histc
# from prob_utils import readProbFile, setupFamilies
from contact_utils import decide_contacts_susc, decide_contacts_pooling, decide_contacts_adaptive_node
from prob_utils import setup_optimal_contacts

"""
Networks model only using SEIR and a single age group.
"""

# Main Function:
def mainCall_SEIR(T, acum, numIDs,numFam, idToCounty, idToFamily,
    _, familyToIDs,countyToIDs, numCounties = 15, numInitExposed = 10,
             min_coworkers = 5, delta_coworkers = 10,
             min_sporadic  = 1, delta_sporadic  = 15,
             coefSpor = .5, coefFam = 1, p  = .21,
             pD = .13, pM = .07, pDM = 0.04, pO = .10,
             coefMin = 0.5, pUseMask = .3, minDist  = 10,
             **kwargs):
    
    mu = 0
    gamma = 0.08
    beta = 0.25
    phi = 0
    kappa = 0.15
    tau = 6
    delta = 0.99986
    steps = 100

    nearCountiesFILE = kwargs.get('nearCountiesFILE', './config_data/nearCounties.csv')
    minmaxBubbleFILE = kwargs.get('minmaxBubbleFILE', './config_data/minmaxBubble.csv')
    nearCounties = pd.read_csv(nearCountiesFILE).values
    minmaxBubble = pd.read_csv(minmaxBubbleFILE).values

    # Utility functions node information
    IDtoOptimal, IDtoOptimalCat, IDtoCurvature = setup_optimal_contacts(minmaxBubble, countyToIDs, numIDs)

    #  Data.
    #  daily data (S, E, I, R: totals, D: accum)
    totalS, totalE, totalI, totalR  = [1]*T, [1]*T, [1]*T, [1]*T

    # daily average contact data:
    avg_cts_ = pd.DataFrame({'day': range(T)})
    for cty in range(numCounties):
        avg_cts_[f'county_{cty}_All'] = [0]*T

    avg_cts_s = avg_cts_.copy()
    avg_cts_s[f'county_{cty}_Low'] = [0]*T
    avg_cts_s[f'county_{cty}_Middle'] = [0]*T
    avg_cts_s[f'county_{cty}_High'] = [0]*T
    avg_cts_e = avg_cts_.copy()
    avg_cts_i = avg_cts_.copy()
    avg_cts_r = avg_cts_.copy()

    #  2. Create household network
    #  keep track of...
    familyAdded = [False]*numFam #  ... added families
    notadded = [True]*numIDs     #  ... not added IDs
    #  edge connectivity per layer
    edges = cell(numIDs, 3)
    #  number of contacts per ID: uniform value on [min, min+delta-1]
    dgrLyr2 = randi((min_coworkers, min_coworkers + delta_coworkers), numIDs) 
    dgrLyr3 = randi((min_sporadic, min_sporadic + delta_sporadic), numIDs)
    # dgrLyr2 = [min_coworkers + value - 1 for value in randi(delta_coworkers, numIDs)]
    # dgrLyr3 = [min_sporadic + value - 1 for value in randi(delta_sporadic, numIDs)]
    degrees = np.array([dgrLyr2, dgrLyr3]).T
    #  list of possible contacts (to save time)
    poolPeople = [[]]*numCounties #  lista de elegibles por cantón
    for k in range(numCounties):
        counties_connected = list(np.where(nearCounties[k,:] > 0)[0])
        poolPeople[k] = list(chain.from_iterable([
            countyToIDs[county_index] for county_index in counties_connected]))

    #  3. Initial conditions
    #  states: 0-S 1-E 2-I 3-R 4-D
    stateID  = [0]*numIDs
    timeStateID = [0]*numIDs
    #  numInitExposed initial exposed IDs
    newExposedIDs = randi(numIDs, numInitExposed)
    for k in newExposedIDs:
        stateID[k] = 1 #  S -> E
        timeStateID[k] = 1 #  on day 1

    #  4.
    for day in tqdm(range(T)):

        # Start the day, record information so far
        stateAge_ID_df = pd.DataFrame({
            'stateID': stateID
        })
        totalS[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 0)].shape[0]
        totalE[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 1)].shape[0]
        totalI[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 2)].shape[0]
        totalR[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 3)].shape[0]

        # ==================================================== #
        # ========== Update Contacts for Layers 0 and 1 ====== #
        # update network: include contacts for new exp. people #
        #===================================================== #

        for currID in newExposedIDs:

            # Add family (layer 0):
            numFamily = idToFamily[currID]
            if(not familyAdded[numFamily]): #  check if family has been added before
                familyNodes = familyToIDs[numFamily]  #  IDs for family
                nodesToAdd = list(compress(
                    familyNodes, [notadded[k] for k in familyNodes]))
                for node in nodesToAdd:
                    notadded[node] = False
                allEdges = list(combinations(familyNodes, 2))    #  family edges
                familyAdded[numFamily] = True                    #  mark family as added
                for m in range(len(allEdges)):
                    #  update edges
                    edges[allEdges[m][0]][0].append(allEdges[m][1])
                    edges[allEdges[m][1]][0].append(allEdges[m][0])

            # Add layer 1:
            layer = 1
            newConnections = int(max(0, degrees[currID][layer - 1] - len(edges[currID][layer])))

            if(newConnections > 0):

                #  who is eligible
                population = poolPeople[idToCounty[currID]]
                newCoworkers = decide_contacts_pooling(currID,
                            newConnections, population, edges[currID], multiple=3)

                #  find preexisting nodes on the graph
                nodesToAdd = list(compress(
                    newCoworkers, [notadded[k] for k in newCoworkers]))
                #  update added
                for node in nodesToAdd:
                    notadded[node] = False
                #  add edges
                allEdges = newCoworkers
                #  update neighbors
                for k in range(len(allEdges)):
                    edges[currID][layer].append(allEdges[k]) # end+1 in matlab is equivalent to append in python

                for m in range(len(allEdges)):
                    edges[allEdges[m]][layer].append(currID)

        # ========== Update Contacts for Layer 2 ============= #
        # ==================================================== #

        layer = 2
        vector = list(np.where((np.array(stateID) == 2)|(np.array(stateID) == 3))[0]) # find()
        for j in range(numIDs):
            edges[j][layer] = []
        for currID in vector:

            newConnections = degrees[currID][layer-1]
            population = poolPeople[idToCounty[currID]]
            newCoworkers = decide_contacts_pooling(currID,
                            newConnections, population, edges[currID], multiple=2)

            #  find preexisting nodes on the graph
            nodesToAdd = list(compress(
                    newCoworkers, [notadded[k] for k in newCoworkers]))
            #  update added
            for node in nodesToAdd:
                notadded[node] = False

            #  update neighbors and add edges:
            # Make this person have all these as contacts
            edges[currID][layer] = newCoworkers
            # Add this person as a contact to each coworker:
            for m in range(len(newCoworkers)):
                edges[newCoworkers[m]][layer].append(currID)

        # ==================================================== #
        # ========== probability of infection: ================ #
        # ===================================================== #
        # state 2, 3, 4 can infect their contacts

        newExposedIDs = []   #  restart list of new E
        indexStoE = list(np.where((np.array(stateID) == 2))[0]) # Infected
        StoE = []
        for i in range(3):
            for index in indexStoE:
                StoE = StoE + edges[index][i]
        StoE = list(set(StoE))
        StoE = [ste for ste in StoE if stateID[ste] == 0] #  check susceptibles

        #  set daily #  mask and social distancing
        if (day==20): #  strict measures on day 20
            pUseMask = .9
            minDist  = .7
            minmaxBubble = randi(5, (numCounties, 2))
            minmaxBubble = sort(minmaxBubble, 2)

        if (day==80):  #  release measures on day 80
            pUseMask = .7
            minDist  = .3
            minmaxBubble = randi(15, (numCounties, 2))
            minmaxBubble = sort(minmaxBubble, 2)

        useMask  = randsample(2, numIDs, weights=[1-pUseMask, pUseMask])
        puse_vals = rand(numCounties,1)
        pUseDist = [(minDist + 10*puse_val)/100 for puse_val in puse_vals]

       #  states: 0-S 1-E 2-I 3-R 4-D

        for currID in StoE:

            # Compute product for probability of exposure.

            # 1. Decide effective contacts:
            numCounty  = idToCounty[currID]
            optimalNode = IDtoOptimal[currID]
            curvatureNone = IDtoCurvature[currID]

            chosenN1_0=edges[currID][1]
            chosenN2_0=edges[currID][2]

            # Decide with random pooling and contact cap
            # chosenN1, chosenN2 = decide_contacts_susc(
            #     currID, numCounty, minmaxBubble,
            #     chosenN1_0=chosenN1_0,
            #     chosenN2_0=chosenN2_0)

            # Decide with adaptive algorithm

            if day == 0:
                avg_cts_s_use = 0
                avg_cts_e_use = 0
                avg_cts_i_use = 0
                avg_cts_r_use = 0
            else:
                idx = int(day-1)
                avg_cts_s_use = avg_cts_s[f'county_{numCounty}'].tolist()[idx]
                avg_cts_e_use = avg_cts_e[f'county_{numCounty}'].tolist()[idx]
                avg_cts_i_use = avg_cts_i[f'county_{numCounty}'].tolist()[idx]
                avg_cts_r_use = avg_cts_r[f'county_{numCounty}'].tolist()[idx]

            chosenN1, chosenN2 = decide_contacts_adaptive_node(
                optimal=optimalNode, curvature=curvatureNone,
                minmaxBubble=minmaxBubble, numCounty=numCounty,
                chosenN1_0=chosenN1_0, chosenN2_0=chosenN2_0,
                totalSday=totalS[day], totalEday=totalE[day],
                totalIday=totalI[day], totalRday=totalR[day],
                mu=mu, gamma=gamma, beta=beta, phi=phi, kappa=kappa,
                tau=tau, delta=delta, t_max=T, steps=steps,
                avg_cts_s=avg_cts_s_use,
                avg_cts_e=avg_cts_e_use,
                avg_cts_i=avg_cts_i_use,
                avg_cts_r=avg_cts_r_use)
            
            # Update edges for this id:
            edges[currID][1] = chosenN1
            edges[currID][2] = chosenN2

            # Remove this id from other edges that are in chosenN1_0 and have this id, but they
            # are not in chosenN1 (final contact selection)
            for idd in chosenN1_0:
                if currID in edges[idd][1] and currID not in chosenN1:
                    edges[idd][1].remove(currID)

            for idd in chosenN2_0:
                if currID in edges[idd][2] and currID not in chosenN2:
                    edges[idd][2].remove(currID)

            # Make sure other edges also have this edge in their contact information
            for idd in chosenN1:
                if currID not in edges[idd][1]:
                    edges[idd][1].append(currID)

            for idd in chosenN2:
                if currID not in edges[idd][2]:
                    edges[idd][2].append(currID)

            #  distribution for people that use mask and apply social distancing
            useMask1 = [useMask[k] + 1 for k in chosenN1]
            useMask2 = [useMask[k] + 1 for k in chosenN2]

            useDist1 = histc(rand(len(chosenN1),1),
                                [0, 1 - pUseDist[numCounty], 1])
            useDist2 = histc(rand(len(chosenN2),1),
                                [0, 1 - pUseDist[numCounty], 1])

            states = [stateID[k] for k in edges[currID][0]] # layer 1
            i_inf = [state == 2 for state in states]
            p1 = pO*sum(i_inf)
            risk = (1-coefFam*p)**(coefMin*p1)

            states = [stateID[k] for k in chosenN1] #  layer 2
            i_inf = [state == 2 for state in states]

            for k in range(len(i_inf)):
                if i_inf[k]:
                    if(useMask1[k]==2 and useDist1[k]==2):
                        risk = risk*((1-pDM)**(coefMin))
                    elif(useMask1[k]==2 and useDist1[k]==1):
                        risk = risk*((1-pM)**(coefMin))
                    elif(useMask1[k]==1 and useDist1[k]==2):
                        risk = risk*((1-pD)**(coefMin))
                    else:
                        risk = risk*((1-p)**(coefMin))

            states = [stateID[k] for k in chosenN2] #  layer 3
            i_inf = [state == 2 for state in states]

            for k in range(len(i_inf)):
                if i_inf[k]:
                    if(useMask2[k]==2 and useDist2[k]==2):
                        risk = risk*((1-coefSpor*pDM)**(coefMin))
                    elif(useMask2[k]==2 and useDist2[k]==1):
                        risk = risk*((1-coefSpor*pM)**(coefMin))
                    elif(useMask2[k]==1 and useDist2[k]==2):
                        risk = risk*((1-coefSpor*pD)**(coefMin))
                    else:
                        risk = risk*((1-coefSpor*p)**(coefMin))

            risk = 1 - risk #  probability infection
            rand_num = np.random.uniform(low=0, high=1, size=1)[0]
            # Make exposed using pseudo-random number
            if rand_num < risk:
                newExposedIDs.append(currID)

        # ==================================================== #
        # ========== Transitions ============================= #
        # ==================================================== #
    
        for k in newExposedIDs:
            timeStateID[k] = day #  first mark S->E
            stateID[k] = 1

        elegible = list(np.where((np.array(stateID) > 0) & (np.array(stateID) < 3 ))[0])
        indexD = []

        for ID in elegible:
            state = stateID[ID]
            time = day - timeStateID[ID]

            # Correct this, change prob file: TODO

            if state == 1:

                event = histc(np.random.uniform(low=0, high=1, size=1)[0], acum['E'][(time+1),:])
                event = event[0]

                if (event != 1):
                    stateID[ID] = event
                    timeStateID[ID] = day

                if (event == 2): #  observed case
                    indexD = indexD + [ID]

            elif state == 2:

                event = histc(np.random.uniform(low=0, high=1, size=1)[0],acum['I'][(time+1),:])
                event = event[0]

                if (event != 2):
                    stateID[ID] = event
                    timeStateID[ID] = day

        # ==================================================== #
        # ========== Save Data =============================== #
        # ==================================================== #

        # End of the day, update information obtained from process
        stateAge_ID_df = pd.DataFrame({
            'stateID': stateID,
            'idToCounty': idToCounty,
            'IDtoOptimalCat': IDtoOptimalCat

        })
        totalS[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 0)].shape[0]
        totalE[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 1)].shape[0]
        totalI[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 2)].shape[0]
        totalR[day] = stateAge_ID_df[
            (stateAge_ID_df['stateID'] == 3)].shape[0]

        # Update mean contacts per county/day/state
        for county in range(numCounties):
            mean_cts = []
            for st in [0,1,2,3]:
                ids_county = stateAge_ID_df[
                (stateAge_ID_df['stateID'] == st) & 
                (stateAge_ID_df['idToCounty'] == county)]['idToCounty'].tolist()
                contacts_per_susc_county = [sum([len(edges[idd][lyr]) for lyr in range(3)]) for idd in ids_county]
                if len(contacts_per_susc_county) == 0:
                    mean_cts.append(0)
                else:
                    mean_cts.append(int(np.mean(contacts_per_susc_county)))
            avg_cts_s.loc[day, f'county_{county}_All'] = mean_cts[0]
            avg_cts_e.loc[day, f'county_{county}_All'] = mean_cts[1]
            avg_cts_i.loc[day, f'county_{county}_All'] = mean_cts[2]
            avg_cts_r.loc[day, f'county_{county}_All'] = mean_cts[3]

            mean_cts_cat = []
            for cat in ['Low', 'Middle', 'High']:
                ids_county = stateAge_ID_df[
                (stateAge_ID_df['stateID'] == 0) & 
                (stateAge_ID_df['idToCounty'] == county) & 
                (stateAge_ID_df['IDtoOptimalCat'] == cat)]['idToCounty'].tolist()
                contacts_per_susc_county = [sum([len(edges[idd][lyr]) for lyr in range(3)]) for idd in ids_county]
                if len(contacts_per_susc_county) == 0:
                    mean_cts_cat.append(0)
                else:
                    mean_cts_cat.append(int(np.mean(contacts_per_susc_county)))
            avg_cts_s.loc[day, f'county_{county}_Low'] = mean_cts_cat[0]
            avg_cts_s.loc[day, f'county_{county}_Middle'] = mean_cts_cat[1]
            avg_cts_s.loc[day, f'county_{county}_High'] = mean_cts_cat[2]

    # Data to return: 
    dict_return = {
        'totalS': totalS,
        'totalE': totalE,
        'totalI': totalI,
        'totalR': totalR,
    }

    model_results = pd.DataFrame(dict_return)
    return model_results, avg_cts_s, avg_cts_e, avg_cts_i, avg_cts_r
