import random
import numpy as np
from matlab_utils import randsample, randi, setdiff, randperm
from adaptive_SEIR import Adaptive_SEIR


def decide_contacts_susc(currID, numCounty, minmaxBubble,
                         chosenN1_0, chosenN2_0, **kwargs):
    """
    From an initial pool of possible contacts,
    decide the number of contacts a susceptible person will actually
    engage with.

    chosenN1_0: possible contacts of this edge with layer 1 (Coworkers close)
    chosenN2_0: possible contacts of this edge with layer 2 (Coworkers far)

    'default' algorithm:
    if the number of contacts is greater than
    a random number in the county contact boundary,
    then reduce the contacts
    to be within said boundary.

    """
    # Start with all contacts options
    chosenN1 = chosenN1_0
    chosenN2 = chosenN2_0

    # Choose number of contacts to engage with
    sizeBubble = random.randrange(minmaxBubble[numCounty, 0], 
                                  minmaxBubble[numCounty, 1] + 1)

    # If contact options are higher than the number of contacts permited, reduce.
    if (len(chosenN1) + len(chosenN2) > sizeBubble):
        nN = randi(sizeBubble, 1)[0]
        nN1 = min(nN, len(chosenN1))
        nN2 = min(sizeBubble - nN1, len(chosenN2))

        while(nN1 + nN2 < sizeBubble):

            if(nN1 < len(chosenN1)):
                nN1 = nN1 + 1

            if(nN2 < len(chosenN2)):
                nN2 = nN2 + 1

        idx1 = randsample(len(chosenN1), nN1)
        idx2 = randsample(len(chosenN2), nN2)

        chosenN1 = [chosenN1[k] for k in idx1]
        chosenN2 = [chosenN2[k] for k in idx2]

    return chosenN1, chosenN2


def decide_contacts_pooling(currID,
                            newConnections, population,
                            nodeEdges, **kwargs):

    """
    From a base population, perform a random pooling using a predefined
    number of connections (newConnections) to pool.
    """

    multiple = kwargs.get('multiple', 1)
    elegiblesW = [population[k] for k in 
                  randi(len(population), multiple*int(newConnections))]
    teem = [currID] + nodeEdges[0] + nodeEdges[1] + nodeEdges[2]
    elegiblesW = setdiff(elegiblesW, teem)
    newCoworkers = [elegiblesW[t] for t in 
                    randperm(len(elegiblesW), k=min(len(elegiblesW),newConnections))]
    return newCoworkers


def decide_contacts_adaptive_node(optimal, curvature, minmaxBubble, numCounty,
                                  chosenN1_0, chosenN2_0,
                                  totalSday, totalEday, totalIday, totalRday,
                                  mu, gamma, beta, phi, kappa, tau, delta, t_max, steps,
                                  avg_cts_s, avg_cts_e, avg_cts_i, avg_cts_r
                                  ):

    """
    Decide contacts for an individual based on the current status of the disease
    and the individual's economical utility specs.
    Apply adaptive algorithm
    """

    # Start with all contacts options
    chosenN1 = chosenN1_0
    chosenN2 = chosenN2_0

    # Choose number of contacts to engage with
    # In this part we apply a MDP for this individual
    x00 = [totalSday, totalEday, totalIday, totalRday]
    if curvature == 'high':
        nu = 0.75
    else:
        nu = 0.25

    if avg_cts_s == 0:
        avg_cts_s = optimal*0.8
    if avg_cts_e == 0:
        avg_cts_e = optimal*0.7
    if avg_cts_i == 0:
        avg_cts_i = optimal*0.4
    if avg_cts_r == 0:
        avg_cts_r = optimal*1.1

    # Construct utility functions
    min_County, max_County = minmaxBubble[numCounty, 0], minmaxBubble[numCounty, 1]
    const_susc = max_County**2 - 2*optimal*min_County

    # Utility function for individual (susceptible)
    def u_s(a):
        return (2*optimal*a - a**2 + const_susc)**nu

    # Construct other utility functions based on average contacts 
    # of each health class up to this moment.
    const_e = max_County**2 - 2*avg_cts_e*min_County
    const_i = max_County**2 - 2*avg_cts_i*min_County
    const_r = max_County**2 - 2*avg_cts_r*min_County

    def u_e(a):
        return (2*avg_cts_e*a - a**2 + const_e)**nu
    def u_i(a):
            return (2*avg_cts_i*a - a**2 + const_i)**nu
    def u_r(a):
            return (2*avg_cts_r*a - a**2 + const_r)**nu

    instance_adaptive = Adaptive_SEIR(
        mu, gamma, beta, phi, kappa,
        tau, delta, u_s, u_e, u_i, u_r,
        t_max, steps, x00, actions=np.linspace(min_County, max_County, 100),
        logs=False, verbose=False)
    
    cs_policies, _ = instance_adaptive.find_optimal_C_at_time(x00, cs_star=avg_cts_s,
                                                              ce_star=avg_cts_e,
                                                              ci_star=avg_cts_i,
                                                              cz_star=avg_cts_r)
    sizeBubble = int(cs_policies[0][0])

    # If contact options are higher than the number of contacts permited, reduce.
    if (len(chosenN1) + len(chosenN2) > sizeBubble):

        nN = randi(sizeBubble, 1)[0]
        nN1 = min(nN, len(chosenN1))
        nN2 = min(sizeBubble - nN1, len(chosenN2))

        while(nN1 + nN2 < sizeBubble):

            if(nN1 < len(chosenN1)):
                nN1 = nN1 + 1

            if(nN2 < len(chosenN2)):
                nN2 = nN2 + 1

        idx1 = randsample(len(chosenN1), nN1)
        idx2 = randsample(len(chosenN2), nN2)

        chosenN1 = [chosenN1[k] for k in idx1]
        chosenN2 = [chosenN2[k] for k in idx2]

    return chosenN1, chosenN2
