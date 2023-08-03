import random
from matlab_utils import randsample, randi, setdiff, randperm


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
    

    sizeBubble = random.randrange(minmaxBubble[numCounty, 0], 
                                  minmaxBubble[numCounty, 1] + 1)
    chosenN1 = chosenN1_0
    chosenN2 = chosenN2_0

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


def decide_contacts_adaptive_node(optimal, curvature, minmaxBubble, chosenN1_0, chosenN2_0, 
                             totalS, totalE, totalR, totalO, totalH, totalU, totalD):

    """
    Decide contacts for an individual based on the current status of the disease
    and the individual's economical utility specs.
    Apply adaptive heuristic
    TODO
    """

    return 