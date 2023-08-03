
function [newCoworkers] = decide_contacts_pooling( ...
    currID, newConnections, population, edges, multiple)
    
        % From a base population, perform a random pooling using a predefined
        % number of connections (newConnections) to pool.
    
        elegiblesW = population(randi(numel(population),1,multiple*newConnections));
        teem = [currID edges{currID,1} edges{currID,2} edges{currID,3}];
        elegiblesW = setdiff(elegiblesW,teem);
        newCoworkers = elegiblesW(randperm(numel(elegiblesW), ...
            min(numel(elegiblesW),newConnections)));
    
    end