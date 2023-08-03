function [chosenN1, chosenN2] = decide_contacts_susc( ...
    numCounty, minmaxBubble, chosenN1_0, chosenN2_0)

% Contact decision for susceptibles using minmax Bubble and random integers
% in range. TODO: alter to incorporate adaptive behavior.

sizeBubble = minmaxBubble(numCounty,1) + ...
randi(minmaxBubble(numCounty,2)-minmaxBubble(numCounty,1)+1,1)-1;
if(numel(chosenN1_0)+numel(chosenN2_0) > sizeBubble)
    nN = randi(sizeBubble,1);
    nN1 = min(nN,length(chosenN1_0));
    nN2 = min(sizeBubble-nN1,length(chosenN2_0));
    while(nN1+nN2<sizeBubble)
        if(nN1<length(chosenN1_0))
            nN1 = nN1 + 1;
        end
        if(nN2<length(chosenN2_0))
            nN2 = nN2 + 1;
        end
    end
    chosenN1 = chosenN1_0(randsample(length(chosenN1_0),nN1));
    chosenN2 = chosenN2_0(randsample(length(chosenN2_0),nN2));
else
    chosenN1 = chosenN1_0;
    chosenN2 = chosenN2_0;
end
end