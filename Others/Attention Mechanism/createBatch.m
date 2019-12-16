function [XSource, XTarget, maskSource, maskTarget] = createBatch(sequencesSource, sequencesTarget, ...
    paddingValueSource, paddingValueTarget)

numObservations = size(sequencesSource,1);
sequenceLengthSource = max(cellfun(@(x) size(x,2), sequencesSource));
sequenceLengthTarget = max(cellfun(@(x) size(x,2), sequencesTarget));

% Initialize masks.
maskSource = false(numObservations, sequenceLengthSource);
maskTarget = false(numObservations, sequenceLengthTarget);

% Initialize mini-batch.
XSource = zeros(1,numObservations,sequenceLengthSource);
XTarget = zeros(1,numObservations,sequenceLengthTarget);

% Pad sequences and create masks.
for i = 1:numObservations
    
    % Source
    L = size(sequencesSource{i},2);
    paddingSize = sequenceLengthSource - L;
    padding = repmat(paddingValueSource, [1 paddingSize]);
    
    XSource(1,i,:) = [sequencesSource{i} padding];
    maskSource(i,1:L) = true;
    
    % Target
    L = size(sequencesTarget{i},2);
    paddingSize = sequenceLengthTarget - L;
    padding = repmat(paddingValueTarget, [1 paddingSize]);
    
    XTarget(1,i,:) = [sequencesTarget{i} padding];
    maskTarget(i,1:L) = true;
end

end