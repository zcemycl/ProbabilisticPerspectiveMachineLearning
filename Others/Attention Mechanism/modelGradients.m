function [gradientsEncoder, gradientsDecoder, maskedLoss] = modelGradients(parametersEncoder, ...
    parametersDecoder, dlXSource, dlXTarget, maskSource, maskTarget, dropout)

% Forward through encoder.
[dlZ, hiddenState] = modelEncoder(dlXSource, parametersEncoder, maskSource);

% Get parameter sizes.
[miniBatchSize, sequenceLength] = size(dlXTarget,2:3);
sequenceLength = sequenceLength - 1;
numHiddenUnits = size(dlZ,1);

% Initialize context vector.
context = dlarray(zeros([numHiddenUnits miniBatchSize]));

% Initialize loss.
loss = dlarray(zeros([miniBatchSize sequenceLength]));

% Get first time step for decoder.
decoderInput = dlXTarget(:,:,1);

% Choose whether to use teacher forcing.
doTeacherForcing = rand < 0.5;

if doTeacherForcing
    for t = 1:sequenceLength
        % Forward through decoder.
        [dlY, context, hiddenState] = modelDecoder(decoderInput, parametersDecoder, context, ...
            hiddenState, dlZ, dropout);
        
        % Update loss.
        dlT = dlarray(oneHot(dlXTarget(:,:,t+1), size(dlY,1)));
        loss(:,t) = crossEntropyAndSoftmax(dlY, dlT);
        
        % Get next time step.
        decoderInput = dlXTarget(:,:,t+1);
    end
else
    for t = 1:sequenceLength
        % Forward through decoder.
        [dlY, context, hiddenState] = modelDecoder(decoderInput, parametersDecoder, context, ...
            hiddenState, dlZ, dropout);
        
        % Update loss.
        dlT = dlarray(oneHot(dlXTarget(:,:,t+1), size(dlY,1)));
        loss(:,t) = crossEntropyAndSoftmax(dlY, dlT);
        
        % Greedily update next input time step.
        prob = softmax(dlY,'DataFormat','CB');
        [~, decoderInput] = max(prob,[],1);
    end
end

% Determine masked loss.
maskedLoss = sum(sum(loss.*maskTarget(:,2:end))) / miniBatchSize;

% Update gradients.
[gradientsEncoder, gradientsDecoder] = dlgradient(maskedLoss, parametersEncoder, parametersDecoder);

% For plotting, return loss normalized by sequence length.
maskedLoss = extractdata(maskedLoss) ./ sequenceLength;

end