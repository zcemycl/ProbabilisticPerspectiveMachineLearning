function [attentionScores, N] = attention(hiddenState, encoderOutputs, weights)

[N, S] = size(encoderOutputs, 2:3);
attentionEnergies = dlarray(zeros( [S N] ));
for tt = 1:S
    % The energy at each time step is the dot product of the hidden state
    % and the learnable attention weights times the encoder output
    attentionEnergies(tt, :) = sum(hiddenState.*(weights*encoderOutputs(:, :, tt)), 1);
end

% Compute softmax scores
attentionScores = softmax(attentionEnergies, 'DataFormat', 'CB');
end