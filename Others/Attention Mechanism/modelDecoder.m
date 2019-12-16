function [dlY, context, hiddenState, attentionScores] = modelDecoder(dlX, parameters, context, ...
    hiddenState, encoderOutputs, dropout)

% Embedding
weights = parameters.emb.Weights;
dlX = embedding(dlX, weights);

% RNN input
dlY = cat(1, dlX, context);

% LSTM 1
initialCellState = dlarray(zeros(size(hiddenState)));

inputWeights = parameters.lstm1.InputWeights;
recurrentWeights = parameters.lstm1.RecurrentWeights;
bias = parameters.lstm1.Bias;

dlY = lstm(dlY, hiddenState, initialCellState, inputWeights, ...
    recurrentWeights, bias, 'DataFormat', 'CBT');

if nargin > 5
    % Dropout
    mask = ( rand(size(dlY), 'like', dlY) > dropout );
    dlY = dlY.*mask;
end

% LSTM 2
inputWeights = parameters.lstm2.InputWeights;
recurrentWeights = parameters.lstm2.RecurrentWeights;
bias = parameters.lstm2.Bias;
[~, hiddenState] = lstm(dlY, hiddenState, initialCellState, ...
    inputWeights, recurrentWeights, bias, 'DataFormat', 'CBT');

% Attention
weights = parameters.attn.Weights;
[attentionScores, N] = attention(hiddenState, encoderOutputs, weights);

% Context
encoderOutputs = permute(encoderOutputs, [1 3 2]);
for ii = 1:N
    context(:, ii) = encoderOutputs(:, :, ii)*attentionScores(:, ii);
end

% Fully connect
weights = parameters.fc.Weights;
bias = parameters.fc.Bias;
dlY = weights*cat(1, hiddenState, context) + bias;

end