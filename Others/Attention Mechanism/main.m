clear all; close all; clc;
%% Load Train Data
filename = fullfile("romanNumerals.csv");

options = detectImportOptions(filename, ...
    'TextType','string', ...
    'ReadVariableNames',false);
options.VariableNames = ["Source" "Target"];
options.VariableTypes = ["string" "string"];

data = readtable(filename,options);

idx = randperm(size(data,1),500);
dataTrain = data(idx,:);
dataTest = data;
dataTest(idx,:) = [];

head(dataTrain)

%% Preprocess Data
startToken = "<start>";
stopToken = "<stop>";
[sequencesSource, sequencesTarget, encSource,...
    encTarget] = preprocessSourceTargetPairs(...
    dataTrain,startToken,stopToken);
strSource = "441";
strSource = strip(replace(strSource,""," "));
strSource = startToken + strSource + stopToken

documentSource = tokenizedDocument(strSource,...
    'CustomTokens',[startToken stopToken])

tokens = string(documentSource);
sequenceSource = word2ind(encSource,tokens)

%% Initialize Model Parameters
embeddingDimension = 256;
numHiddenUnits = 200;
dropout = 0.05;

inputSize = encSource.NumWords + 1;
parametersEncoder.emb.Weights = dlarray(randn([embeddingDimension inputSize]));

parametersEncoder.lstm1.InputWeights = dlarray(uniformNoise([4*numHiddenUnits embeddingDimension],1/numHiddenUnits));
parametersEncoder.lstm1.RecurrentWeights = dlarray(uniformNoise([4*numHiddenUnits numHiddenUnits],1/numHiddenUnits));
parametersEncoder.lstm1.Bias = dlarray(uniformNoise([4*numHiddenUnits 1],1/numHiddenUnits));

parametersEncoder.lstm2.InputWeights = dlarray(uniformNoise([4*numHiddenUnits numHiddenUnits],1/numHiddenUnits));
parametersEncoder.lstm2.RecurrentWeights = dlarray(uniformNoise([4*numHiddenUnits numHiddenUnits],1/numHiddenUnits));
parametersEncoder.lstm2.Bias = dlarray(uniformNoise([4*numHiddenUnits 1],1/numHiddenUnits));

outputSize = encTarget.NumWords + 1;
parametersDecoder.emb.Weights = dlarray(randn([embeddingDimension outputSize]));

parametersDecoder.attn.Weights = dlarray(uniformNoise([numHiddenUnits numHiddenUnits],1/numHiddenUnits));

parametersDecoder.lstm1.InputWeights = dlarray(uniformNoise([4*numHiddenUnits embeddingDimension+numHiddenUnits],1/numHiddenUnits));
parametersDecoder.lstm1.RecurrentWeights = dlarray(uniformNoise([4*numHiddenUnits numHiddenUnits],1/numHiddenUnits));
parametersDecoder.lstm1.Bias = dlarray( uniformNoise([4*numHiddenUnits 1],1/numHiddenUnits));

parametersDecoder.lstm2.InputWeights = dlarray(uniformNoise([4*numHiddenUnits numHiddenUnits],1/numHiddenUnits));
parametersDecoder.lstm2.RecurrentWeights = dlarray(uniformNoise([4*numHiddenUnits numHiddenUnits],1/numHiddenUnits));
parametersDecoder.lstm2.Bias = dlarray(uniformNoise([4*numHiddenUnits 1], 1/numHiddenUnits));

parametersDecoder.fc.Weights = dlarray(uniformNoise([outputSize 2*numHiddenUnits],1/(2*numHiddenUnits)));
parametersDecoder.fc.Bias = dlarray(uniformNoise([outputSize 1], 1/(2*numHiddenUnits)));

%% Specify Training Options
miniBatchSize = 32;
numEpochs = 40;
learnRate = 0.002;
gradientThreshold = 5;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
plots = "training-progress";

%% Train Model
sequenceLengthsEncoder = cellfun(@(sequence) size(sequence,2), sequencesSource);
[~,idx] = sort(sequenceLengthsEncoder);
sequencesSource = sequencesSource(idx);
sequencesTarget = sequencesTarget(idx);

if plots == "training-progress"
    figure
    lineLossTrain = animatedline;
    xlabel("Iteration")
    ylabel("Loss")
end

trailingAvgEncoder = [];
trailingAvgSqEncoder = [];

trailingAvgDecoder = [];
trailingAvgSqDecoder = [];

numObservations = numel(sequencesSource);
numIterationsPerEpoch = floor(numObservations/miniBatchSize);

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        [XSource, XTarget, maskSource, maskTarget] = createBatch(sequencesSource(idx), ...
            sequencesTarget(idx), inputSize, outputSize);
        
        % Convert mini-batch of data to dlarray.
        dlXSource = dlarray(XSource);
        dlXTarget = dlarray(XTarget);
        
        % Compute loss and gradients.
        [gradientsEncoder, gradientsDecoder, loss] = dlfeval(@modelGradients, parametersEncoder, ...
            parametersDecoder, dlXSource, dlXTarget, maskSource, maskTarget, dropout);
        
        % Gradient clipping.
        gradientsEncoder = dlupdate(@(w) clipGradient(w,gradientThreshold), gradientsEncoder);
        gradientsDecoder = dlupdate(@(w) clipGradient(w,gradientThreshold), gradientsDecoder);
        
        % Update encoder using adamupdate.
        [parametersEncoder, trailingAvgEncoder, trailingAvgSqEncoder] = adamupdate(parametersEncoder, ...
            gradientsEncoder, trailingAvgEncoder, trailingAvgSqEncoder, iteration, learnRate, ...
            gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update decoder using adamupdate.
        [parametersDecoder, trailingAvgDecoder, trailingAvgSqDecoder] = adamupdate(parametersDecoder, ...
            gradientsDecoder, trailingAvgDecoder, trailingAvgSqDecoder, iteration, learnRate, ...
            gradientDecayFactor, squaredGradientDecayFactor);
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(loss)))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
    
    % Shuffle data.
    idx = randperm(numObservations);
    sequencesSource = sequencesSource(idx);
    sequencesTarget = sequencesTarget(idx);
end

%% Generate Translations
numObservationsTest = 16;
idx = randperm(size(dataTest,1),numObservationsTest);
dataTest(idx,:)

strSource = dataTest{idx,1};
strTarget = dataTest{idx,2};

documentsSource = transformText(strSource,...
    startToken,stopToken);

sequencesSource = doc2sequence(encSource,documentsSource, ...
    'PaddingDirection','right', ...
    'PaddingValue',inputSize);
XSource = cat(3,sequencesSource{:});
XSource = permute(XSource,[1 3 2]);
dlXSource = dlarray(XSource);
[dlZ, hiddenState] = modelEncoder(dlXSource,...
    parametersEncoder);

decoderInput = repmat(word2ind(encTarget,startToken),[1 numObservationsTest]);
decoderInput = dlarray(decoderInput);

context = dlarray(zeros([size(dlZ, 1) numObservationsTest]));
sequencesTranslated = cell(1,numObservationsTest);
attentionScores = cell(1,numObservationsTest);

stopIdx = word2ind(encTarget,stopToken);
stopTranslating = false(1, numObservationsTest);

while ~all(stopTranslating)
    
    % Forward through decoder.
    [dlY, context, hiddenState, attn] = modelDecoder(decoderInput, parametersDecoder, context, ...
        hiddenState, dlZ);
    
    % Loop over observations.
    for i = 1:numObservationsTest
        % Skip already-translated sequences.
        if stopTranslating(i)
            continue
        end
        
        % Update attention scores.
        attentionScores{i} = [attentionScores{i} extractdata(attn(:,i))];
        
        % Predict next time step.
        prob = softmax(dlY(:,i), 'DataFormat', 'CB');
        [~, idx] = max(prob(1:end-1,:), [], 1);
        
        % Set stopTranslating flag when translation done.
        if idx == stopIdx
            stopTranslating(i) = true;
        else
            sequencesTranslated{i} = [sequencesTranslated{i} extractdata(idx)];
            decoderInput(i) = idx;
        end
    end
end

tbl = table;
tbl.Source = strSource;
tbl.Target = strTarget;
tbl.Translated = cellfun(@(sequence) join(ind2word(encTarget,sequence),""),sequencesTranslated)';
tbl

%% Plot Attention Scores
idx = 1;
figure
xlabs = [ind2word(encTarget,sequencesTranslated{idx}) stopToken];
ylabs = string(documentsSource(idx));

heatmap(attentionScores{idx}, ...
    'CellLabelColor','none', ...
    'XDisplayLabels',xlabs, ...
    'YDisplayLabels',ylabs);

xlabel("Translation")
ylabel("Source")
title("Attention Scores")