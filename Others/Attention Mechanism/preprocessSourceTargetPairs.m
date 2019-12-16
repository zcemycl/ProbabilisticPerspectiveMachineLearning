function [sequencesSource, sequencesTarget, encSource, encTarget] = preprocessSourceTargetPairs(data,startToken,stopToken)

% Extract text data.
strSource = data{:,1};
strTarget = data{:,2};

% Create tokenized document arrays.
documentsSource = transformText(strSource,startToken,stopToken);
documentsTarget = transformText(strTarget,startToken,stopToken);

% Create word encodings.
encSource = wordEncoding(documentsSource);
encTarget = wordEncoding(documentsTarget);

% Convert documents to numeric sequences.
sequencesSource = doc2sequence(encSource, documentsSource,'PaddingDirection','none');
sequencesTarget = doc2sequence(encTarget, documentsTarget,'PaddingDirection','none');

end