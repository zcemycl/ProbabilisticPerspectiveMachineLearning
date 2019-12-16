function documents = transformText(str,startToken,stopToken)

% Split text into characters.
str = strip(replace(str,""," "));

% Add start and stop tokens.
str = startToken + str + stopToken;

% Create tokenized document array.
documents = tokenizedDocument(str,'CustomTokens',[startToken stopToken]);

end
