function loss = crossEntropyAndSoftmax(dlY, dlT)

offset = max(dlY);
logSoftmax = dlY - offset - log(sum(exp(dlY-offset)));
loss = -sum(dlT.*logSoftmax);

end