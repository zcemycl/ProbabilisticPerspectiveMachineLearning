clear all; close all; clc;
%% Basic Variational Autoencoder
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
% Parameters for model architecture and training options
settings.latentDim = 10; settings.imageSize = [28,28,1];
settings.numEpochs = 50; settings.miniBatchSize = 512;
settings. lr = 1e-3; 
%% Initialization 
%% Encoder Weights
paramsEn.CNW1 = dlarray(initializeGaussian([3,3,1,32]));
paramsEn.CNb1 = dlarray(zeros(32,1,'single'));
paramsEn.CNW2 = dlarray(initializeGaussian([3,3,32,64]));
paramsEn.CNb2 = dlarray(zeros(64,1,'single'));
paramsEn.FCW1 = dlarray(initializeGaussian(...
    [2*settings.latentDim,64*7*7]));
paramsEn.FCb1 = dlarray(zeros(2*settings.latentDim,1,'single'));
%% Decoder Weights
paramsDe.FCW1 = dlarray(initializeGaussian([64*7*7,...
    settings.latentDim]));
paramsDe.FCb1 = dlarray(zeros(64*7*7,1,'single'));
paramsDe.TCW1 = dlarray(initializeGaussian([3,3,32,64]));
paramsDe.TCb1 = dlarray(zeros(32,1,'single'));
paramsDe.TCW2 = dlarray(initializeGaussian([3,3,1,32]));
paramsDe.TCb2 = dlarray(zeros(1,1,'single'));
%% Train
avgGradientsEncoder = []; avgGradientsSquaredEncoder = [];
avgGradientsDecoder = []; avgGradientsSquaredDecoder = [];
numIterations = floor(size(trainX,4)/settings.miniBatchSize);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    trainXshuffle = trainX(:,:,:,randperm(size(trainX,4)));
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.miniBatchSize+1:i*settings.miniBatchSize;
        XBatch=gpdl(single(trainXshuffle(:,:,:,idx)),'SSCB');
        
        if i==1 || rem(i,20)==0
            idall = [1,2,3,4,5,6,8,14,16,18];
            dlx = gpdl(single(trainX(:,:,:,idall)),'SSCB');
            progressplot(dlx,paramsEn,paramsDe,global_iter)
        end

        [GradEn,GradDe] = ...
            dlfeval(@modelGradients,XBatch,paramsEn,paramsDe);

        % Update
        [paramsEn,avgGradientsEncoder,avgGradientsSquaredEncoder] = ...
            adamupdate(paramsEn, GradEn, ...
            avgGradientsEncoder, avgGradientsSquaredEncoder, global_iter);
        [paramsDe,avgGradientsDecoder,avgGradientsSquaredDecoder] = ...
            adamupdate(paramsDe, GradDe, ...
            avgGradientsDecoder, avgGradientsSquaredDecoder, global_iter);
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.numEpochs
        out = true;
    end    
end
%% Helper functions
%% preprocess
function x = preprocess(x)
x = x/255;
x = reshape(x,28,28,1,[]);
end
%% initialize weights
function parameter = initializeGaussian(parameterSize)
parameter = randn(parameterSize, 'single') .* 0.01;
end
%% modelGradients
function [GradEn,GradDe] = modelGradients(dlx,paramsEn,paramsDe)
dly = Encoder(dlx,paramsEn);
[zSampled,zMean,zLogvar] = sampling(dly);
dly = Decoder(zSampled,paramsDe);
xPred = sigmoid(dly);

% Loss
squares = 0.5*(xPred-dlx).^2;
reconstructionLoss  = sum(squares, [1,2,3]);
KL = -.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
Loss = mean(reconstructionLoss + KL);

% Gradients
[GradEn,GradDe] = dlgradient(Loss,paramsEn,paramsDe);
end
%% Encoder
function dly = Encoder(dlx,paramsEn)
% convolutions
dly = dlconv(dlx,paramsEn.CNW1,paramsEn.CNb1,...
    'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.1);
dly = dlconv(dly,paramsEn.CNW2,paramsEn.CNb2,...
    'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.1);
% fully connected
dly = gpdl(reshape(dly,64*7*7,[]),'CB');
dly = fullyconnect(dly,paramsEn.FCW1,paramsEn.FCb1);

end
%% Decoder
function dly = Decoder(dlx,paramsDe)
% fully connected
dly = fullyconnect(dlx,paramsDe.FCW1,paramsDe.FCb1);
dly = leakyrelu(dly,0.1);
% transpose convolution
dly = gpdl(reshape(dly,7,7,64,[]),'SSCB');
dly = dltranspconv(dly,paramsDe.TCW1,paramsDe.TCb1,...
    'Stride',2,'Cropping','same');
dly = leakyrelu(dly,0.1);
dly = dltranspconv(dly,paramsDe.TCW2,paramsDe.TCb2,...
    'Stride',2,'Cropping','same');
end
%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = gpuArray(dlarray(x,labels));
end
%% sampling latent space
function [zSampled,zMean,zLogvar] = sampling(y)
d = size(y,1)/2;
zMean = y(1:d,:);
zLogvar = y(1+d:end,:);

sz = size(zMean);
epsilon = randn(sz);
sigma = exp(.5 * zLogvar);
z = epsilon .* sigma + zMean;
zSampled = gpdl(z, 'CB');
end
%% progressplot
function progressplot(dlx,paramsEn,paramsDe,count)
dly = Encoder(dlx,paramsEn);
[zSampled,zMean,zLogvar] = sampling(dly);
dly = Decoder(zSampled,paramsDe);
xPred = sigmoid(dly);

fig = gcf;

if ~isempty(fig.Children)
    delete(fig.Children)
end

h = fig;
subplot(1,2,1)
I = imtile(gatext(dlx));
I = rescale(I);
imagesc(I)
set(gca,'visible','off')
pbaspect([1 1 1])

subplot(1,2,2)
I = imtile(gatext(xPred));
I = rescale(I);
imagesc(I)
set(gca,'visible','off')
pbaspect([1 1 1])

drawnow;

% Capture the plot as an image 
frame = getframe(h); 
im = frame2im(frame); 
[imind,cm] = rgb2ind(im,256); 
% Write to the GIF File 
if count == 1 
  imwrite(imind,cm,'VAEmnist.gif','gif', 'Loopcount',inf); 
else 
  imwrite(imind,cm,'VAEmnist.gif','gif','WriteMode','append'); 
end 

end


