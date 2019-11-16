clear all; close all; clc;
%% Beta Variational Autoencoder
%% Load Data
load('data.mat')
trainX = preprocess(data);
%% Settings
settings.latentDim = 10; settings.imageSize = [64,64,1];
settings.numEpochs = 100000; settings.miniBatchSize = 64;
settings. lr = 5e-4; settings.beta = 150;


%% Initialization
%% Encoder Weight
paramsEn.CNW1 = dlarray(initializeGaussian([4,4,1,32]));
paramsEn.CNb1 = dlarray(zeros(32,1,'single'));
paramsEn.CNW2 = dlarray(initializeGaussian([4,4,32,32]));
paramsEn.CNb2 = dlarray(zeros(32,1,'single'));
paramsEn.CNW3 = dlarray(initializeGaussian([4,4,32,32]));
paramsEn.CNb3 = dlarray(zeros(32,1,'single'));
paramsEn.CNW4 = dlarray(initializeGaussian([4,4,32,32]));
paramsEn.CNb4 = dlarray(zeros(32,1,'single'));
paramsEn.FCW1 = dlarray(initializeGaussian([256,32*4*4]));
paramsEn.FCb1 = dlarray(zeros(256,1,'single'));
paramsEn.FCW2 = dlarray(initializeGaussian(...
    [2*settings.latentDim,256]));
paramsEn.FCb2 = dlarray(zeros(2*settings.latentDim,1,'single'));
%% Decoder Weight
paramsDe.FCW1 = dlarray(initializeGaussian([256,...
    settings.latentDim]));
paramsDe.FCb1 = dlarray(zeros(256,1,'single'));
paramsDe.FCW2 = dlarray(initializeGaussian([32*4*4,256]));
paramsDe.FCb2 = dlarray(zeros(32*4*4,1,'single'));
paramsDe.TCW1 = dlarray(initializeGaussian([4,4,32,32]));
paramsDe.TCb1 = dlarray(zeros(32,1,'single'));
paramsDe.TCW2 = dlarray(initializeGaussian([4,4,32,32]));
paramsDe.TCb2 = dlarray(zeros(32,1,'single'));
paramsDe.TCW3 = dlarray(initializeGaussian([4,4,32,32]));
paramsDe.TCb3 = dlarray(zeros(32,1,'single'));
paramsDe.TCW4 = dlarray(initializeGaussian([4,4,1,32]));
paramsDe.TCb4 = dlarray(zeros(1,1,'single'));
%% Train
avgGradientsEncoder = []; avgGradientsSquaredEncoder = [];
avgGradientsDecoder = []; avgGradientsSquaredDecoder = [];
numIterations = floor(size(trainX,4)/settings.miniBatchSize);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    trainXshuffle = trainX(:,:,:,randperm(size(trainX,4)));
    if rem(epoch,10)==0
        idall = [0:15]*16+[1:16];
        dlx = gpdl(single(trainX(:,:,:,idall)),'SSCB');
        progressplot(dlx,paramsEn,paramsDe,epoch)
    end
    
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.miniBatchSize+1:i*settings.miniBatchSize;
        XBatch=gpdl(single(trainXshuffle(:,:,:,idx)),'SSCB');

        [GradEn,GradDe] = ...
            dlfeval(@modelGradients,XBatch,paramsEn,paramsDe,settings);

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
x = reshape(x,64,64,1,[]);
end
%% initialize weights
function parameter = initializeGaussian(parameterSize)
parameter = randn(parameterSize, 'single') .* 0.01;
end
%% modelGradients
function [GradEn,GradDe] = modelGradients(dlx,paramsEn,paramsDe,settings)
dly = Encoder(dlx,paramsEn);
[zSampled,zMean,zLogvar] = sampling(dly);
dly = Decoder(zSampled,paramsDe);
xPred = sigmoid(dly);

% Loss
% squares = 0.5*(xPred-dlx).^2;
% reconstructionLoss  = sum(squares, [1,2,3]);
reconstructionLoss = mean(sum(-(dlx.*log(xPred)+(1-dlx).*log(1-xPred)),[1,2,3]));
KL = -.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1);
if isnan(gatext(reconstructionLoss))
    xlogxp = gatext(dlx.*log(xPred)); 
    x_log_xp = gatext((1-dlx).*log(1-xPred));
    xlogxp(find(isnan(xlogxp))) = 0;
    x_log_xp(find(isnan(x_log_xp))) = 0;
    reconstructionLoss = mean(sum(-(xlogxp+x_log_xp),[1,2,3]));
    reconstructionLoss = dlarray(recon_loss,'SSCB');
end

Loss = mean(reconstructionLoss + settings.beta*KL);

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
dly = dlconv(dly,paramsEn.CNW3,paramsEn.CNb3,...
    'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.1);
dly = dlconv(dly,paramsEn.CNW4,paramsEn.CNb4,...
    'Stride',2,'Padding','same');
dly = leakyrelu(dly,0.1);
% fully connected
dly = gpdl(reshape(dly,32*4*4,[]),'CB');
dly = fullyconnect(dly,paramsEn.FCW1,paramsEn.FCb1);
dly = leakyrelu(dly,0.1);
dly = fullyconnect(dly,paramsEn.FCW2,paramsEn.FCb2);
end
%% Decoder
function dly = Decoder(dlx,paramsDe)
% fully connected
dly = fullyconnect(dlx,paramsDe.FCW1,paramsDe.FCb1);
dly = leakyrelu(dly,0.1);
dly = fullyconnect(dly,paramsDe.FCW2,paramsDe.FCb2);
dly = leakyrelu(dly,0.1);
% transpose convolution
dly = gpdl(reshape(dly,4,4,32,[]),'SSCB');
dly = dltranspconv(dly,paramsDe.TCW1,paramsDe.TCb1,...
    'Stride',2,'Cropping','same');
dly = leakyrelu(dly,0.1);
dly = dltranspconv(dly,paramsDe.TCW2,paramsDe.TCb2,...
    'Stride',2,'Cropping','same');
dly = leakyrelu(dly,0.1);
dly = dltranspconv(dly,paramsDe.TCW3,paramsDe.TCb3,...
    'Stride',2,'Cropping','same');
dly = leakyrelu(dly,0.1);
dly = dltranspconv(dly,paramsDe.TCW4,paramsDe.TCb4,...
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
I = imtile(cat(4,gatext(dlx),gatext(xPred)),'GridSize',[4,8]);
I = rescale(I);
imagesc(I)
set(gca,'visible','off')
pbaspect([2 1 1])

drawnow;
% 
% % Capture the plot as an image 
% frame = getframe(h); 
% im = frame2im(frame); 
% [imind,cm] = rgb2ind(im,256); 
% % Write to the GIF File 
% if count == 1 
%   imwrite(imind,cm,'VAEmnist.gif','gif', 'Loopcount',inf); 
% else 
%   imwrite(imind,cm,'VAEmnist.gif','gif','WriteMode','append'); 
% end 

end