clear all; close all; clc;
%% Beta Variational Autoencoder
%% Load Data
fprintf('======Loading dSprites datasets======\n')
filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5';
hinfo = hdf5info(filename);
% explain the data structure in hdf5
fprintf('Image data: %s (%dx%dx%d)\n',...
    hinfo.GroupHierarchy.Datasets(1).Name,...
    hinfo.GroupHierarchy.Datasets(1).Dims)
% only load the image data
data0 = hdf5read(hinfo.GroupHierarchy.Datasets(1));
data0 = permute(data0,[2,1,3]);
% shuffle
trainX = reshape(data0,[64,64,1,size(data0,3)]);
%% Settings
settings.latentDim = 10; settings.imageSize = [64,64,1];
settings.miniBatchSize = 64;
settings.lr = 5e-4; settings.gamma = 100;
settings.Cmax = 20; settings.C_stop_iter = 1e5;
settings.max_iter = 1.5e6;

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
idall = [700000];
while ~out
    tic; 
    trainXshuffle = trainX(:,:,:,randperm(size(trainX,4)));
    
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.miniBatchSize+1:i*settings.miniBatchSize;
        XBatch=gpdl(single(trainXshuffle(:,:,:,idx)),'SSCB');

        [GradEn,GradDe] = ...
            dlfeval(@modelGradients,XBatch,paramsEn,paramsDe,...
                    settings,global_iter);

        % Update
        [paramsEn,avgGradientsEncoder,avgGradientsSquaredEncoder] = ...
            adamupdate(paramsEn, GradEn, ...
            avgGradientsEncoder, avgGradientsSquaredEncoder, global_iter,...
            settings.lr);
        [paramsDe,avgGradientsDecoder,avgGradientsSquaredDecoder] = ...
            adamupdate(paramsDe, GradDe, ...
            avgGradientsDecoder, avgGradientsSquaredDecoder, global_iter,...
            settings.lr);
        
        if i == 1 || rem(i,10)==0
            dlx = gpdl(single(trainX(:,:,:,idall)),'SSCB');
            progressplot(dlx,paramsEn,paramsDe,epoch)
        end
        
        if global_iter == settings.max_iter
            out = true;
        end    
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    
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
function [GradEn,GradDe] = modelGradients(dlx,paramsEn,paramsDe,settings,global_iter)
dly = Encoder(dlx,paramsEn);
[zSampled,zMean,zLogvar] = sampling(dly);
dly = Decoder(zSampled,paramsDe);
xPred = sigmoid(dly);

% Loss
% squares = 0.5*(xPred-dlx).^2;
% reconstructionLoss  = mean(sum(squares, [1,2,3]));
reconstructionLoss = mean(sum(-(dlx.*log(xPred)+(1-dlx).*log(1-xPred)),[1,2,3]));
KL = mean(-.5 * sum(1 + zLogvar - zMean.^2 - exp(zLogvar), 1));
if isnan(gatext(reconstructionLoss))
    xlogxp = gatext(dlx.*log(xPred)); 
    x_log_xp = gatext((1-dlx).*log(1-xPred));
    xlogxp(find(isnan(xlogxp))) = 0;
    x_log_xp(find(isnan(x_log_xp))) = 0;
    reconstructionLoss = mean(sum(-(xlogxp+x_log_xp),[1,2,3]));
    reconstructionLoss = dlarray(reconstructionLoss,'SSCB');
end
C = max(min(settings.Cmax*global_iter/settings.C_stop_iter,settings.Cmax),0);
Loss = reconstructionLoss + abs(settings.gamma*(KL-C));

% Gradients
[GradEn,GradDe] = dlgradient(Loss,paramsEn,paramsDe);
end
%% Encoder
function dly = Encoder(dlx,paramsEn)
% convolutions
dly = dlconv(dlx,paramsEn.CNW1,paramsEn.CNb1,...
    'Stride',2,'Padding','same');
dly = relu(dly);
dly = dlconv(dly,paramsEn.CNW2,paramsEn.CNb2,...
    'Stride',2,'Padding','same');
dly = relu(dly);
dly = dlconv(dly,paramsEn.CNW3,paramsEn.CNb3,...
    'Stride',2,'Padding','same');
dly = relu(dly);
dly = dlconv(dly,paramsEn.CNW4,paramsEn.CNb4,...
    'Stride',2,'Padding','same');
dly = relu(dly);
% fully connected
dly = gpdl(reshape(dly,32*4*4,[]),'CB');
dly = fullyconnect(dly,paramsEn.FCW1,paramsEn.FCb1);
dly = relu(dly);
dly = fullyconnect(dly,paramsEn.FCW2,paramsEn.FCb2);
end
%% Decoder
function dly = Decoder(dlx,paramsDe)
% fully connected
dly = fullyconnect(dlx,paramsDe.FCW1,paramsDe.FCb1);
dly = relu(dly);
dly = fullyconnect(dly,paramsDe.FCW2,paramsDe.FCb2);
dly = relu(dly);
% transpose convolution
dly = gpdl(reshape(dly,4,4,32,[]),'SSCB');
dly = dltranspconv(dly,paramsDe.TCW1,paramsDe.TCb1,...
    'Stride',2,'Cropping','same');
dly = relu(dly);
dly = dltranspconv(dly,paramsDe.TCW2,paramsDe.TCb2,...
    'Stride',2,'Cropping','same');
dly = relu(dly);
dly = dltranspconv(dly,paramsDe.TCW3,paramsDe.TCb3,...
    'Stride',2,'Cropping','same');
dly = relu(dly);
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
