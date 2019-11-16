clear all; close all; clc;
%% Basic Variational Autoencoder
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
settings.latent_dim = 100;
settings.batch_size = 16; settings.image_size = [28,28,1]; 
settings.lrD = 0.001; settings.lrG = 0.001; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 50;

%% Initialization
%% Generator
paramsGen.FCW1 = dlarray(...
    initializeGaussian([256,settings.latent_dim]));
paramsGen.FCb1 = dlarray(zeros(256,1,'single'));
paramsGen.BNo1 = dlarray(zeros(256,1,'single'));
paramsGen.BNs1 = dlarray(ones(256,1,'single'));
paramsGen.FCW2 = dlarray(initializeGaussian([512,256]));
paramsGen.FCb2 = dlarray(zeros(512,1,'single'));
paramsGen.BNo2 = dlarray(zeros(512,1,'single'));
paramsGen.BNs2 = dlarray(ones(512,1,'single'));
paramsGen.FCW3 = dlarray(initializeGaussian([1024,512]));
paramsGen.FCb3 = dlarray(zeros(1024,1,'single'));
paramsGen.BNo3 = dlarray(zeros(1024,1,'single'));
paramsGen.BNs3 = dlarray(ones(1024,1,'single'));
paramsGen.FCW4 = dlarray(initializeGaussian(...
    [prod(settings.image_size),1024]));
paramsGen.FCb4 = dlarray(zeros(prod(settings.image_size)...
    ,1,'single'));

stGen.BN1 = []; stGen.BN2 = []; stGen.BN3 = [];

%% Discriminator
paramsDis.FCW1 = dlarray(initializeGaussian([512,...
     prod(settings.image_size)]));
paramsDis.FCb1 = dlarray(zeros(512,1,'single'));
paramsDis.FCW2 = dlarray(initializeGaussian([256,512]));
paramsDis.FCb2 = dlarray(zeros(256,1,'single'));
paramsDis.FCW3 = dlarray(initializeGaussian([1,256]));
paramsDis.FCb3 = dlarray(zeros(1,1,'single'));

% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
%% Train
% noise = randn([settings.latent_dim,settings.batch_size]);
% [dly,stGem] = Generator(gpdl(noise,'CB'),paramsGen,stGen);
% dly2 = Discriminator(gpdl(single(trainX(:,1:16)),'CB'),paramsDis);

numIterations = floor(size(trainX,2)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    trainXshuffle = trainX(:,randperm(size(trainX,2)));
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        noise = gpdl(randn([settings.latent_dim,...
            settings.batch_size]),'CB');
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=gpdl(single(trainXshuffle(:,idx)),'CB');

        [GradGen,GradDis,stGen] = ...
                dlfeval(@modelGradients,XBatch,noise,...
                paramsGen,paramsDis,stGen);

        % Update Discriminator network parameters
        [paramsDis,avgG.Dis,avgGS.Dis] = ...
            adamupdate(paramsDis, GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            settings.lrD, settings.beta1, settings.beta2);

        % Update Generator network parameters
        [paramsGen,avgG.Gen,avgGS.Gen] = ...
            adamupdate(paramsGen, GradGen, ...
            avgG.Gen, avgGS.Gen, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
        
        if i==1 || rem(i,20)==0
            progressplot(paramsGen,stGen,settings);
        end
        
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.numEpochs
        out = true;
    end    
end
%% Helper Functions
%% preprocess
function x = preprocess(x)
x = x/127.5-1;
x = reshape(x,28*28,[]);
end
%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end
%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = gpuArray(dlarray(x,labels));
end
%% Weight initialization
function parameter = initializeGaussian(parameterSize)
parameter = randn(parameterSize, 'single') .* 0.01;
end
%% Generator
function [dly,st] = Generator(dlx,params,st)
% fully connected
%1
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,0.2);
% if isempty(st.BN1)
%     [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,params.BNs1,...
%         'MeanDecay',0.8);
% else
%     [dly,st.BN1.mu,st.BN1.sig] = batchnorm(dly,params.BNo1,...
%         params.BNs1,st.BN1.mu,st.BN1.sig,'MeanDecay',0.8);
% end
%2
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,0.2);
% if isempty(st.BN2)
%     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,params.BNs2,...
%         'MeanDecay',0.8);
% else
%     [dly,st.BN2.mu,st.BN2.sig] = batchnorm(dly,params.BNo2,...
%         params.BNs2,st.BN2.mu,st.BN2.sig,'MeanDecay',0.8);
% end
%3
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,0.2);
% if isempty(st.BN3)
%     [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,params.BNs3,...
%         'MeanDecay',0.8);
% else
%     [dly,st.BN3.mu,st.BN3.sig] = batchnorm(dly,params.BNo3,...
%         params.BNs3,st.BN3.mu,st.BN3.sig,'MeanDecay',0.8);
% end
%4
dly = fullyconnect(dly,params.FCW4,params.FCb4);
% tanh
dly = tanh(dly);
end
%% Discriminator
function dly = Discriminator(dlx,params)
% fully connected 
%1
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,0.2);
%2
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,0.2);
%3
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,0.2);
% sigmoid
dly = sigmoid(dly);
end
%% modelGradients
function [GradGen,GradDis,stGen]=modelGradients(x,z,paramsGen,...
    paramsDis,stGen)
[fake_images,stGen] = Generator(z,paramsGen,stGen);
d_output_real = Discriminator(x,paramsDis);
d_output_fake = Discriminator(fake_images,paramsDis);

% Loss due to true or not
d_loss = -mean(log(d_output_real)+log(1-d_output_fake));
g_loss = -mean(log(d_output_fake));

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% progressplot
function progressplot(paramsGen,stGen,settings)
r = 5; c = 5;
noise = gpdl(randn([settings.latent_dim,r*c]),'CB');
gen_imgs = Generator(noise,paramsGen,stGen);
gen_imgs = 0.5*reshape(gen_imgs,28,28,[])+0.5;

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

I = imtile(gatext(gen_imgs));
I = rescale(I);
imagesc(I)
title("Generated Images")

drawnow;

end