clear all; close all; clc;
%% Conditional Generative Adversarial Network
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
settings.latent_dim = 100;
settings.num_labels = 10;
settings.batch_size = 32; settings.image_size = [28,28,1]; 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 50;

%% Initialization
%% Generator
paramsGen.FCW1 = dlarray(...
    initializeGaussian([256,settings.latent_dim],.02));
paramsGen.FCb1 = dlarray(zeros(256,1,'single'));
paramsGen.EMW1 = dlarray(...
    initializeUniform([settings.latent_dim,...
    settings.num_labels]));
paramsGen.EMb1 = dlarray(zeros(1,settings.num_labels,'single'));
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
     prod(settings.image_size)],.02));
paramsDis.FCb1 = dlarray(zeros(512,1,'single'));
paramsDis.EMW1 = dlarray(...
    initializeUniform([prod(settings.image_size),...
    settings.num_labels]));
paramsDis.EMb1 = dlarray(zeros(1,settings.num_labels,'single'));
paramsDis.BNo1 = dlarray(zeros(1024,1,'single'));
paramsDis.BNs1 = dlarray(ones(1024,1,'single'));
paramsDis.FCW2 = dlarray(initializeGaussian([512,512]));
paramsDis.FCb2 = dlarray(zeros(512,1,'single'));
paramsDis.BNo2 = dlarray(zeros(512,1,'single'));
paramsDis.BNs2 = dlarray(ones(512,1,'single'));
paramsDis.FCW3 = dlarray(initializeGaussian([512,512]));
paramsDis.FCb3 = dlarray(zeros(512,1,'single'));
paramsDis.FCW4 = dlarray(initializeGaussian([1,512]));
paramsDis.FCb4 = dlarray(zeros(1,1,'single'));

stDis.BN1 = []; stDis.BN2 = [];

% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.Gen = []; avgGS.Gen = [];
%% Train
numIterations = floor(size(trainX,2)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX,2));
    trainXshuffle = trainX(:,shuffleid);
    trainYshuffle = trainY(shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        noise = gpdl(randn([settings.latent_dim,...
            settings.batch_size]),'CB');
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=gpdl(single(trainXshuffle(:,idx)),'CB');
        YBatch=gpdl(single(trainYshuffle(idx)),'B');

        [GradGen,GradDis,stGen,stDis] = ...
                dlfeval(@modelGradients,XBatch,YBatch,noise,...
                paramsGen,paramsDis,stGen,stDis,...
                settings);

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
%             if i==1 || (epoch>=0 && i==1) 
%                 h = gcf;
%                 % Capture the plot as an image 
%                 frame = getframe(h); 
%                 im = frame2im(frame); 
%                 [imind,cm] = rgb2ind(im,256); 
%                 % Write to the GIF File 
%                 if epoch == 0
%                   imwrite(imind,cm,'GANmnist.gif','gif', 'Loopcount',inf); 
%                 else 
%                   imwrite(imind,cm,'GANmnist.gif','gif','WriteMode','append'); 
%                 end 
%             end
        end
        
    end

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end
%% Helper Functions
%% preprocess
function x = preprocess(x)
% x = (x-127.5)/127.5;
x = x/255;
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
function parameter = initializeGaussian(parameterSize,sigma)
if nargin < 2
    sigma = 0.05;
end
parameter = randn(parameterSize, 'single') .* sigma;
end
function parameter = initializeUniform(parameterSize,sigma)
if nargin < 2
    sigma = 0.05;
end
parameter = 2*sigma*rand(parameterSize, 'single')-sigma;
end
%% Generator
function [dly,st] = Generator(dlx,labels,params,st)
dly = embedding(dlx,labels,params);
% fully connected
%1
dly = fullyconnect(dly,params.FCW1,params.FCb1);
dly = leakyrelu(dly,0.2);
%2
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,0.2);
%3
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,0.2);
%4
dly = fullyconnect(dly,params.FCW4,params.FCb4);
% tanh
dly = tanh(dly);
end
%% Discriminator
function [dly,st] = Discriminator(dlx,labels,params,st)
dly = embedding(dlx,labels,params);
% fully connected 
%1
dly = fullyconnect(dly,params.FCW1,params.FCb1);
dly = leakyrelu(dly,0.2);
%2
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,0.2);
dly = dropout(dly,.4);
%3
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,0.2);
dly = dropout(dly,.4);
%4
dly = fullyconnect(dly,params.FCW4,params.FCb4);
% sigmoid
dly = sigmoid(dly);
end
%% modelGradients
function [GradGen,GradDis,stGen,stDis]=modelGradients(x,y,z,paramsGen,...
    paramsDis,stGen,stDis,settings)
y0 = randi([0,9],[settings.batch_size,1]);
[fake_images,stGen] = Generator(z,y0,paramsGen,stGen);
d_output_real = Discriminator(x,y,paramsDis,stDis);
[d_output_fake,stDis] = Discriminator(fake_images,y,paramsDis,stDis);

% Loss due to true or not
d_loss = -mean(log(d_output_real+eps)+log(1-d_output_fake+eps));
g_loss = -mean(log(d_output_fake+eps));

% For each network, calculate the gradients with respect to the loss.
GradGen = dlgradient(g_loss,paramsGen,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% progressplot
function progressplot(paramsGen,stGen,settings)
r = 2; c = 5;
labels = gpdl(single([0:9]'),'B');
noise = gpdl(randn([settings.latent_dim,r*c]),'CB');
gen_imgs = Generator(noise,labels,paramsGen,stGen);
gen_imgs = reshape(gen_imgs,28,28,[]);

fig = gcf;
if ~isempty(fig.Children)
    delete(fig.Children)
end

I = imtile(gatext(gen_imgs));
I = rescale(I);
imagesc(I)
title("Generated Images")
colormap gray

drawnow;
end
%% dropout
function dly = dropout(dlx,p)
if nargin < 2
    p = .3;
end
[n,d] = rat(p);
mask = randi([1,d],size(dlx));
mask(mask<=n)=0;
mask(mask>n)=1;
dly = dlx.*mask;

end
%% embedding
function dly = embedding(dlx,labels,params)
% params EM W (latent_dim,num_labels)
%               / (img_elements,num_labels)
%           b (latent_dim,1) (ignore)
%               / (img_elements,1)
maskW = params.EMW1(:,labels+1);
maskb = params.EMb1(:,labels+1);
dly = dlx.*maskW;
% dly = dlx.*maskW+maskb;
end