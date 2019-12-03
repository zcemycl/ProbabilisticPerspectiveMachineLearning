clear all; close all; clc;
%% Adversarial AutoEncoder
%% Load Data
load('mnistAll.mat')
trainX = preprocess(mnist.train_images); 
trainY = mnist.train_labels;
testX = preprocess(mnist.test_images); 
testY = mnist.test_labels;
%% Settings
settings.latent_dim = 10;
settings.batch_size = 32; settings.image_size = [28,28,1]; 
settings.lrD = 0.0002; settings.lrG = 0.0002; settings.beta1 = 0.5;
settings.beta2 = 0.999; settings.maxepochs = 50;
%% Initialization
%% Encoder
paramsEn.FCW1 = dlarray(initializeGaussian([512,...
     prod(settings.image_size)],.02));
paramsEn.FCb1 = dlarray(zeros(512,1,'single'));
paramsEn.FCW2 = dlarray(initializeGaussian([512,512]));
paramsEn.FCb2 = dlarray(zeros(512,1,'single'));
paramsEn.FCW3 = dlarray(initializeGaussian([2*settings.latent_dim,512]));
paramsEn.FCb3 = dlarray(zeros(2*settings.latent_dim,1,'single'));
%% Decoder
paramsDe.FCW1 = dlarray(initializeGaussian([512,settings.latent_dim],.02));
paramsDe.FCb1 = dlarray(zeros(512,1,'single'));
paramsDe.FCW2 = dlarray(initializeGaussian([512,512]));
paramsDe.FCb2 = dlarray(zeros(512,1,'single'));
paramsDe.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),512]));
paramsDe.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
%% Discriminator
paramsDis.FCW1 = dlarray(initializeGaussian([512,settings.latent_dim],.02));
paramsDis.FCb1 = dlarray(zeros(512,1,'single'));
paramsDis.FCW2 = dlarray(initializeGaussian([256,512]));
paramsDis.FCb2 = dlarray(zeros(256,1,'single'));
paramsDis.FCW3 = dlarray(initializeGaussian([1,256]));
paramsDis.FCb3 = dlarray(zeros(1,1,'single'));

% average Gradient and average Gradient squared holders
avgG.Dis = []; avgGS.Dis = []; avgG.En = []; avgGS.En = [];
avgG.De = []; avgGS.De = [];
%% Train
dlx = gpdl(trainX(:,1),'CB');
dly = Encoder(dlx,paramsEn);
numIterations = floor(size(trainX,2)/settings.batch_size);
out = false; epoch = 0; global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX,2));
    trainXshuffle = trainX(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch=gpdl(single(trainXshuffle(:,idx)),'CB');

        [GradEn,GradDe,GradDis] = ...
                dlfeval(@modelGradients,XBatch,...
                paramsEn,paramsDe,paramsDis,settings);

        % Update Discriminator network parameters
        [paramsDis,avgG.Dis,avgGS.Dis] = ...
            adamupdate(paramsDis, GradDis, ...
            avgG.Dis, avgGS.Dis, global_iter, ...
            settings.lrD, settings.beta1, settings.beta2);

        % Update Encoder network parameters
        [paramsEn,avgG.En,avgGS.En] = ...
            adamupdate(paramsEn, GradEn, ...
            avgG.En, avgGS.En, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
        
        % Update Decoder network parameters
        [paramsDe,avgG.De,avgGS.De] = ...
            adamupdate(paramsDe, GradDe, ...
            avgG.De, avgGS.De, global_iter, ...
            settings.lrG, settings.beta1, settings.beta2);
        
        if i==1 || rem(i,20)==0
            progressplot(paramsDe,settings);
            if i==1 
                h = gcf;
                % Capture the plot as an image 
                frame = getframe(h); 
                im = frame2im(frame); 
                [imind,cm] = rgb2ind(im,256); 
                % Write to the GIF File 
                if epoch == 0
                  imwrite(imind,cm,'AAEmnist.gif','gif', 'Loopcount',inf); 
                else 
                  imwrite(imind,cm,'AAEmnist.gif','gif','WriteMode','append'); 
                end 
            end
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
%% model Gradients
function [GradEn,GradDe,GradDis]=modelGradients(x,paramsEn,paramsDe,paramsDis,settings)
dly = Encoder(x,paramsEn);
latent_fake = dly(1:settings.latent_dim,:)+...
    dly(settings.latent_dim+1:2*settings.latent_dim)*...
    randn(settings.latent_dim,settings.batch_size);
latent_real = gpdl(randn(settings.latent_dim,settings.batch_size),'CB');

% Train the discriminator
d_output_fake = Discriminator(latent_fake,paramsDis);
d_output_real = Discriminator(latent_real,paramsDis);
d_loss = -.5*mean(log(d_output_real+eps)+log(1-d_output_fake+eps));

% Train the enocder and decoder
x_ = Decoder(latent_fake,paramsDe);
g_loss = .999*mean(mean(.5*(x_-x).^2,1))-.001*mean(log(d_output_fake+eps));

% For each network, calculate the gradients with respect to the loss.
[GradEn,GradDe] = dlgradient(g_loss,paramsEn,paramsDe,'RetainData',true);
GradDis = dlgradient(d_loss,paramsDis);
end
%% preprocess
function x = preprocess(x)
x = double(x)/255;
x = (x-.5)/.5;
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
%% Encoder
function dly = Encoder(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,.2);
end
%% Decoder
function dly = Decoder(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,.2);
dly = tanh(dly);
end
%% Discriminator
function dly = Discriminator(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = sigmoid(dly);
end
%% progressplot
function progressplot(paramsDe,settings)
r = 5; c = 5;
noise = gpdl(randn([settings.latent_dim,r*c]),'CB');
gen_imgs = Decoder(noise,paramsDe);
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