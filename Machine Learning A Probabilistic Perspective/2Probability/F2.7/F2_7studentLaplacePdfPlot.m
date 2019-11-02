% gaussian distribution
s = randn(10000,1);
subplot(231)
histogram(s)

x = -4:8/10000:4;
y = GaussDist(x,0,1);
subplot(234)
plot(x,y)

% laplacian distribution
s = randl([10000,1]);
subplot(232)
histogram(s)

y = LaplaceDist(x,0,sqrt(1/2));
subplot(235)
plot(x,y)

% student distribution
s = trnd(100,10000,1);
subplot(233)
histogram(s)

y = StudentDist(x,0,1,1);
subplot(236)
plot(x,y)

%% Helper Functions
function y = GaussDist(x,mu,sigma)
y = exp(-(x-mu).^2/2/sigma^2)/sqrt(2*pi*sigma^2);
end

function s = randl(shape)
b = sqrt(1/2); mu = 0;
u = rand(shape)-0.5;
s = mu-b*sign(u).*log(1-2*abs(u));
end

function y = LaplaceDist(x,mu,b)
y = exp(-abs(x-mu)/b)/2/b;
end

function y = StudentDist(x,mu,sigma,v)
y = (1+((x-mu)/sigma).^2/v).^(-(v+1)/2);
end