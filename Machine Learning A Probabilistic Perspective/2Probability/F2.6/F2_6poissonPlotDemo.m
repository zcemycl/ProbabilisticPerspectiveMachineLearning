subplot(221)
x = 0:10;
y = poisson(1,10);
bar(x,y)

subplot(222)
x = 0:25;
y = poisson(10,25);
bar(x,y)

%Sampling from binomial to poisson
N = 100000; n = 100; % since p(0) = 1/100;
s = randi(n,n,N); s(s>1) = 0;
countbinary = [];
for i = 1:N
    countbinary = [countbinary,nnz(s(:,i))];
end
subplot(223)
histogram(countbinary,10)

n = 10; % since p(0) = 1/10;
s = randi(n,100,N); s(s>1) = 0;
countbinary = [];
for i = 1:N
    countbinary = [countbinary,nnz(s(:,i))];
end
subplot(224)
histogram(countbinary,25)

function y = poisson(lambda,n)
y = [];
for i =0:n
    tmp = exp(-lambda)*lambda^i/factorial(i);
    y = [y,tmp];
end
end