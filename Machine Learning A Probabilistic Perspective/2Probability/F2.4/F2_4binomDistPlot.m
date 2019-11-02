subplot(221)
x = 0:10;
y = binom(10,0.25);
bar(x,y)

subplot(222)
y = binom(10,0.9);
bar(x,y)

%Sampling from uniform to binomial
N = 100000; n = 4; % since p(0) = 0.25;
s = randi(n,10,N); s(s>1) = 0;
countbinary = [];
for i = 1:N
    countbinary = [countbinary,nnz(s(:,i))];
end
subplot(223)
histogram(countbinary)

n = 10; % since p(0) = 0.9;
s = randi(n,10,N); s(s>1) = 0;
countbinary = [];
for i = 1:N
    countbinary = [countbinary,10-nnz(s(:,i))];
end
subplot(224)
histogram(countbinary)



function y = binom(n,p)
y = [];
for i =0:n
    bc = nchoosek(n,i);
    y = [y,bc*p^i*(1-p)^(n-i)];
end
end