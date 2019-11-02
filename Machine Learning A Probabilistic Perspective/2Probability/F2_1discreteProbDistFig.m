samples = uniform(1,5,[1000000,1]);
subplot(121)
histogram(samples,4)

subplot(122)
histogram(uniform(1,2,[100000,1]),1);
xlim([1,5])



function samples = uniform(a,b,shape)
samples = (b-a).*rand(shape)+a;
end