samples = uniform(1,5,[1000000,1]);
subplot(221)
histogram(samples,4)

subplot(222)
histogram(uniform(1,2,[1000000,1]),1);
xlim([1,5])

subplot(223)
histogram(randi([1,4],1000000,1));
title('discrete')

subplot(224)
histogram(uniform(1,4,[1000000,1]));
title('continuous')

function samples = uniform(a,b,shape)
samples = (b-a).*rand(shape)+a;
end