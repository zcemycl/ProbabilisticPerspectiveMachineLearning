%% Data Generation
% data from normal distribution
ndatapts = 200;
data = randn(ndatapts,1);
outliers = [8 ; 8.75 ; 9.5];
noutliers = 3;
nbins = 7; % histogram properties

% Visualize the data
subplot(221)
[counts,locs] = histcounts(data,nbins);
sCounts = counts./ndatapts;
bar(locs,[0,sCounts])

% Now fit 
subplot(223)
hold on;
[x,y] = GaussFit(data);
plot(x,y)
[x,y] = LaplaceFit(data);
plot(x,y);
[x,y]=StudentFit(data);
plot(x,y);
bar(locs,[0,sCounts])
legend('Gaussian','Laplacian','Student')
hold off;

% data from normal + outlier
subplot(222)
[counts,locs] = histcounts(data,nbins);
sCounts = counts./(ndatapts+noutliers);
bar(locs,[0,sCounts])
hold on;
[counts,locs] = histcounts(outliers,noutliers);
sCounts = counts./(ndatapts+noutliers);
bar(locs,[0,sCounts])
hold off;

% Now fit
subplot(224)
hold on;
[x,y] = GaussFit([data;outliers]);
plot(x,y)
[x,y] = LaplaceFit([data;outliers]);
plot(x,y)
[x,y]=StudentFit([data;outliers]);
plot(x,y);
[counts,locs] = histcounts(data,nbins);
sCounts = counts./(ndatapts+noutliers);
bar(locs,[0,sCounts])
legend('Gaussian','Laplacian','Student')
hold off;






%% Helper Functions
% GaussianFit (1D is easy)
function [x,y] = GaussFit(data)
mu = mean(data); sigma = std(data);
dmin = min(data); dmax = max(data);

x = dmin:(dmax-dmin)/10000:dmax;
y = exp(-.5*(x-mu).^2/sigma^2)./sqrt(2*pi*sigma^2);
end

% LaplacianFit
function [x,y] = LaplaceFit(data)
dmin = min(data); dmax = max(data);
mu = median(data); b = sum(abs(data-mu))/length(data);

x = dmin:(dmax-dmin)/10000:dmax;
y = exp(-abs(x-mu)/b)./(2*b);
end
