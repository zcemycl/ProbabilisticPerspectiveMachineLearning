function [x,y]=StudentFit(data)
% I basically filter out some multivariate stuff
model.mu = []; model.Sigma = []; model.dof = []; 
dof0 = [];
dofEstimator = @(model)estimateDofNLL(model, data);

initFn  = @(model)init(model); 
estepFn = @(model, X)estep(model, X);
mstepFn = @(model, ess)mstep(model, ess, dofEstimator);
[model, loglikHist] = emAlgo(model, data, initFn, estepFn, mstepFn);

dmin = min(data); dmax = max(data);
x = dmin:(dmax-dmin)/10000:dmax;
mu = model.mu; sigma = model.Sigma;
v = model.dof;
coef = gamma((v+1)/2)/sqrt(v*pi)/gamma(v/2);
y=coef*(1+((x-mu)/sigma).^2/v).^(-(v+1)/2);
end

%% Helper Functions
% Given settings of mu and sigma, find the best settings
% of dof to minimize the negative log likelihood using 
% 1D linesearch
function dof = estimateDofNLL(model, X)
mu = model.mu;
Sigma = model.Sigma;
nllfn = @(v) -sum(studentLogprob(studentCreate(mu, Sigma, v), X));
dofMax = 1000; dofMin = 0.1;
dof = fminbnd(nllfn, dofMin, dofMax);
end

% Log probability of student distribution
function logp = studentLogprob(arg1,arg2)
mu = arg1.mu; Sigma = arg1.Sigma; nu = arg1.dof;
X = arg2;
d = size(Sigma, 1);
X = X(:) - mu(:);

mahal = sum((X/Sigma).*X,2);
logc = gammaln(nu/2 + d/2) - gammaln(nu/2)-.5*log(Sigma) ...
    - (d/2)*log(nu) - (d/2)*log(pi);
logp = logc  -(nu+d)/2*log1p(mahal/nu);
end

function model  = studentCreate(mu, Sigma, dof)
model.mu = mu;
model.Sigma = Sigma;
model.dof = dof;
model.ndims = length(mu); 
model.modelType = 'student';
end

function model = init(model)
model.mu = randn;
model.Sigma = diag(rand);
model.dof = ceil(5*rand());
end

function [ess, loglik] = estep(model, X)
%% Compute the expected sufficient statistics
loglik   = sum(studentLogprob(model, X));
mu       = model.mu;
Sigma    = model.Sigma;
dof      = model.dof;
[N, D]   = size(X);
XC = X-mu;
delta =  sum((XC/Sigma).*XC,2);
w = (dof+D) ./ (dof+delta);      % E[tau(i)]
Xw = X .* repmat(w(:), 1, D);
ess.Sw  = sum(w);
ess.SX  = sum(Xw, 1)'; % sum_i u(i) xi, column vector
ess.SXX = Xw'*X;       % sum_i u(i) xi xi'
ess.denom = N;
end

function model = mstep(model, ess, dofEstimator)
%% Maximize
SX    = ess.SX;
Sw    = ess.Sw;
SXX   = ess.SXX;
denom = ess.denom;
model.mu    = SX / Sw;
model.Sigma = (1/denom)*(SXX - SX*SX'/Sw); % Liu,Rubin eqn 16
if ~isempty(dofEstimator)
    model.dof = dofEstimator(model);
end
end

function [model, loglikHist, llHists] = emAlgo(model, data, init, estep, mstep)
%% Perform EM
maxIter = 50;
convTol = 1e-4;
model = init(model);

iter = 1;
done = false;
loglikHist = zeros(maxIter + 1, 1);
while ~done
    [ess, ll] = estep(model, data);
    loglikHist(iter) = ll;
    model = mstep(model, ess);
    if iter > maxIter
      done = true;
    elseif iter > 1
      done = convergenceTest(loglikHist(iter), loglikHist(iter-1), convTol);
    end
    iter = iter + 1;
end
loglikHist = loglikHist(1:iter-1);
llHists{1} = loglikHist;
end

function converged = convergenceTest(fval, previous_fval, threshold)
converged = 0;
delta_fval = abs(fval - previous_fval);
avg_fval = (abs(fval) + abs(previous_fval) + eps)/2;
if (delta_fval / avg_fval) < threshold, converged = 1; end
end
