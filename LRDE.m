function [P, Z, E] = LRDE(X,Y,P,Sw,St,options)



[n,m] = size(X); %% n is original dimension and m is the size.
np = size(Y,1);  %% np is the low-dimension.

if (~exist('P','var'))
   P = rand(n,np);
end

if (~exist('options','var'))
   options = [];
end

gama = 0.6;
if isfield(options,'gama')
    gama = options.gama;
end
Lw = Sw-gama*St; 

alpha = 1e-2;
if isfield(options,'alpha')
    alpha = options.alpha;
end

lambda = 5;
if isfield(options,'lambda')
    lambda = options.lambda;
end

beta = 1e-1;
if isfield(options,'beta')
    beta = options.beta;
end

Z_Method = 'low-rank';
if isfield(options,'Z_Method')
    Z_Method = options.Z_Method;
end

maxIter = 15;
if isfield(options,'maxIter')
    maxIter = options.maxIter;
end

%% initilize other parameters
max_mu = 10^6;
mu = 1e-4;
rho = 1.2;
%rho = 1.8;

Q = zeros(n,np);
Z = zeros(m,m);
J = zeros(m,m);
E = zeros(np,m);

Y1 = zeros(np,m); %%<P'X - YZ - E>
Y2 = zeros(m,m);  %%<Z-J>
Y3 = zeros(n,np); %% <P-Q>


for iter = 1: maxIter
    %fprintf('Iteration: %d\n',iter);
    %% update P
    if(iter > 1)
        P1 = 2*lambda*Lw + mu*X*X' + mu*eye(n);
        P2 = mu*X*(Y*Z+E)' - X*Y1' + mu*Q - Y3;        
        P = P1\P2;
        %P = orth(P);        
    end
    
    [~,error]=classifier(X',options.trainlabels,options.testdata,P,options.testlables);
    fprintf('iter=%d, acc=%4f\n',iter,1-error);
%    [~,Qs,~] = svd(Q,'econ');
%    Qs = alpha*sum(diag(Qs));
    %loss=Qs;
%     [~,Qs,~] = svd(J,'econ');
%     Qs = sum(diag(Qs));
%     loss=loss+Qs;
%     Qs=lambda*trace(P'*Lw*P);
%     loss=loss+Qs;
%     Qs=beta*norm(E,2);
%     loss=loss+Qs;
%     fprintf('  ,loss=%4f\n',loss);
%     clear Qs loss
    
    
    
    %update Q
    tmp_Q = P + Y3/mu;
    [QU,Qs,QV] = svd(tmp_Q,'econ');
    Qs = diag(Qs);
    svp = length(find(Qs>beta/mu));
    if svp>=1
        Qs = Qs(1:svp)-beta/mu;
    else
        svp = 1;
        Qs = 0;
    end
    Q = QU(:,1:svp)*diag(Qs)*QV(:,1:svp)';


    %%update J
    %% two methods: low-rank and sparse
    tmp_Z = Z + Y2/mu; 
    if strcmp(Z_Method,'low-rank')
        [U,sigma,V] = svd(tmp_Z,'econ');
        sigma = diag(sigma);
        svp = length(find(sigma>1/mu));
        if svp>=1
            sigma = sigma(1:svp)-1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    end
    if strcmp(Z_Method,'sparse')
        J = max(0,tmp_Z - 1/mu)+min(0,tmp_Z + 1/mu);
    end

    
    %% for Z
    Z1 = (2*eye(m) + Y'*Y);
    Z2 = Y'*(P'*X - E) + J + (Y'*Y1 - Y2)/mu;
    Z = Z1\Z2;

    %% for E
    tmp_E = P' * X - Y * Z + Y1/mu;
    E = max(0,tmp_E - alpha/mu)+min(0,tmp_E + alpha/mu);

    
    %% for Y1~Y3, mu
    leq1 = P'*X - Y*Z - E;
    leq2 = Z - J;
    leq3 = P - Q;

    Y1 = Y1 + mu*leq1;
    Y2 = Y2 + mu*leq2;
    Y3 = Y3 + mu*leq3;

    mu = min(rho*mu, max_mu);   
    stopALM = norm(leq1,'fro');
    stopALM = max(norm(leq2,'fro'),stopALM);
    stopALM = max(norm(leq3,'fro'),stopALM);

    if stopALM < 1e-3
        break;
    end
end
end


