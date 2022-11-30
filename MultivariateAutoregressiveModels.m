%% 3.3 Multivariate Autoregressive Models
% Diego Vidaurre, 2022
% updated by Christine Ahrends

%% load data

% (optional:) load synthetic normalised example timecourse 
% load('example_tc.mat')
% X = tc_real;
% alternatively use own data
% assuming here that timecourse is timepoints x channels

p = size(X,2); % number of channels

%% estimate autoregressive coefficients of timeseries
% X = Y*A_star + residuals

L = 2; % order of multivariate autoregressive model
Y = zeros(size(X,1) - L , L * p);
for j = 1:L
    Y(:,(1:p)+(j-1)*p) = X(L-j+1:end-j,:);
end
A_star = pinv(Y) * X(L+1:end,:); % autoregressive coefficients

%% example scenario: sample from linear dynamical system

rng('shuffle')

p = 5; 
total_session_duration = 1000;
L = 1; % order of multivariate autoregressive model
true_A = 0.5*eye(p); 
for j = 1:p-1
    true_A(j,j+1) = 0.25; true_A(j+1,j) = 0.25;
end

for i = 1:1000 % number of iterations
    X = randn(total_session_duration,p); % random samples
    for t = 2:total_session_duration
        X(t,:) = X(t,:) + X(t-1,:) * true_A;
    end
    Y = zeros(size(X,1) - L , L * p);
    for j = 1:L
        Y(:,(1:p)+(j-1)*p) = X(L-j+1:end-j,:);
    end
    A_star(:,:,i) = pinv(Y) * X(L+1:end,:); % estimate autoregressive coefficients
end

%% create Figure 6

for ii = 1:p
    label_vec(ii,1) = (ii-1)*p+1;
end
label_vec(:,2) = 21:25;

figure; 
for i = 1:p
    for j = 1:p
        subplot(p,p,(i-1)*p+j); histogram(A_star(i,j,:)); hold on;
        xline(0, 'r', 'LineWidth', 3); xlim([-0.1, 0.7]);
        if ismember((i-1)*p+j,label_vec(:,1))
            ylabel(['Region ' num2str(i)]);
        end
        if ismember((i-1)*p+j,label_vec(:,2))
            xlabel(['Region ' num2str(j)]);
        end
        hold off;
    end
end


