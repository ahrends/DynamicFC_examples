%% 3.1.1 Sliding windows
% Diego Vidaurre, 2022
% updated by Christine Ahrends

%% Sliding window analysis using correlation

% (optional:) load synthetic normalised example timecourse 
% load('example_tc.mat')
% X = tc_real;
% alternatively use own data
% assuming here that timecourse is timepoints x channels

p = size(X,2); % no. of channels
total_session_duration = size(X,1); % no. of timepoints

window_length = 100;
C = zeros(p,p,total_session_duration - window_length + 1);
for t = 1:total_session_duration - window_length + 1
C(:,:,t) = corr(X(t:t+window_length-1,:));
end

% plot correlation of first 10 windows (compare Figure 4B)
figure;
for i = 1:10
    subplot(1,10,i); imagesc(squeeze(C(:,:,i))); axis square;
end

%% Sliding window analysis using partial correlation

lambda = 0.1; % regularisation constant
for t = 1:total_session_duration - window_length + 1
    inv_mat = inv(C(:,:,t) + lambda * eye(p));
    inv_mat = - (inv_mat ./ ...
     	repmat(sqrt(abs(diag(inv_mat))),1,p)) ./ ...
        repmat(sqrt(abs(diag(inv_mat)))',p,1);
    inv_mat(inv_mat(p)>0)=0;
    iC(:,:,t) = inv_mat;
end

%% Estimation noise in sliding windows

% create synthetic timecourse with static FC:
true_C = cov(tc_real); % covariance from real timecourse
total_session_duration = 2000; 
p = 10; 

% generate data from multivariate Gaussian distribution with same
% covariance as real data
clear X
X = mvnrnd(zeros(total_session_duration,p),true_C);
for j = 1:p
    X(:,j) = smooth(X(:,j),10);
end
X = X(101:end-100,:);

% sliding windows analysis of synthetic data w static covariance (full
% correlation)
total_session_duration = size(X,1); 
window_length = 100;
C = zeros(p,p,total_session_duration - window_length + 1);
for t = 1:total_session_duration - window_length + 1
C(:,:,t) = corr(X(t:t+window_length-1,:));
end
% partial correlation
lambda = 0.1; % regularisation constant
iC = zeros(p,p,total_session_duration - window_length + 1);
for t = 1:total_session_duration - window_length + 1
    inv_mat = inv(C(:,:,t) + lambda * eye(p));
    inv_mat = - (inv_mat ./ ...
     	repmat(sqrt(abs(diag(inv_mat))),1,p)) ./ ...
        repmat(sqrt(abs(diag(inv_mat)))',p,1);
    inv_mat(inv_mat(p)>0)=0;
    iC(:,:,t) = inv_mat;
end 

% create Figure 5
figure; 
% Figure 5A
subplot(2,2,1);
plot(squeeze(C(1,2,:))); hold on; plot(squeeze(iC(1,2,:)));
yline(true_C(1,2),'--'); yline(corr(X(:,1),X(:,2)));
xlabel('Volumes'); ylabel('Correlation/Partial correlation');

% create synthetic data with changing covariance matrix
total_session_duration = 2000; 
p = 10; 

% create synthetic covariance matrices that are variations of real
% covariance matrix (see also 4.2 simulations)
[U,S,~] = svd(true_C);
e = diag(S); e = cumsum(e) / sum(e); 
true_C1 = zeros(size(true_C));
true_C2 = zeros(size(true_C));
% less than 30% variation
J = find(e>=0.3,1);
for j = 1:J
    true_C1 = true_C1 + U(:,j) * S(j,j) * U(:,j)'; 
    true_C2 = true_C2 + U(:,j) * S(j,j) * U(:,j)'; 
end
for j = J+1:10
	c1 = U(randperm(size(U,1)),j);
    true_C1 = true_C1 + c1 * S(j,j) * c1'; 
    c2 = U(randperm(size(U,1)),j);
    true_C2 = true_C2 + c2 * S(j,j) * c2'; 
end

% generate random MVN data from synthetic covariance matrices
X1a = mvnrnd(zeros(total_session_duration/4,p),true_C1);
X1b = mvnrnd(zeros(total_session_duration/4,p),true_C1);
X2 = mvnrnd(zeros(total_session_duration/2,p),true_C2);
X = [X1a; X2; X1b]; 
for j = 1:p
    X(:,j) = smooth(X(:,j),10);
end
X = X(101:end-100,:);

% sliding windows analysis of synthetic data w changing covariance (full
% correlation)
total_session_duration = size(X,1); 
window_length = 100;
C = zeros(p,p,total_session_duration - window_length + 1);
for t = 1:total_session_duration - window_length + 1
C(:,:,t) = corr(X(t:t+window_length-1,:));
end
% partial correlation
lambda = 0.1; % regularisation constant
iC = zeros(p,p,total_session_duration - window_length + 1);
for t = 1:total_session_duration - window_length + 1
    inv_mat = inv(C(:,:,t) + lambda * eye(p));
    inv_mat = - (inv_mat ./ ...
     	repmat(sqrt(abs(diag(inv_mat))),1,p)) ./ ...
        repmat(sqrt(abs(diag(inv_mat)))',p,1);
    inv_mat(inv_mat(p)>0)=0;
    iC(:,:,t) = inv_mat;
end 

% Figure 5B
subplot(2,2,3);
% plot sliding window correlation and partial correlation
plot(squeeze(C(1,2,:))); hold on; plot(squeeze(iC(1,2,:)));
% overlay ground truth correlation true_C1 and true_C2 as dotted line
plot((1:total_session_duration/4),repmat(true_C1(1,2), total_session_duration/4),'k--');
plot((total_session_duration/4+1):(total_session_duration/4*3), ... 
    repmat(true_C2(1,2), total_session_duration/2),'k--');
plot((total_session_duration/4*3+1):total_session_duration, ...
    repmat(true_C1(1,2), total_session_duration/4), 'k--');
% overlay empirical correlation as solid line
plot((1:total_session_duration/4),...
    repmat(corr(X(1:total_session_duration/4,1),X(1:total_session_duration/4,2)),...
    total_session_duration/4), 'k');
plot((total_session_duration/4+1):(total_session_duration/4*3), ... 
    repmat(corr(X((total_session_duration/4+1):(total_session_duration/4*3),1),...
    X((total_session_duration/4+1):(total_session_duration/4*3),2)),...
    total_session_duration/2), 'k');
plot((total_session_duration/4*3+1):total_session_duration, ...
    repmat(corr(X((total_session_duration/4*3+1):total_session_duration,1),...
    X((total_session_duration/4*3+1):total_session_duration,2)),...
    total_session_duration/4), 'k');
xlabel('Volumes'); ylabel('Correlation/Partial correlation');
