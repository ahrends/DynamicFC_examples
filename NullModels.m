%% 4.1 Testing against null models
% Christine Ahrends

%% Multivariate Gaussian null data
% (optional:) load synthetic normalised example timecourse 
% load('example_tc.mat')
% alternatively use own data
% assuming here that timecourse is timepoints x channels

T = 1000; % no. of timepoints to simulate
n_areas = size(tc_real,2);
mu = mean(tc_real);
Sigma = cov(tc_real);
% generate random data from multivariate Gaussian distribution
tc_mvn = mvnrnd(mu, Sigma,T); % timecourse with same mean and covariance as real data

% compute time-varying FC, e.g. sliding windows
window_length = 10;
real_tvFC_tmp10 = zeros(n_areas, n_areas, T - window_length + 1);
mvn_tvFC_tmp10 = zeros(n_areas, n_areas, T - window_length + 1);
for t = 1:T - window_length + 1
   real_tvFC_tmp10(:,:,t) = corr(tc_real(t:t+window_length-1,:)); % tvFC matrices real data
   mvn_tvFC_tmp10(:,:,t) = corr(tc_mvn(t:t+window_length-1,:)); % tvFC matrices MVN surrogate data  
   % for visualisation only tvFC between parcels 1 & 2:
   real_tvFC10(1,t) = real_tvFC_tmp10(1,2,t); 
   mvn_tvFC10(1,t) = mvn_tvFC_tmp10(1,2,t);
end
% temporal variability in surrogate timeseries although covariance is
% static due to sampling variability - especially w small window size

% sliding windows analysis w larger window size
window_length = 200;
real_tvFC_tmp200 = zeros(n_areas, n_areas, T - window_length + 1);
mvn_tvFC_tmp200 = zeros(n_areas, n_areas, T - window_length + 1);
for t = 1:T - window_length + 1
   real_tvFC_tmp200(:,:,t) = corr(tc_real(t:t+window_length-1,:));
   mvn_tvFC_tmp200(:,:,t) = corr(tc_mvn(t:t+window_length-1,:));
   real_tvFC200(1,t) = real_tvFC_tmp200(1,2,t);   
   mvn_tvFC200(1,t) = mvn_tvFC_tmp200(1,2,t);
end
% less temporal variability in surrogate due to large window size (less
% sampling variability)

% create Figure 7
figure;
% row 1: plot two parcels' timecourses
subplot(3,2,1); plot(tc_real(:,1:2)); 
title('Real data'); xlabel('Time'); ylabel('Amplitude');
subplot(3,2,2); plot(tc_mvn(:,1:2));
title('Surrogate data (multivariate Gaussian)'); xlabel('Time'); ylabel('Amplitude');
% row 2: plot tvFC at window size 10
subplot(3,2,3); plot(real_tvFC10); 
title('Sliding window correlations (window size: 10 TR)'); xlabel('Time');
ylim([-1,1]); ylabel('Correlation coefficient');
subplot(3,2,4); plot(mvn_tvFC10); xlabel('Time');
ylim([-1,1]); ylabel('Correlation coefficient');
% row 3: plot tvFC at window size 200
subplot(3,2,5); plot(real_tvFC200); 
title('Sliding window correlations (window size: 200 TR)'); xlabel('Time');
ylim([-1,1]); ylabel('Correlation coefficient');
subplot(3,2,6); plot(mvn_tvFC200); xlabel('Time');
ylim([-1,1]); ylabel('Correlation coefficient');

%% Autoregressive randomisation model
% Example following Liegeois et al. (2017) NeuroImage 

% get distribution parameters from real data
% Y = beta*X+residuals
Y = tc_real(2:end,:)';
X(1,:) = ones(1,T-1); % contains the intercept
X(2:n_areas+1,:) = tc_real(1:T-1, :)';
beta = (Y*X')/(X*X');
residuals = Y-beta*X;

% generate data from autoregressive randomisation moddel
c = beta(:,1); % intercept
weights = beta(:, 2:n_areas+1);
tc_real_flip = tc_real';
tc_arr = zeros(T, n_areas);
tc_arr(1,:) = tc_real_flip(:,randi(T-1))';
rand_t = randperm(T-1);
res_mu = mean(residuals,2); % mean of residuals (to generate noise)
res_Sigma = cov(residuals'); % covariance of residuals (to generate noise)
noise = (mvnrnd(res_mu,res_Sigma,T-1))'; % noise to be added
for i = 2:T
   tc_arr(i,:) = c' + (weights*tc_arr(i-1,:)')' + ...   
      noise(:,rand_t(i-1))'; % timecourse from ARR model
end

window_length = 200;
arr_tvFC_tmp200 = zeros(n_areas, n_areas, T - window_length + 1);
for t = 1:T - window_length + 1
   arr_tvFC_tmp200(:,:,t) = corr(tc_arr(t:t + window_length-1,:)); % tvFC matrices ARR surogate data
   % for visualisation only tvFC between parcels 1 & 2:
   arr_tvFC200(1,t) = arr_tvFC_tmp200(1,2,t); 
end

% create Figure 8
figure;
% row 1: plot two parcels' timecourses
subplot(2,2,1); plot(tc_real(:,1:2)); 
title('Real data'); xlabel('Time'); ylabel('Amplitude');
subplot(2,2,2); plot(tc_arr(:,1:2));
title('Surrogate data (1st order autoregressive randomisation model)'); 
xlabel('Time'); ylabel('Amplitude');
% row 3: plot tvFC at window size 200
subplot(2,2,3); plot(real_tvFC200); 
title('Sliding window correlations (window size: 200 TR)'); xlabel('Time');
ylim([-1,1]); ylabel('Correlation coefficient');
subplot(2,2,4); plot(arr_tvFC200); xlabel('Time');
ylim([-1,1]); ylabel('Correlation coefficient');
