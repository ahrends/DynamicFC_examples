%% 4.2 Comparing models with simulated data
% Diego Vidaurre, 2022
% updated by Christine Ahrends

%% load data
% (optional:) load synthetic normalised example timecourse 
% load('example_tc.mat')
% real_data = tc_real;
% alternatively use own data
% assuming here that timecourse is timepoints x channels

%% generate data from multivariate Gaussian distribution with same covariance as real data

some_time = 5000; % number of timepoints to be simulated
p = 10; % number of regions to be simulated
C = cov(real_data);
X = mvnrnd(zeros(some_time,p),C);
for j = 1:p
    X(:,j) = smooth(X(:,j),10);
end

%% generate synthetic covariance matrices with small variations

[U,S,~] = svd(C);
e = diag(S); e = cumsum(e) / sum(e); 
C_synth = zeros(size(C));
% less than 90% variation
J = find(e>=0.9,1);
for j = 1:J
    C_synth = C_synth + U(:,j) * S(j,j) * U(:,j)'; 
end
for j = J+1:p
	c = U(randperm(size(U,1)),j);
    C_synth = C_synth + c * S(j,j) * c'; 
end
% C_synth is a synthetic covariance matrix 

%% sample ground truth state timecourses

total_session_duration = 20000;
max_duration = 100; min_duration = 5;
true_stc = zeros(total_session_duration,1);
t=1;
while t <= total_session_duration
    k = randi(4,1);
    L = randi(max_duration-min_duration,1) + min_duration;
    if t+L-1 > total_session_duration
		L = total_session_duration - t + 1; 
    end
    true_stc(t:t+L-1,k) = 1;
    t = t + L + 1;
end

%% compare performance of HMM and clustering approach

% running this part takes a while, so model accuracies on 100 random 
% iterations of the below simulations are also provided (note though that
% these are not exactly the same as in the main text):
% load('accuracies_sim.mat')

% running the HMM requires the HMM-MAR toolbox (available at
% https://github.com/OHBA-analysis/HMM-MAR)
addpath(genpath('/HMM-MAR-master'))

% singular value decomposition of covariance matrix from real data
[U,S,~] = svd(C);
e = diag(S); e = cumsum(e) / sum(e); 

% set up simulation options
state_variation = [1,3,6];
transition_speed = [100, 1000; 5, 50];
total_session_duration = 20000;
K = 4;

% set up HMM options
options = struct();
options.K = K;
options.covtype = 'full';
options.zeromean = 1; % states only have a covariance matrix

% set up sliding window clustering options
window_length = 100;
N_windows = total_session_duration - window_length + 1;
rng('shuffle')

for n = 1:100 % number of iterations
    for jj = 1:3
        % create synthetic covariance matrices for 4 states with different
        % degrees of variation between states
        C_synth = cell(1,K);
        for k = 1:K
            C_synth{k} = zeros(size(C));
            for j = 1:state_variation(jj)
                C_synth{k} = C_synth{k} + U(:,j) * S(j,j) * U(:,j)';
            end
            for j = state_variation(jj)+1:p
            	c = U(randperm(size(U,1)),j);
                C_synth{k} = C_synth{k} + c * S(j,j) * c';
            end
        end
        for i = 1:2
            % generate ground truth state timecourse with slow vs. fast
            % state transitions
            max_duration = transition_speed(1,i); min_duration = transition_speed(2,i);
            true_stc = zeros(total_session_duration,1);
            t=1;
            while t <= total_session_duration
                k = randi(4,1);
                L = randi(max_duration-min_duration,1) + min_duration;
                if t+L-1 > total_session_duration
            		L = total_session_duration - t + 1;
                end
                true_stc(t:t+L-1,k) = 1;
                % generate timecourse for duration of state visit from
                % multivariate Gaussian with covariance matrix
                % corresponding to respective state
                X(t:t+L-1,:) = mvnrnd(zeros(L,p),C_synth{k});
                t = t + L + 1;
            end
            for j = 1:p
                X(:,j) = smooth(X(:,j),10);
            end
            % run HMM on simulated timecourse
            [hmm,stc_hmm] = hmmmar(X,size(X,1),options);
            % align HMM states with ground truth states
            [assig_hmm,cost_hmm] = munkres(1-corr(true_stc,stc_hmm));
            stc_hmm = stc_hmm(:,assig_hmm);
            % get accuracy of HMM in recovering ground truth state
            % timecourses
            accuracy_hmm(i,jj,n) = corr(stc_hmm(:),true_stc(:));
            % run sliding windows + clustering on simulated timecourse
            % sliding windows FC estimates
            C_SW = zeros(p,p,total_session_duration - window_length + 1);
            for t = 1:total_session_duration - window_length + 1
                C_SW(:,:,t) = corr(X(t:t+window_length-1,:));
            end
            % clustering
            C_unwrapped = zeros(N_windows,p*(p-1)/2);
            for j = 1:N_windows
                Cj = C_SW(:,:,j); Cj = Cj(triu(true(p),1));
                C_unwrapped(j,:) = Cj(:);
            end
            idx = kmeans(C_unwrapped,K); % cluster assignment of each timepoints FC estimate
            stc_clustering = zeros(total_session_duration,K);
            for k = 1:K
                stc_clustering(idx==k,k) = 1; % timecourses of all K states (rows are timepoints, columns are states)
            end
            % align clustering states with ground truth states
            [assig_clustering,cost_clustering] = munkres(1-corr(true_stc,stc_clustering));
            stc_clustering = stc_clustering(:,assig_clustering);
            % get accuracy of clustering in recovering ground truth state
            % timecourses
            accuracy_clustering(i,jj,n) = corr(stc_clustering(:),true_stc(:));
        end
    end
end

% save('accuracies_sim.mat', 'accuracy_hmm', 'accuracy_clustering');

% create Figure 9
transition_label = {'Faster', 'Slower'};
statevar_label = {'larger', 'medium', 'smaller'};

figure;
for jj = 1:3
    for i = 1:2
        subplot(3,2,(jj-1)*2+(i*-1+3)); 
        bar([1,2],[mean(accuracy_clustering(i,jj,:)), mean(accuracy_hmm(i,jj,:))], 'w'); hold on;
        swarmchart(ones(100,1),squeeze(accuracy_clustering(i,jj,:)),'.');
        swarmchart((ones(100,1)+1),squeeze(accuracy_hmm(i,jj,:)), '.');
        xlim([0,3]); xticks([1,2]), xticklabels({'clustering','HMM'}); 
        ylim([0,1]); ylabel('Accuracy');
        title([transition_label{i} ' transitions, ' statevar_label{jj} ' state variations']); hold off;
    end
end

