%% 3.2.1 Clustering approaches
% Diego Vidaurre, 2022
% updated by Christine Ahrends

%% load data

% (optional:) load synthetic normalised example timecourse 
% load('example_tc.mat')
% X = tc_real;
% alternatively use own data
% assuming here that timecourse is timepoints x channels

p = size(X,2); % no. of channels
total_session_duration = size(X,1); % no. of timepoints

%% k-means clustering of sliding windows estimates

% get sliding window FC estimates
window_length = 100;
C = zeros(p,p,total_session_duration - window_length + 1);
for t = 1:total_session_duration - window_length + 1
    C(:,:,t) = corr(X(t:t+window_length-1,:));
end

% k-means clustering:
% cluster sliding window FC estimates into K states
N_windows = total_session_duration - window_length + 1;
C_unwrapped = zeros(N_windows,p*(p-1)/2);
for j = 1:N_windows
    Cj = C(:,:,j); Cj = Cj(triu(true(p),1));
    C_unwrapped(j,:) = Cj(:);
end
K = 4; % example for 4 states
idx = kmeans(C_unwrapped,K); % cluster assignment of each timepoints FC estimate
state_time_courses = zeros(length(total_session_duration),K);
for k = 1:K
    state_time_courses(idx==k,k) = 1; % timecourses of all K states (rows are timepoints, columns are states)
end

% get the fractional occupancies of all states and display state time courses
mean(state_time_courses)
area(state_time_courses)

%% Leading Eigenvector Dynamics Analysis (LEiDA)
% see https://github.com/juanitacabral/LEiDA for full functionality

clear C idx

% compute instantaneous phase of signal
Phase = zeros(total_session_duration,p);
for j=1:p
    Phase(:,j) = angle(hilbert(X(:,j))); % Hilbert transform
end

% extract leading eigenvectors and calculate cosine similarity between each
% pair of signals
Eigenvectors = zeros(total_session_duration,p);
for t = 1:total_session_duration
    C = zeros(p);
    for j1 = 1:p-1
        for j2 = j1+1:p
            d = abs(Phase(t,j1) - Phase(t,j2) );
            if d > pi, d = 2*pi-d; end
            C(j1,j2) = cos(d);
            C(j2,j1) = C(j1,j2);
        end
    end
    [v,d] = eig(C); % singular value decomposition
    [~,i] = max(diag(d)); % get index of leading eigenvector
    Eigenvectors(t,:) = v(:,i);
end

K = 4; % Example for 4 states
[idx,eigen_centroids] = kmeans(Eigenvectors,K);
% idx assigns a state to each timepoint, eigen_centroids define states

