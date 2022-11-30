%% 3.2.2 Generative models: Hidden Markov Model (HMM)
% Diego Vidaurre, 2022
% updated by Christine Ahrends

%% load data

% (optional:) load synthetic normalised example timecourse 
% load('example_tc.mat')
% X = tc_real;
% alternatively use own data
% assuming here that timecourse is timepoints x channels

n_subs = 10; % number of subjects/sessions
n_ts = 100; % number of timepoints per subject/session
T = repmat(n_ts,[n_subs,1]); % either double or cell of length n_subs 
% where each entry is the number of timepoints for that subject
%% requires HMM-MAR toolbox
% available at https://github.com/OHBA-analysis/HMM-MAR

addpath(genpath('/HMM-MAR-master')) % path to HMM-MAR toolbox

%% set up options and run HMM

options = struct();
options.K = 8; % number of HMM states
options.covtype = 'full'; % states are defined by a full covariance matrix (FC)
% options.zeromean = 1; % to estimate only time-varying FC, set the mean to 0
[hmm,Gamma,~,vpath] = hmmmar(X,T,options);
% hmm is the HMM structure, Gamma are the state probabilities, vpath is the
% Viterbi path (state time courses)

FO = getFractionalOccupancy(Gamma,T,options);
for k = 1:8
   state_time_courses(vpath==k,k) = 1; % timecourses of all K states (rows are timepoints, columns are states)
end
% display fractional occupancies:
% fraction of each subject's timecourse spent in each state
figure; imagesc(FO)
% state visits over time, e.g. for subject 1
figure; area(state_time_courses(1:T(1),:)) 
% compare to state probabilities over time
figure; area(Gamma(1:T(1),:)); ylim([0,1])
