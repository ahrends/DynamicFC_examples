%% 2.2 Parcellations and timecourse extraction
% Christine Ahrends, 2022

% Note that rows 3 and 4 of figures may look different due to stochasticity
% of PCA and ICA (incl. possible sign-flips)
%% Scenario 1: homogeneous voxel timecourses

% create cluster A (10 x 10 voxels)
A = zeros(10,10);
% peak cluster activity at cluster center
for i = 1:5
    A(i, i:(end-(i-1))) = 0.2*i;
    A(i:(end-(i-1)), i) = 0.2*i;
    A(end-(i-1), i+1:end-(i-1)) = 0.2*i;
    A(i+1:end-(i-1), end-(i-1)) = 0.2*i;
end

% cluster B has same structure as cluster A
B = A;

% create timecourses
T = 1:100; % no. of timepoints
yA = A(:)*sin(T);
yB = B(:)*sin(T)*0.7; % cluster B's amplitude is 70% of cluster A

% binary parcellation (single parcel), timecourse extraction: mean
for t = T
    parcel_mean(t) = mean([yA(:,t); yB(:,t)]); % parcel timecourse
end

% binary parcellation (single parcel), timecourse extraction: 1st PC
parcel_pc_tmp = pca([yA; yB]);
parcel_pc = parcel_pc_tmp(:,1); % parcel timecourse

% weighted data-driven parcellation (single parcel), timecourse extraction:
% regression
ica_mdl = rica([yA;yB],1); % ICA-model with one component (parcel)
parcel_ic = ica_mdl.TransformWeights; % parcel timecourse
z = transform(ica_mdl,[yA;yB]); % voxel weights (all voxels)
weights_ICA_A = reshape(z(1:100),[10,10]);  % voxel weights cluster A
weights_ICA_B = reshape(z(101:end),[10,10]); % voxel w2eights cluster B

% create Figure 1
figure; 
% left column: timecourses
subplot(4,2,1); 
plot(T,yA'); title('Empirical voxel timecourses');
xlabel('Time'); hold on;
plot(T,yB'-2.5); hold off;
subplot(4,2,3); 
xlabel('Time');
plot(T,parcel_mean); ylim([-1,1]); title('Binary parcellation, mean extracted');
subplot(4,2,5); 
xlabel('Time');
plot(T,parcel_pc); ylim([-1,1]); title('Binary parcellation, 1st PC extracted');
subplot(4,2,7); 
xlabel('Time');
plot(parcel_ic); ylim([-1,1]); title('Weighted parcellation, regression coefficients extracted');

% right column: voxel weights
subplot(4,2,2);
imagesc([A;B]); xlim([0,10]); axis equal tight; colorbar;
title('Empirical voxel weights');
subplot(4,2,4);
imagesc(ones(20,10)); colorbar; axis equal tight;
title('Voxel weights binary parcel')
subplot(4,2,6);
imagesc(ones(20,10)); colorbar; axis equal tight;
title('Voxel weights binary parcel')
subplot(4,2,8);
imagesc([weights_ICA_A; weights_ICA_B]); colorbar; axis equal tight;
title('Voxel weights weighted parcel')

%% Scenario 2: heterogeneous voxel timecourses

% create cluster A (10 x 10 voxels)
A = zeros(10,10);
% peak cluster activity at cluster center
for i = 1:5
    A(i, i:(end-(i-1))) = 0.2*i;
    A(i:(end-(i-1)), i) = 0.2*i;
    A(end-(i-1), i+1:end-(i-1)) = 0.2*i;
    A(i+1:end-(i-1), end-(i-1)) = 0.2*i;
end

% cluster B has same structure as cluster A
B = A;

% create timecourses
T = 1:100; % no. of timepoints
yA = A(:)*sin(T);
yB = B(:)*-sin(T); % cluster B's timecourse is negatively correlated w cluster A

% binary parcellation (single parcel), timecourse extraction: mean
for t = T
    parcel_mean(t) = mean([yA(:,t); yB(:,t)]); % parcel timecourse
end

% binary parcellation (single parcel), timecourse extraction: 1st PC
parcel_pc_tmp = pca([yA; yB]);
parcel_pc = parcel_pc_tmp(:,1); % parcel timecourse

% weighted data-driven parcellation (single parcel), timecourse extraction:
% regression
ica_mdl = rica([yA;yB],1); % ICA-model with one component (parcel)
parcel_ic = ica_mdl.TransformWeights; % parcel timecourse
z = transform(ica_mdl,[yA;yB]); % voxel weights (all voxels)
weights_ICA_A = reshape(z(1:100),[10,10]);  % voxel weights cluster A
weights_ICA_B = reshape(z(101:end),[10,10]); % voxel weights cluster B

% create Figure 2
figure; 
% left column: time courses
subplot(4,2,1); 
plot(T,yA'); title('Empirical voxel timecourses');
xlabel('Time'); hold on;
plot(T,yB'-2.5); hold off;
subplot(4,2,3); 
xlabel('Time');
plot(T,parcel_mean); ylim([-1,1]); title('Binary parcellation, mean extracted');
subplot(4,2,5); 
xlabel('Time');
plot(T,parcel_pc); ylim([-1,1]); title('Binary parcellation, 1st PC extracted');
subplot(4,2,7); 
xlabel('Time');
plot(parcel_ic); ylim([-1,1]); title('Weighted parcellation, regression coefficients extracted');

% right column: voxel weights
subplot(4,2,2);
imagesc([A;B]); xlim([0,10]); axis equal tight; colorbar;
title('Cluster A & Cluster B');
subplot(4,2,4);
imagesc(ones(20,10)); colorbar; axis equal tight;
title('Voxel weights binary parcel')
subplot(4,2,6);
imagesc(ones(20,10)); colorbar; axis equal tight;
title('Voxel weights binary parcel')
subplot(4,2,8);
imagesc([weights_ICA_A; weights_ICA_B]); colorbar; axis equal tight;
title('Voxel weights ICA parcellation')

%% Example 3 (dynamic FC)

% create cluster A (10 x 10 voxels)
A = zeros(10,10);
% peak cluster activity at cluster center
for i = 1:5
    A(i, i:(end-(i-1))) = 0.2*i;
    A(i:(end-(i-1)), i) = 0.2*i;
    A(end-(i-1), i+1:end-(i-1)) = 0.2*i;
    A(i+1:end-(i-1), end-(i-1)) = 0.2*i;
end

% cluster B has same structure as cluster A
B = A;

% create timecourses
T = 1:100; % no. of timepoints
yA = A(:)*sin(T);
yB(:,1:30) = B(:)*sin(T(1:30)); % first 30 timepoints - B is correlated with A
yB(:,31:70) = B(:)*sin(31:0.5:50.5); % timepoints 31:70 - B oscillates at 1/2 frequency of A
yB(:,71:100) = B(:)*sin(T(71:100)); % timepoints 70:100 - B is correlated with A

% binary parcellation (2 parcels), timecourse extraction: mean
for t = T
    parcel_mean(1,t) = mean([yA(1:50,t); yB(1:50,t)]); % timecourse parcel 1
    parcel_mean(2,t) = mean([yA(51:100,t); yB(51:100,t)]); % timecourse parcel 2
end

% binary parcellation (2 parcels), timecourse extraction: 1st PC
parcel_pc_tmp_1 = pca([yA(1:50,:); yB(1:50,:)]); 
parcel_pc(:,1) = parcel_pc_tmp_1(:,1); % timecourse parcel 1
parcel_pc_tmp_2 = pca([yA(51:100,:); yB(51:100,:)]);
parcel_pc(:,2) = parcel_pc_tmp_2(:,1); % timecourse parcel 2

% weighted data-driven parcellation (single parcel), timecourse extraction:
% regression
ica_mdl = rica([yA;yB],2); % ICA model with 2 components
parcel_ic = ica_mdl.TransformWeights; % timecourses parcels 1 and 2
z = transform(ica_mdl,[yA;yB]); 
weights_ICA_A_1 = reshape(z(1:100,1),[10,10]); % voxel weights cluster A parcel 1
weights_ICA_B_1 = reshape(z(101:end,1),[10,10]); % voxel weights cluster B parcel 1
weights_ICA_A_2 = reshape(z(1:100,2), [10,10]); % voxel weights cluster A parcel 2
weights_ICA_B_2 = reshape(z(101:end,2),[10,10]); % voxel weights cluster B parcel 2

% time-varying FC (sliding windows, see 3.1.1 and sliding_windows.m)
n_areas = 2;
n_timepoints = 100;
window_length = 10;
orig_tvFC_tmp = zeros(n_areas, n_areas, n_timepoints - window_length + 1);
mean_tvFC_tmp = zeros(n_areas, n_areas, n_timepoints - window_length + 1);
pc_tvFC_tmp = zeros(n_areas, n_areas, n_timepoints - window_length + 1);
for t = 1:n_timepoints - window_length + 1 % slide windows through timecourses and get correlation between parcels
    orig_tvFC_tmp(:,:,t) = corr(mean(yA(:,t:t + window_length-1))', mean(yB(:,t:t + window_length - 1))');
    orig_tvFC(t) = orig_tvFC_tmp(1,2,t); % empirical dFC
    mean_tvFC_tmp(:,:,t) = corr(parcel_mean(:,t:t + window_length-1)');
    mean_tvFC(t) = mean_tvFC_tmp(1,2,t); % dFC of mean tc from 2 binary parcels
    pc_tvFC_tmp(:,:,t) = corr(parcel_pc(t:t + window_length-1,:));
    pc_tvFC(t) = pc_tvFC_tmp(1,2,t); % dFC of 1st PC tc from 2 binary parcels
    ic_tvFC_tmp(:,:,t) = corr(parcel_ic(t:t + window_length-1,:));
    ic_tvFC(t) = ic_tvFC_tmp(1,2,t); % dFC of ICs
end

T_windows = 1:(n_timepoints - window_length + 1);

% create Figure 3
figure; 
% left column: timecourses
subplot(4,3,1); 
plot(T,yA'); title('Empirical voxel timecourses'); ylim([-4,2]); 
xlabel('Time'); hold on;
plot(T,yB'-2.5); hold off;
subplot(4,3,4); 
xlabel('Time');
plot(T,parcel_mean(1,:)); ylim([-4,2]); title('Binary parcellation, mean extracted');
hold on;
plot(T,parcel_mean(2,:)-2.5);
subplot(4,3,7); 
xlabel('Time');
plot(T,parcel_pc(:,1)); ylim([-4,2]); title('Binary parcellation, 1st PC extracted');
hold on;
plot(T,parcel_pc(:,2)-2.5);
subplot(4,3,10); 
xlabel('Time');
plot(parcel_ic(:,1)); ylim([-4,2]); title('Timecourses IC parcel'); 
hold on;
plot(parcel_ic(:,2)-2.5);

% middle column: voxel weights
subplot(4,3,2);
imagesc([A;B]); colorbar; axis equal tight; 
title('Cluster A & Cluster B');
subplot(4,3,5);
binary_weights1 = ones(20,10);
binary_weights1(:,6:10) = 0;
binary_weights2 = ones(20,10);
binary_weights2(:,1:5) = 0;
imagesc([binary_weights1,binary_weights2]); colorbar; axis equal tight; 
title('Voxel weights binary parcel')
subplot(4,3,8);
imagesc([binary_weights1, binary_weights2]); colorbar; axis equal tight; 
title('Voxel weights binary parcel')
subplot(4,3,11);
imagesc([weights_ICA_A_1, weights_ICA_A_2; weights_ICA_B_1, weights_ICA_B_2]); colorbar; axis equal tight;
title('Voxel weights ICA parcellation')

% right column: dFC
subplot(4,3,3);
plot(T_windows, orig_tvFC); ylim([-1.2,1.2]); title('empirical tvFC');
xlabel('Time');
subplot(4,3,6);
plot(T_windows, mean_tvFC); ylim([-1.2,1.2]); title('tvFC binary parcels, mean');
xlabel('Time');
subplot(4,3,9);
plot(T_windows, pc_tvFC); ylim([-1.2,1.2]); title('tvFC binary parcels, 1st PC')
subplot(4,3,12);
xlabel('Time');
plot(T_windows, ic_tvFC); ylim([-1.2,1.2]); title('tvFC weighted parcels, dual regression');
xlabel('Time');
