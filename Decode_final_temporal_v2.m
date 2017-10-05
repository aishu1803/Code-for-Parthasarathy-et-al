function performance = Decode_final_temporal_v2(dataset,m,st,session,Trial_Label,trials,bins,etrials,dataset_e,m_e,st_e,Test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function decodes a trial label from the dataset built from the
% recorded neurons. Specifically, the function evaluates the performance of
% the decoder when you dissociate the  time windows from which you pick the
% training and testing data. This dataset may be built from a pseudo-population or
% from simultaneously recorded ensemble of neurons. This code generates
% results for Fig 2a, 2e, 4a, 4f, 6a, 6b, 6c, 6e, 6f and 6g. This
% code is run 1000 times to generate a distribution of performances. The
% mean of this distribution of performances is plotted as a heatmap
% in these figures.
% Any questions?? Please contact Aishwarya Parthasarathy at aishu.parth@gmail.com
% 30th August 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs-

% dataset : Size - Nneurons x Ntrials x Nbins.
% This matrix contains the activity of all the neurons used in decoding
% during correct trials in relevant time bins. Ntrials is the maximum
% number of trials recorded by the neurons. If a neuron is recorded with
% less than Ntrials trials, then the rest of the values in the matrix till
% Ntrials is zero-padded. Nbins is the length of the time bins for which
% this decoding was performed. Therefore the 5th neuron's activity in the
% 6th trial during the 8th time bin is denoted by dataset(5,6,8)

% m : Size - Nneurons x 1
% Each element in this array is the mean activity across all trials during
% the baseline period (300ms prior to the target onset) of a neuron

% st: Size - Nneurons x 1
% Each element in this array is the standard deviation of the activity
% activity across all trials during the baseline period (300ms prior to the
% target onset) of a neuron

% session: Size - Nneurons x 2
% Each element in the first column refers to the session in which the
% neuron was recorded. Also, this value as the index of the trials variable
% fetches all the trials performed in the session where the said neuron was
% recorded.

% Trial_Label - 'target' or 'distractor'
% this string refers to the trial label to be decoded from the dataset.

% trials: Size - struct 1 x Nsession
% Nsession is the number of recorded sessions from which we pool neurons to
% form dataset. Nsession also equals the number of unique values in the
% first column of the sessions array.

% dataset_e,m_e,st_e,etrials are the same as dataset,m,st and trials
% repectively but for error trials.

% Test is a string that takes the value 'Correct' or 'Error' to denote if
% we are decoding correct or error trials. This string is useful in
% differentiating the decoding in Fig 2 and Fig 4 of Parthasarathy et al
% Note while decoding the target from data recorded during error trials,
% the decoder is trained using data from correct trials and tested using data
% from error trials.
%
% Output -
% performance: Size - Nbinstrain x Nbinstest
% An element (x,y) in this array refers to the performance of the decoder when
% trained using data from the xbin and tested using data from ybin.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Builds a set of training and testing trial labels to build the
% training and testing dataset that will eventually be fed to the decoder.
[training_label,train_trials,testing_label,test_trials] = MakeTrialSet(Trial_Label,trials,etrials,session,Test);
% Initializing the training dataset with zeros
train_data = zeros(size(dataset,1),size(train_trials,2),size(dataset,3));
% Intializing the testing dataset with zeros
test_data = zeros(size(dataset,1),size(test_trials,2),size(dataset,3));
% Filling up train_data from dataset. All the values are z_scored using m,st to normalize the firing rate across neurons
for i_co = 1:size(dataset,1)
    
    train_data(i_co,:,:) = ((dataset(i_co,train_trials(i_co,:),:))-m(i_co))./st(i_co);
end
% Filling up test_data from dataset. All the values are z_scored using m,st to normalize the firing rate across neurons
if strcmp(Test,'Correct') || strcmp(Test,'correct')
    for i_co = 1:size(dataset,1)
        test_data(i_co,:,:) = ((dataset(i_co,test_trials(i_co,:),:))-m(i_co))./st(i_co);
    end
elseif strcmp(Test,'Error') || strcmp(Test,'error')
    for i_co = 1:size(dataset,1)
        test_data(i_co,:,:) = ((dataset_e(i_co,test_trials(i_co,:),:))-m_e(i_co))./st_e(i_co);
    end
else
    for i_co = 1:size(dataset,1)
        test_data(i_co,:,:) = ((dataset(i_co,test_trials(i_co,:),:))-m(i_co))./st(i_co);
    end
end
% Looping through the training time bins
for i_b = 1:size(bins,2)-1
    % Looping through the testing time bins
    for i_bins = 1:size(bins,2)-1
        % De-noising the dataset to feed into the decoder
        [train_data_new,test_data_new] = Build_DataSet(squeeze(train_data(:,:,i_b)),squeeze(test_data(:,:,i_bins)));
        % Decoding and computing the percentage of the trials that the
        % decoder classified correctly for corresponding training and
        % testing window
        [performance(i_b,i_bins) ] = ComputePerformance(train_data_new,test_data_new,training_label,testing_label);
    end
end
end
%% Function to create a matrix of training and testing labels
function [train_label,train_label_no,test_label,test_label_no] = MakeTrialSet(Trial_Label,trials,etrials,session,Test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function builds a matrix of trial indices and labels to build the
% training set. Trials from every session are split into training and
% testing pool. Further, two uniform distributions of 7 target labels was
% built to define the training and testing set's trial label. The length of
% these uniform distribution is defined by the number of trials used to
% train and test the decoder. For example, if the first trial label in the training set
% is target location 1, the function picks one trial for each neuron in the
% training pool with the target presented at target 1. Similarly, this is
% repeated for all the trial labels in the training and testing set.
% Inputs -
% Trial_Label - string - 'target' or 'distractor'
% trials - struct- passed on from the main function
% etrials - struct - passed on from the main function
% session - Nneurons x 2 matrix - passed on from the main function.
% Outputs -
% train_label - Ntraintrials x 1 matrix - uniform distribution of the 7
% trial_labels (target or distractor)
% train_label_no - Nneurons x Ntraintrials - indices of trials with
% trial label specified by train_label for all Nneurons.
% test_label and test_label_no are similar to train_label and
% train_label_no respectively but the trials to build test_label_no are
% chosen from the testing pool. Usually the length of train_label is set to
% 1500 and the length of test_label is set to 100.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deciphering the trial label to decode and assigning label the
% corresponding value, label = 1 when target is decoded and label = 2 when
% distractor is decoded.
if strcmp(Trial_Label,'Target') || strcmp(Trial_Label,'target')
    label=1;
elseif strcmp(Trial_Label,'Distractor') || strcmp(Trial_Label,'distractor')
    label=2;
end
% Dividing trials from each session into training and testing groups
for i_session = 1:length(trials)
    % Randomly picking 50% of the trials in a session to be under the
    % training pool
    train_num = randsample(length(trials(i_session).val),round(0.50*(length(trials(i_session).val))));
    % Assigning the other 50% to be the testing pool. While decoding error
    % trials it will be replaced with 1:length(etrials(i_session).val)
    if strcmp(Test,'Correct') || strcmp(Test,'correct')
        test_num = setdiff(1:length(trials(i_session).val),train_num);
    elseif strcmp(Test,'Error') || strcmp(Test,'error')
        test_num = 1:length(etrials(i_session).val);
    else
        test_num = setdiff(1:length(trials(i_session).val),train_num);
    end
    % Building the train_set with trial details for each session using
    % train_num
    train_set(i_session).val = trials(i_session).val(train_num);
    % Storing their original indices from trials
    train_set(i_session).orgind = train_num;
    % Storing the target labels of all the trials in train_set. Please note
    % that AssignTrialLabel function is specific for our dataset. This
    % function identifies the label (target or distractor location) for each 
    % trial. This needs to be modified if you are not using the dataset
    % used in Parthasarathy et al and subsequently the lines of code using
    % the output of AssignTrialLabel.
    train_set(i_session).tarlabel = AssignTrialLabel(train_set(i_session).val,label);
    % Similar variables for test_set
    if strcmp(Test,'Correct') || strcmp(Test,'correct')
        test_set(i_session).val = trials(i_session).val(test_num);
        test_set(i_session).orgind = test_num;
        test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    elseif strcmp(Test,'Error') || strcmp(Test,'error')
        test_set(i_session).val = etrials(i_session).val;
        test_set(i_session).orgind = test_num;
        test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    else
        test_set(i_session).val = trials(i_session).val(test_num);
        test_set(i_session).orgind = test_num;
        test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    end
    train_num=[];test_num=[];
end
% Setting Ntraintrials
train_tr = 1500;
% Setting Ntesttrials
test_tr = 100;
% count_sess stores the number of neurons recorded in each session
% Initializing count_sess
count_sess = zeros(length(trials),1);
for i_session = 1:length(trials)
    count_sess(i_session,1) = length(find(session(:,1)==i_session));
end
% Creates a uniform distribution of trial labels (based on Trial_Label)
% between 1 and 7 of length train_tr
if strcmp(Test,'Correct') || strcmp(Test,'correct')
    train_label = randsample(1:7,train_tr,true);
    %Creates a uniform distribution of trial labels between 1 and 7 of length
    %test_tr
    test_label = randsample(1:7,test_tr,true);
elseif strcmp(Test,'Error') || strcmp(Test,'error')
    train_label = randsample([2 3 5 6],train_tr,true);
    test_label = randsample([2 3 5 6],test_tr,true);
else
    train_label = randsample([2 3 5 6],train_tr,true);
    test_label = randsample([2 3 5 6],test_tr,true);
end
% id is a counter for the number of cells used in this analysis.
id = 1;
% i_session loops through the number of recorded sessions used in this
% analysis.
for i_session = 1:length(trials)
    % Checking if there are any neurons recorded in the session
    if count_sess(i_session,1)~=0
        % if there are neurons in that recorded session, loop through the
        % length of training set. For each value in train_label, find the
        % trials from the training pool for that recorded session
        % (represented as i_session) And repeat this for
        % every trial label in train_label
        for i_len = 1:train_tr
            % Initializing temporary variables
            train_label_tmp=[];ind=[];
            % Finding all the trials in the training pool with the i_len th
            % value of train_label.
            ind = find(train_set(i_session).tarlabel==train_label(i_len));
            % Sample with replacement from ind as many times as the number
            % of neurons in the recorded session (i_session)
            train_label_tmp = (randsample(length(ind),count_sess(i_session),true));
            % Build train_label_no with the selected trials with
            % train_label_tmp. Note id is the cell counter and
            % id+count_sess(i_session,1) is the counter after adding all
            % the neurons recorded in i_session.
            train_label_no(id:id+count_sess(i_session,1)-1,i_len) = train_set(i_session).orgind(ind(train_label_tmp));
        end
        % Go through the same loop for test trials to build hte test_label_no
        for i_len = 1:length(test_label)
            ind=[];test_label_tmp=[];
            ind = find(test_set(i_session).tarlabel==test_label(i_len));
            test_label_tmp = (randsample(length(ind),count_sess(i_session),true));
            test_label_no(id:id+count_sess(i_session)-1,i_len) = test_set(i_session).orgind(ind(test_label_tmp));
        end
    end
    id = id+count_sess(i_session);
end
clearvars -except train_label train_label_no test_label test_label_no
end
%% Function to build the training and testing set
% This function denoises the train and test dataset using PCA. The data
% used to decode is the data projected onto the principal components.
function [train_data_new,test_data_new] = Build_DataSet(train_data,test_data)
% Initializing a matrix to store all the neuron indices with NaN values.
% These are neurons with very low firing.
ind_nan=[];
% Checking for neurons with NaN values within the train data
for i = 1:size(train_data,1)
    if ~isfinite(train_data(i,1))
        ind_nan = [ind_nan i];
    end
end
% Checking for neurons with NaN values within the test data
for i = 1:size(test_data,1)
    if ~isfinite(test_data(i,1))
        ind_nan = [ind_nan i];
    end
end
% Pick all the neurons that had NaNs in the train and/or test data.
ind_nan = unique(ind_nan);
% Getting rid of these neurons in the train and test dataset.
train_data(ind_nan,:)=[];test_data(ind_nan,:)=[];
% Creating a PCA space with training and testing data.
A = [train_data';test_data'] - mean([train_data';test_data'],2);
[V,D] = eig(cov(A));
%[coeff,score,latent] = princomp([train_data';test_data']);
score = A*fliplr(V);
% Computing the proportion of explained variance for each component
latent = sort(diag(D),'descend');
latent = cumsum(latent);
latent = latent/latent(end);
% Finding the number of components explaining 90% of the variance.
expl_var = dsearchn(latent,0.9);
% The denoised train data is the projection of the original data on the
% first n components of the PCA space created in line 215.
train_data_new = score(1:size(train_data,2),1:expl_var);
% Denoising the test data similarly.
test_data_new = score(size(train_data,2)+1:end,1:expl_var);

end
%% Calculating decoding performance using an LDA
function [performance] = ComputePerformance(train_data,test_data,training_label,testing_label)
% class is the predicted target label from the test_data
[class,~,~,~,~] = classify(test_data,train_data,training_label);
% Checking how different the predicted label is from the actual label for
% all the test trials
perf = class-testing_label';
% Extracting all the correctly predicted target label
perf_ind = find(perf==0);
% The performance of the decoder is computed as the percentege of number of correct
% predictions in the decoding.
performance = length(find(perf==0))*100/length(testing_label);
end
