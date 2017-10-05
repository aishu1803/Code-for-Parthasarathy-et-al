function [projected_data,label,projected_data_e,label_e] =  NeuralTraj_v2(dataset,dataset_e,trials,session,etrials,count,m,st,m_e,st_e,xc1,xc2,Test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code computes the projection of neural activity onto
% principal components generated from correct trials. Please note that even
% though the code can generate projections for correct and error trials, we
% only use the projections from correct trials for Fig 2 and use both from
% correct and error trials for Fig 4 of Parthasarathy et al. The principal components are always
% generated using data from correct trials. However, Fig 2 uses all 7
% locations while Fig 4 uses only 4 locations.
% Any questions?? Please contact Aishwarya Parthasarathy at aishu.parth@gmail.com
% 30th August 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input -
% xc1 and xc2 are the time limits that denote the time period of the data
% that is used to build the principal components  For ex: if
% the principal components need to be defined by the Delay 1 space:
% xc1 = 800ms (300ms of target period +  first 500ms of Delay 1. 0
% always denotes target onset) and xc2 = 1300ms.

% dataset represents the activity of all the neurons in spike counts used to build this principal component space
% (and hence the data projected onto this space) in xms bins across the
% length of the trial for all the recorded correct trials. This is a 3d matrix.Size of 1st dimension represents
% the number of neurons used to build this space. Size of the 2nd dimension represents
% the max number of trials performed while recording any neuron included in
% this matrix. For neurons with lesser trials than this maximum, the matrix is zero-padded. And the last dimension represents the number of time bins. Similarly, dataset_e represents the matrix for error trials
%
% session refers to a matrix that specifies the session during which the
% neuron in the dataset was recorded. For ex: If dataset contained activity
% from 256 neurons. size of session is 256 x 2, where the first column
% specifies the identity of the neuron within each session and the second
% column specifies the session during which the neuron was recorded.

% trials - size of this structure equals the number of sessions from which
% the neurons were used in this analysis. For every session, the structure
% contains timing information for the different task epochs and information
% about the presented target and distractor for all correct trials.
% etrials represents a similar structure for error trials

% count - number of neurons in dataset

% region - 'dlpfc' or 'fef'

% Nboot - number of bootstraps

% m/m_e - mean activity during baseline (before target presentation) for
% correct/error trials

% st/st_e - standard deviation of the activity during baseline (before
% target presentation) for correct/error trials

% Test is a string that takes the value 'Correct' or 'Error' to denote if
% we are generating the projections seen in Fig 3 (7 locations) or Fig 4 (4 locations).


% Output variables

% projected_data/projected_data_e - neural activity from correct/error
% trials spanning across the length of the trial from n neurons when
% projected onto all the principal components. 3d matrix - Ntrials x Ndimensions x Nbins

% label - Trial identity (target location/distractor location) of the data
% projected onto the principal components. Size - Ntrials x 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calling the function to build a set of pseudotrials for all training and
% testing sets. 
[training_label,train_trials,testing_label,test_trials,testing_label2,test_trials2] = MakeTrialSet('target',trials,etrials,session,count,Test);
% Initializing a variable (pr_data) to store the data used to build the PCA
% space
pr_data = [];
loc = unique(training_label);
% Looping through time bins from xc1 to xc2 to build the data for PCA space
for i_b = xc1:xc2
    % Initializing the variable for train data (averaged across trial) 
    train_data = zeros(count,length(unique(training_label)));
    % Looping through the seven trial labels.
    for q = 1:length(unique(training_label))
        % Finding all trials with label q
        tr_ind = find(training_label==loc(q));
        % Initializing variable for train data (not averaged across trials)
        train_data1=[];
        % Looping to fill in train_data1 with trial averaged data for each
        % target label.
        for i_len = 1:length(tr_ind)
            for i_co = 1:size(dataset,1)
                % Computing the z-score
                train_data1(i_co,i_len) = ((dataset(i_co,train_trials(i_co,tr_ind(i_len)),i_b))-m(i_co))./st(i_co);
            end
        end
        % Averaging over all trials in each label.
        train_data(:,q) = mean(train_data1,2);   
    end
    % Appending the data to a matrix that has data from previous time bins
    pr_data = [pr_data train_data];
end
% Finding nan's in the data (for creating the pcs) due to low firing neurons
[r,c] = find(isnan(pr_data));
% Creating a dummy test data (data to be projected) to check for nans. This
% data only has trials with label 2. And only compute the z-score for the
% first time bin
for w = 1:50
    ind_tr = find(testing_label==2);
    ind_tr_subsample = ind_tr(randsample(length(ind_tr),25,true));
    for i_bins = 1:1
        for i_len = 1:25
            for i_co = 1: size(dataset,1)
                test_data(i_co,i_len) = ((dataset_e(i_co,test_trials(i_co,ind_tr_subsample(i_len)),i_bins))-m_e(i_co))./st_e(i_co);
            end
        end
    end
end
% Finding the nan's in the dummy test_data
[r1,c1] = find(isnan(test_data));
% Finding the neurons that have nan's in train and test data.
r_fin = unique([r; r1]);
% Eliminating those neurons from train data
pr_data(unique(r_fin),:)=[];
% Performing PCA on the train data to extract principal components. It is
% defined by the variable V
A = pr_data - mean(pr_data,2);
[V,D] = eig(cov(A'));
% Building test data to project on the principal components 
loc = unique(testing_label);
% Initializing count for the instances of projected data.
count = 1;
% Looping over all the trial labels to build a dataset of error trials to
% project onto the pcs
for q = 1:length(loc)
    % Picking 50 trial averaged data points for each trial label
    for w = 1:50
        % Finding all the trials amongst the testing label with label q
        ind_tr = find(testing_label==loc(q));
        % Subsample from this list to get 25 trials.
        ind_tr_subsample = ind_tr(randsample(length(ind_tr),25,true));
        % Looping over all the time bins
        for i_bins = 1:58
            % Looping over the subsampled 25 trials
            for i_len = 1:length(ind_tr_subsample)
                % Looping over each neuron
                for i_co = 1: size(dataset,1)
                    % Computing the z-score
                    test_data(i_co,i_len) = ((dataset_e(i_co,test_trials(i_co,ind_tr_subsample(i_len)),i_bins))-m_e(i_co))./st_e(i_co);
                end
            end
            % Eliminating all the neurons with nans (found in a previous step)
            test_data(r_fin,:)=[];
            % Projecting the mean data from the subsampled 25 trials onto
            % the pcs
            tmp_proj_data = mean(test_data,2)'*fliplr(V);
            projected_data_e(:,count,i_bins) = tmp_proj_data;
        end
        label_e(count) = loc(q);
        count = count+1;
    end
end
% Similarly, to build a dataset from correct trials to project onto the
% pcs.
count = 1;
for q = 1:length(loc)
    for w = 1:50
        ind_tr = find(testing_label2==loc(q));
        ind_tr_subsample = ind_tr(randsample(length(ind_tr),25,true));
        for i_bins = 1:58
            for i_len = 1:25
                for i_co = 1: size(dataset,1)
                    test_data(i_co,i_len) = ((dataset(i_co,test_trials2(i_co,ind_tr_subsample(i_len)),i_bins))-m(i_co))./st(i_co);
                end
            end
            test_data(r_fin,:)=[];
            tmp_proj_data = mean(test_data,2)'*fliplr(V);
            projected_data(:,count,i_bins) = tmp_proj_data;
        end
        label(count) = loc(q);
        count = count+1;
    end
end
end
%% Function to create labels and trials for training and testing set.
function [train_label,train_label_no,test_label,test_label_no,test1_label,test1_label_no] = MakeTrialSet(Trial_Label,trials,etrials,session,count,Test)
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
else
    label=3;
end
% Dividing all the trials into training and testing pile. Also created a
% testing pile for error trials. The training pile is from which the PCA
% space is built. The data recorded in trials from the testing pile are
% projected onto this PCA space
for i_session = 1:length(trials)
    train_num = randsample(length(trials(i_session).val),round(0.50*(length(trials(i_session).val))));
    test_num1 = setdiff(1:length(trials(i_session).val),train_num);
    test_num = 1:length(etrials(i_session).val);
    train_set(i_session).val = trials(i_session).val(train_num);
    train_set(i_session).orgind = train_num;
    train_set(i_session).tarlabel = AssignTrialLabel(train_set(i_session).val,label);
    test1_set(i_session).val = trials(i_session).val(test_num1);
    test_set(i_session).val = etrials(i_session).val;
    test_set(i_session).orgind = test_num;
    test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    test1_set(i_session).orgind = test_num1;
    test1_set(i_session).tarlabel = AssignTrialLabel(test1_set(i_session).val,label);
    train_num=[];test_num=[];
end
% train_tr, test_tr and test1_tr define the number of pseudo-trials that
% need to be built for building the pca space and the data projected onto
% the space.
train_tr = length([train_set.val;]);
test_tr = length([test_set.val;]);
test1_tr = length([test1_set.val;]);
% count_sess stores the number of neurons recorded in each session
% Initializing count_sess
for i_session = 1:length(trials)
    count_sess(i_session) = length(find(session(:,1)==i_session));   
end
if strcmp(Test,'Correct') || strcmp(Test,'correct')
    % Creates a uniform distribution of trial labels (based on Trial_Label)
    % between 1 and 7 of length train_tr
    train_label = randsample(1:7,train_tr,true);
    % Creates a uniform distribution of trial labels between 1 and 7 of length
    % test_tr
    test_label = randsample(1:7,test_tr,true);
    % Creates a uniform distribution of trial labels between 1 and 7 of length
    % test1_tr
    test1_label = randsample(1:7,test1_tr,true);
else
    % Only 4 locations were chosen while performing the analysis on Fig 4
    % as only these 4 locations were well represented in error trials
    % across all recording sessions. 
    train_label = randsample([2 3 5 6],train_tr,true);
    test_label = randsample([2 3 5 6],test_tr,true);
    test1_label = randsample([2 3 5 6],test1_tr,true);
end
% id is a counter for the number of cells used in this analysis.
id = 1;
% Initializing the variables storing the pseudo-trials for each neuron
% described by the trial label in train_label/test_label/test1_label
train_label_no = zeros(count,train_tr);test_label_no = zeros(count,test_tr);
% i_session loops through the number of recorded sessions used in this
% analysis.
for i_session = 1:length(trials)
    % Checking if there are any neurons recorded in the session
    if count_sess(i_session)~=0
        % if there are neurons in that recorded session, loop through the
        % length of training set. For each value in train_label, find the
        % trials from the training pool for that recorded session
        % (represented as i_session) And repeat this for
        % every trial label in train_label
        for i_len = 1:length(train_label)
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
            train_label_no(id:id+count_sess(i_session)-1,i_len) = train_set(i_session).orgind(ind(train_label_tmp));
        end
        % The same process for building pseudotrials for the test set
        for i_len = 1:length(test_label)
            ind=[];test_label_tmp=[];
            ind = find(test_set(i_session).tarlabel==test_label(i_len));
            if ~isempty(ind)
                test_label_tmp = (randsample(length(ind),count_sess(i_session),true));
                test_label_no(id:id+count_sess(i_session)-1,i_len) = test_set(i_session).orgind(ind(test_label_tmp));
                % if the test set does not have trials in that session for
                % any particular trial label - the following loop is
                % executed.
            else
                while isempty(ind)
                    tmp = setdiff(1:7,test_label(i_len));
                    test_label(i_len) = tmp(1);
                    ind = find(test_set(i_session).tarlabel==test_label(i_len));
                    if ~isempty(ind)
                        test_label_tmp = (randsample(length(ind),count_sess(i_session),true));
                        test_label_no(id:id+count_sess(i_session)-1,i_len) = test_set(i_session).orgind(ind(test_label_tmp));
                    end
                end
            end
        end
        % Same process for error trials if we need to project error trials.
        for i_len = 1:length(test1_label)
            ind=[];test_label_tmp=[];
            ind = find(test1_set(i_session).tarlabel==test1_label(i_len));
            if ~isempty(ind)
                test_label_tmp = (randsample(length(ind),count_sess(i_session),true));
                test1_label_no(id:id+count_sess(i_session)-1,i_len) = test1_set(i_session).orgind(ind(test_label_tmp));
            else
                tmp = setdiff(1:7,test1_label(i_len));
                test1_label(i_len) = tmp(1);
                ind = find(test1_set(i_session).tarlabel==test1_label(i_len));
                test_label_tmp = (randsample(length(ind),count_sess(i_session),true));
                test1_label_no(id:id+count_sess(i_session)-1,i_len) = test1_set(i_session).orgind(ind(test_label_tmp));
            end
        end
    end
    id = id+count_sess(i_session);
end
clearvars -except train_label train_label_no test_label test_label_no test1_label test1_label_no
end

