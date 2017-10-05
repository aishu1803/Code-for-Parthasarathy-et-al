function trial_label = AssignTrialLabel(trials,flag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function identifies the stimulus label
% (target/distractor/target-distractor location/euclidean distance between
% target and distractor location in trials) for the trials under trials.
% Any questions?? Please contact Aishwarya Parthasarathy at aishu.parth@gmail.com
% 30th August 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
% trials - array of structs containing timing and stimulus information from correct/
% error / aborted trials. This information about stimulus identity is
% stored as row,column co-ordinates. This function parses the row,column co-ordinates to
% to produces a discrete label for each stimulus label. Please note that for aborted trials, the animal
% should be engaged in the trial until the stimulus (label for which is
% identified in the function) is presented in the trial. Usually when
% AssignTrialLabel is called in a script/function, trials structure passed
% to the function contains information about trials recorded in one
% session.
% flag -  describes which stimulus label to identify
% information about
% flag == 1 Target
% flag == 2 Distractor
% flag == 3 Target-Distractor pair
% flag == 4 Distance between target and distractor.
%
% Output
% trial_label - array of the same length as trials. Each entry in this
% variable corresponds to the discrete stimulus label identified in this
% function.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discrete_xy defines the discrete stimulus label for the row-column coordinates
discrete_xy = [1 2 2;2 2 3;3 2 4;4 3 2;5 3 4;6 4 2;7 4 3;8 4 4];
% discrete_xy_joint defines the discrete stimulus label for
% target_distractor pairs presented in the trial. There are 56/42 possible
% combinations if there are 8 target locations. The 2 columns of
% discrete_xy_joint define the different combinations of  discrete target and
% distractor location respectively. The 3rd column of this
% particular combination is its discrete label.
discrete_xy_joint = nchoosek(1:8,2);
k = size(discrete_xy_joint,1);
% As nchoosek does not duplicate pairs of target - distractor label
% (it treats 2-3 the same as 3-2 and hence only returns only one of the
% two), The next two lines add the duplicates to the list.
discrete_xy_joint(k+1:k+k,1) = discrete_xy_joint(1:k,2);
discrete_xy_joint(k+1:k+k,2) = discrete_xy_joint(1:k,1);
% By sorting this list, the discrete labels are ordered according to the
% target locations presented amongst the pair. For ex - labels 1 to 7
% contain all the pairs that have target at location 1 and distractor at
% any of the other 7 locations.
[~,b] = sort(discrete_xy_joint(:,1),'ascend');
discrete_xy_joint = discrete_xy_joint(b,:);
discrete_xy_joint(:,3) = 1:size(discrete_xy_joint,1);
% The fourth column of this list contains the euclidean distance between
% target and distractor location presented in the trial.
for t = 1:size(discrete_xy_joint,1)
    x = discrete_xy(discrete_xy_joint(t,1),2:3);
    y = discrete_xy(discrete_xy_joint(t,2),2:3);
    discrete_xy_joint(t,4) = sqrt((y(2) - x(2))^2 + (y(1) - x(1))^2);
end
% Initializing the output trial_label.
trial_label = zeros(1,length(trials));
% The loop runs through every trial in the struct trials to parse the
% stimulus identity.
for i = 1:length(trials)
    % If Target labels are parsed
    if flag==1
        % Identify the row of the target location 
        row = trials(i).target.row;
        % Identifying the column
        col = trials(i).target.column;
        % Find the discrete target label from discrete_xy for the particular
        % row and column of the target.
        ind = discrete_xy(:,3)==row & discrete_xy(:,2)==col;
        % Assigning the discrete target label for trial i in trial_label
        trial_label(1,i) = discrete_xy(ind,1);
        % If distractor labels are parsed
    elseif flag==2
        row = trials(i).distractors(2);
        col = trials(i).distractors(3);
        ind = discrete_xy(:,3)==row & discrete_xy(:,2)==col;
        trial_label(1,i) = discrete_xy(ind,1);
        % If target - distractor paired labels are parsed
    elseif flag==3
        % Identifying discrete target label
        row_t = trials(i).target.row;
        col_t = trials(i).target.column;
        ind_t = discrete_xy(:,3)==row_t & discrete_xy(:,2)==col_t;
        trial_label_tar = discrete_xy(ind_t,1);
        % Identifying discrete distractor label
        row_d = trials(i).distractors(2);
        col_d = trials(i).distractors(3);
        ind_d = discrete_xy(:,3)==row_d & discrete_xy(:,2)==col_d;
        trial_label_dist = discrete_xy(ind_d,1);
        % Identifying the label for target-distractor pair from column 3 of
        % discrete_xy_joint
        ind = discrete_xy_joint(:,1)==trial_label_tar & discrete_xy_joint(:,2)==trial_label_dist;
        trial_label(1,i) = discrete_xy_joint(ind,3);
        % If the distance between target and distractor labels ina  trial is
        % parsed.
    elseif flag==4
        row_t = trials(i).target.row;
        col_t = trials(i).target.column;
        ind_t = discrete_xy(:,3)==row_t & discrete_xy(:,2)==col_t;
        trial_label_tar = discrete_xy(ind_t,1);
        row_d = trials(i).distractors(2);
        col_d = trials(i).distractors(3);
        ind_d = discrete_xy(:,3)==row_d & discrete_xy(:,2)==col_d;
        trial_label_dist = discrete_xy(ind_d,1);
        % Identifying the distance between target and distractor locations
        % from the 4th column of discrete_xy_joint
        ind = discrete_xy_joint(:,1)==trial_label_tar & discrete_xy_joint(:,2)==trial_label_dist;
        trial_label(1,i) = discrete_xy_joint(ind,4);
    end 
end
end
