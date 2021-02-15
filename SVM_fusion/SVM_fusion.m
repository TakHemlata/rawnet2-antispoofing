clear;clc;closeall;

%% We fused our all 3 RawNet2 (linear,mel and inverse-mel) systems with our high-resolution LFCC baseline system (Interspeech 2020 paper)

%% add path
addpath(genpath('tDCF_v1'));

%% Load devlopment and evalution scores with target labels

load S_dev.mat 
load S_eval.mat


% sort the eval scores and targets 
[eval_labels, ind] = sort(eval_labels,'descend');
S_eval = S_eval(ind,:);

S_dev = S_dev';
S_eval = S_eval';

% get indices of genuine and spoof files
bonafideIdx_dev = find(dev_labels==1);
spoofIdx_dev = find(dev_labels==0);

bonafideIdx_eval = find(eval_labels==1);
spoofIdx_eval = find(eval_labels==0);


%% SVM fusion

% We train SVM on development scores generated (from all 4 systems) and
% test on evalution scores

SVMModel = fitcsvm(S_dev',dev_labels','KernelFunction','linear','KernelScale','auto','Standardize',true);

% Score prediction on dev data
[~,scores_cm_dev] = predict(SVMModel,S_dev'); scores_cm_dev = scores_cm_dev(:,2);

% read development protocol
fileID = fopen(fullfile('ASVspoof2019.LA.cm.dev.trl.txt'));
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

attacks = protocol{4};
type = protocol{3};

% metric for ASVspoof2019
evaluate_tDCF_asvspoof19_v1(attacks, type, scores_cm_dev, 'dev')

% Score prediction on eval data

[~,scores_cm_eval] = predict(SVMModel,S_eval'); scores_cm_eval = scores_cm_eval(:,2);

% read eval protocol
fileID = fopen(fullfile('ASVspoof2019.LA.cm.eval.trl.txt'));
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

attacks = protocol{4};
attacks = attacks(ind);
type = protocol{3};
type = type(ind);

% metric for ASVspoof2019
evaluate_tDCF_asvspoof19_v1(attacks, type, scores_cm_eval, 'eval')

