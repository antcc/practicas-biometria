%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FaceRecognition_SVM.m
%%%
%%% Luis Antonio Ortega Andrés
%%% Antonio Coín Castro
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Global variables

rng(2021);  % Seed random number generator

%% Initialize and load images

close all
clear all
clc

% We add paths to facilitate the code
addpath(cd)
addpath('DetPlots')
cd FaceDatabaseATT
dirListA=dir;
dirList=dirListA(4:43);
n_users = numel(dirList);

% Divide the number of images per user for Train and Test
Train=6;
Test=10-Train;

% Each image is 92px x 112px
width=92*112;

% Initialize the Feature and Label Matrix
MatrixTrainFeats=zeros(Train*40, width);
MatrixTestFeats=zeros(Test*40, width);
MatrixTrainLabels=zeros(Train*40,1); % Each row contains the ID of the user
MatrixTestLabels=zeros(Test*40,1);

for i=1:n_users  % For each user
    % Move directory to current user
    cd(dirList(i).name);
    images=dir('*.pgm');

    %%% Feature extraction for Training Dataset

    for j=1:Train %Train images
       % Read image
       im=imread(images(j).name);
       im=double(im);

       % Convert image to row vector (flatten)
       im = reshape(im.', 1, []);

       % Fill train matrix
       MatrixTrainFeats((i-1)*Train + j, :) = im;
       MatrixTrainLabels((i-1)*Train + j, 1)= i;  % User i
    end

    %%% Feature extraction for Test Dataset

    for j=(Train+1):10
       % Read image
       im=imread(images(j).name);
       im=double(im);

       % Convert image to row vector (flatten)
       im = reshape(im.', 1, []);

       % Fill test matrix
       MatrixTestFeats((i-1)*Test + j - Train, :) = im;
       MatrixTestLabels((i-1)*Test + j - Train, 1) = i;  % User i
    end

    % Return to database directory
    cd ..
end

%% Cross-validation of the whole process

% Range of principal components (change at will between 1 and 239)
min_ncomp = 12;
max_ncomp = 12;
E = zeros(max_ncomp - min_ncomp + 1, 1);  % Array to save EERs
plot_DET = true;  % Whether to plot DET curve for each run

for NComp=min_ncomp:max_ncomp  % For each number of principal components

    %%% Extract features using PCA

    [coeff_PCA, MatrixTrainPCAFeats, ~, ~, ~, mu] = pca(MatrixTrainFeats);
    MatrixTrainPCAFeats = MatrixTrainPCAFeats(:, 1:NComp);
    MatrixTestPCAFeats = (MatrixTestFeats - mu)*coeff_PCA(:, 1:NComp);
    [n_test, n_features_pca] = size(MatrixTestPCAFeats);

    %%% Train SVMs

    n_train_extra = 234;  % No. of negative train samples for each SVM (max 234)
    svms = cell(1, n_users);  % Pre-allocate cell for SVMs
    labels = [ones(Train, 1) ;  % Genuine class labels
              zeros(n_train_extra, 1)];

    % Specify kernel ('linear', 'polynomial' or 'rbf')
    kernel = 'rbf';
    scale = 'auto';
    degree = 2;

    for i=1:n_users  % Train one classifier for each user
        % Get random training examples from other users
        pop = [1:(i-1)*Train i*Train+1:n_users*Train];
        random = randsample(pop, n_train_extra);

        % Compute training matrix for user i
        labelsUser = MatrixTrainLabels == i;
        matrix = [MatrixTrainPCAFeats(labelsUser, :) ;
                  MatrixTrainPCAFeats(random, :)];

        % Train and save model
        if strcmp(kernel, 'polynomial')
            svmModel = fitcsvm(matrix, labels, ...
                'Standardize', true, ...
                'KernelFunction', kernel, ...
                'PolynomialOrder', degree);
        elseif strcmp(kernel, 'rbf')
            svmModel = fitcsvm(matrix, labels, ...
                'KernelFunction', kernel, ...
                'KernelScale', scale);
        else
             svmModel = fitcsvm(matrix, labels);
        end

        svms{i} = svmModel;
    end

    %%% Compute EER

    TargetScores = [];
    NonTargetScores = [];

    for i=1:n_users  % For each user
        % Predict using the SVM associated with user i
        [labelsSVM, scores]=predict(svms{i}, MatrixTestPCAFeats);

        % Fill Target or NonTarget depending on label coincidence
        maskUser = MatrixTestLabels(:, 1) == i;
        TargetScores=[TargetScores, scores(maskUser, 2)'];
        NonTargetScores=[NonTargetScores, scores(~maskUser, 2)'];
    end

    % Return to root directory
    cd ..

    % Compute EER and optionally plot DET curve (the function has been modified
    % to allow this option)
    E(NComp - min_ncomp + 1, 1) = ...
        Eval_Det(TargetScores, NonTargetScores, 'b', plot_DET);
end

%% Plot EER evolution

ncomp_range = min_ncomp:max_ncomp;

if numel(ncomp_range) > 1
    figure;
    h = plot(ncomp_range, E, '-');
    set(h,'LineWidth',1.8);
    ylim([0 25]);
    title('Evolution of EER with principal components');
    xlabel('Principal components');
    ylabel('EER [%]');
end
