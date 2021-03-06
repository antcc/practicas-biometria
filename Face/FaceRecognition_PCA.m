%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FaceRecognition_PCA.m
%%%
%%% Luis Antonio Ortega Andrés
%%% Antonio Coín Castro
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
n_train = numel(MatrixTrainLabels);
n_test = numel(MatrixTestLabels);

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

% Perform PCA on training matrix
[coeff_PCA, MatrixTrainPCAFeatsAll, ~, ~, explained, mu] = ...
    pca(MatrixTrainFeats);

% Range of principal components (change at will between 1 and 239)
min_ncomp = 31;
max_ncomp = 31;
E = zeros(max_ncomp - min_ncomp + 1, 1);  % Array to save EERs
plot_DET = true;  % Whether to plot DET curve for each run

for NComp=min_ncomp:max_ncomp  % For each number of principal components

    %%% Extract features using PCA

    MatrixTrainPCAFeats = MatrixTrainPCAFeatsAll(:, 1:NComp);
    MatrixTestPCAFeats = (MatrixTestFeats - mu)*coeff_PCA(:, 1:NComp);

    %%% Similarity Computation Stage

    % Each test image is going to be compared to each of the training images
    % for each of the 40 users (N comparisons). Then, the final score is the
    % result of the min of the N comparisons of each test image with the N
    % training images of each user, which can be a genuine comparison (target)
    % or an impostor comparison (NonTarget)

    TargetScores=[];
    NonTargetScores=[];

    for i=1:n_test  % For each Test image
        for j=1:n_train % Comparison with each Training image
            my_distance(j)= ...  % Compute the distance measure
                mean(abs(MatrixTestPCAFeats(i,:) - MatrixTrainPCAFeats(j,:)));

            %if it's a genuine comparison
            if(MatrixTestLabels(i,:) == MatrixTrainLabels(j,:))
                LabelTest(j)=1;
            else % otherwise
                LabelTest(j)=0;
            end
        end

        % The final score is the min of the 6 comparisons of each Test image
        % against the training images of each user
        contF=1;
        for k=1:Train:n_train
            %Extract the scores of the N training signatures and select the min
            my_distanceRed(contF)=min(my_distance(k:k+Train-1));

            if LabelTest(k)==1 % target score
                TargetScores=[TargetScores, my_distanceRed(contF)];
            else % non target score
                NonTargetScores=[NonTargetScores, my_distanceRed(contF)];
            end

            contF=contF+1;
        end

    end

    % Multiply by -1 to have higher values for genuine comparisons,
    % as we have a distance computation. With other type of classifier
    % this wouldn't be necessary.
    TargetScores=-TargetScores;
    NonTargetScores=-NonTargetScores;

    % Return to root directory
    cd ..

    % Compute EER and optionally plot DET curve (the function has been modified
    % to allow this option)
    E(NComp - min_ncomp + 1, 1) = ...
        Eval_Det(TargetScores, NonTargetScores, 'b', plot_DET);
end

%% Plot explained variance and EER evolution

% Plot explained variance
figure;
h=plot(1:numel(explained), explained,'-');
set(h,'LineWidth',1.8);
ylim([0 20]);
title('Variance explained by principal components');
xlabel('Principal components');
ylabel('Fraction of variation explained [%]');

% Plot EER evolution
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

%% Sum of explained variance in the optimal value

optimal = 31;
sum_var_optimal = sum(explained(1:optimal, 1));
