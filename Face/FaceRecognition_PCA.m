%%
%cd ..
%%
close all
clear all
clc

% Divide the number of images per user for Train and Test:
Train=6;
Test=10-Train;

% we add paths to facilitate the code
addpath(cd)
addpath('DetPlots')
cd FaceDatabaseATT

dirListA=dir;
dirList=dirListA(4:43);

% each image is 92px x 112px
width=92*112;

%Initialize the Feature and Label Matrix
MatrixTrainFeats=zeros(Train*40, width);
MatrixTestFeats=zeros(Test*40, width);
MatrixTrainLabels=zeros(Train*40,1); %each row contains the ID of the user
MatrixTestLabels=zeros(Test*40,1);

for i=1:numel(dirList) %Loop for each user

    % Move directory to current user
    cd(dirList(i).name);
    images=dir('*.pgm');

    %%% Feature extraction for Training Dataset

    for j=1:Train %Train images
       im=imread(images(j).name);
       im=double(im);

       % Convert image to row vector (flatten)
       im = reshape(im.',1,[]);

       % Fill train and test matrix
       MatrixTrainFeats((i-1)*Train + j,:) = im;
       MatrixTrainLabels((i-1)*Train + j,1)=i; %user i
    end

      %%% Feature extraction for Test Dataset

    for j=(Train+1):10
       im=imread(images(j).name);
       im=double(im);

        %%Feat Extraction

       im = reshape(im.',1,[]);

       MatrixTestFeats((i-1)*Test + j, :) = im;
       MatrixTestLabels((i-1)*Test + j,1)= i; %user i
    end

    cd ..

end

%%

%Perform PCA on training matrix,
[coeff_PCA, MatrixTrainPCAFeats, ~, ~, explained_var, mu] = pca(MatrixTrainFeats);
MatrixTestPCAFeats = (MatrixTestFeats - mu)*coeff_PCA;
[~, N] = size(MatrixTrainPCAFeats);

%%

%Similarity Computation Stage

% Each test image is going to be compared to each of the training images
% for each of the 40 users (N comparisons). Then, the final score is the
% result of the min of the N comparisons of each test image with the N
% training images of each user, which can be a genuine comparison (target)
% or an impostor comparison (NonTarget)

% Repeat for different numbers of PCA components
min_components = 31; % change at will
max_components = 31; % change at will
E = zeros(max_components - min_components + 1, 1);
for l=min_components:max_components
    MatrixTestPCAFeatsReduced = MatrixTestPCAFeats(:, 1:l);
    MatrixTrainPCAFeatsReduced = MatrixTrainPCAFeats(:, 1:l);
    
    TargetScores=[];
    NonTargetScores=[];

  for i=1:numel(MatrixTestLabels) %For each Test image
      contTest=1;
      for j=1:numel(MatrixTrainLabels) %Comparison with each Training image

          my_distance(contTest)=mean(abs(MatrixTestPCAFeatsReduced(i,:)-MatrixTrainPCAFeatsReduced(j,:))); %Compute the distance measure

          if(MatrixTestLabels(i,:)==MatrixTrainLabels(j,:)) %if it's a genuine comparison
              LabelTest(contTest)=1;

          else % otherwise
              LabelTest(contTest)=0;
          end
          contTest=contTest+1;

      end

      %The final score is the min of the 6 comparisons of each Test image against the training images of each user
      contF=1;
      for k=1:Train:numel(my_distance)
          my_distanceRed(contF)=min(my_distance(k:k+Train-1)); %Extract the scores of the N training signatures and select the min

          if LabelTest(k)==1 %target score
              TargetScores=[TargetScores, my_distanceRed(contF)];
          else %non target score
              NonTargetScores=[NonTargetScores, my_distanceRed(contF)];
          end

          contF=contF+1;
      end

  end

  %Multiply by -1 to have higher values for genuine comparisons, as we have a distance computation. With other type of classifier this wouldn't be necessary.

  TargetScores=-TargetScores;
  NonTargetScores=-NonTargetScores;

  %cd ..

  %save('ParametrizaATT','TargetScores','NonTargetScores');

  E(l-min_components +1) = Eval_Det(TargetScores,NonTargetScores,'b'); %Plot Det curve

end
cd ..

% Plot explained variance
figure
h=plot(1:N, explained_var,'-');
set(h,'LineWidth',1.8)
ylim([0 20])
title('Variance explained by principal components')
xlabel('Principal components')
ylabel('Fraction of variation explained [%]')

% Plot EER evolution
n_components_range=min_components:max_components;
figure
h = plot(n_components_range, E,'-');
set(h,'LineWidth',1.8)
ylim([0 25])
title('Evolution of EER with principal components')
xlabel('Principal components')
ylabel('EER [%]')

% Sum of explained variance in the optimum value
sum_var_optimum = sum(explained_var(1:31, 1)) 
