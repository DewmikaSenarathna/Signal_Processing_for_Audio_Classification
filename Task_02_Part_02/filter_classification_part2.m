clear all; close all; clc;

%% ------------------ SETUP PATHS ------------------
basePath = 'C:\Users\dewmi\Desktop\Task_02\Task_02_Part_02\part 02\Filter'; 

trainAmbulancePath = fullfile(basePath, 'train', 'ambulance');
trainFiretruckPath = fullfile(basePath, 'train', 'firetruck');
testAmbulancePath  = fullfile(basePath, 'test', 'ambulance');
testFiretruckPath  = fullfile(basePath, 'test', 'firetruck');

Fs = 16000;
NFFT = 2048;

%% ------------------ DESIGN FILTERS ------------------
bp1 = designfilt('bandpassiir','FilterOrder',6,...
    'HalfPowerFrequency1', 500, 'HalfPowerFrequency2', 1200, 'SampleRate', Fs);
bp2 = designfilt('bandpassiir','FilterOrder',6,...
    'HalfPowerFrequency1', 1500, 'HalfPowerFrequency2', 2500, 'SampleRate', Fs);
bp3 = designfilt('bandpassiir','FilterOrder',6,...
    'HalfPowerFrequency1', 3000, 'HalfPowerFrequency2', 4200, 'SampleRate', Fs);

get_energy_features = @(x) [
    sum(filtfilt(bp1, x).^2);
    sum(filtfilt(bp2, x).^2);
    sum(filtfilt(bp3, x).^2)
];

%% ------------------ EXTRACT TRAINING FEATURES ------------------
disp("Extracting Training Features...");

filesAmb = dir(fullfile(trainAmbulancePath, '*.wav'));
filesFire = dir(fullfile(trainFiretruckPath, '*.wav'));

Xtrain = [];
Ytrain = [];

% Process Ambulance training files
for k = 1:length(filesAmb)
    [x, ~] = audioread(fullfile(filesAmb(k).folder, filesAmb(k).name));
    feat = get_energy_features(x);
    ratio1 = feat(1) / (feat(2) + eps);
    ratio2 = feat(2) / (feat(3) + eps);
    Xtrain = [Xtrain; ratio1, ratio2];
    Ytrain = [Ytrain; 1]; % 1 = Ambulance
end

% Process Firetruck training files
for k = 1:length(filesFire)
    [x, ~] = audioread(fullfile(filesFire(k).folder, filesFire(k).name));
    feat = get_energy_features(x);
    ratio1 = feat(1) / (feat(2) + eps);
    ratio2 = feat(2) / (feat(3) + eps);
    Xtrain = [Xtrain; ratio1, ratio2];
    Ytrain = [Ytrain; 0]; % 0 = Firetruck
end

%% ------------------ TRAIN CLASSIFIER ------------------
disp("Training k-NN Classifier...");
mdl = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 3);

%% ------------------ PREDICT AND EVALUATE ------------------
disp("Classifying and Checking File Folder Matches...");

% Get test files
testAmbFiles = dir(fullfile(testAmbulancePath, '*.wav'));
testFireFiles = dir(fullfile(testFiretruckPath, '*.wav'));
testFiles = [testAmbFiles; testFireFiles];

Xtest = [];
Ytrue = [];      % Ground truth labels from folder names
fileNames = {};
folderTypes = {};  % To hold folder info (for display)

for k = 1:length(testFiles)
    [x, ~] = audioread(fullfile(testFiles(k).folder, testFiles(k).name));
    feat = get_energy_features(x);
    ratio1 = feat(1) / (feat(2) + eps);
    ratio2 = feat(2) / (feat(3) + eps);
    Xtest = [Xtest; ratio1, ratio2];

    % Determine ground truth label by folder name
    if contains(lower(testFiles(k).folder), 'ambulance')
        Ytrue = [Ytrue; 1];
        folderTypes{end+1} = 'Ambulance';
    else
        Ytrue = [Ytrue; 0];
        folderTypes{end+1} = 'Firetruck';
    end

    fileNames{end+1} = testFiles(k).name;
end

% Predict test data using method syntax to avoid errors
predicted = mdl.predict(Xtest);

% Calculate accuracy
accuracy = sum(predicted == Ytrue) / length(Ytrue);
fprintf("\n Final Classification Accuracy: %.2f%%\n\n", accuracy * 100);

% Define label names for printing
labels = ["Firetruck", "Ambulance"];

fprintf(" Classification Result Per File:\n\n");
for k = 1:length(predicted)
    actual = Ytrue(k);         % Actual class from folder
    predict = predicted(k);    % Predicted class

    % Check if prediction matches the actual label
    if actual == predict
        status = "Accept";
    else
        status = "Reject";
    end

    % Get predicted label string
    labelName = labels(predict + 1);

    fprintf(" %-20s | Predicted: %-10s | Folder: %-10s | %s\n", ...
        fileNames{k}, labelName, folderTypes{k}, status);
end














