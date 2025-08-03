%% Audio Classification Using Hybrid MFCC & FFT Features with Filtering (No Plotting, Weighted Features)
clear all; close all; clc;

%% Settings and Folders
class1Dir = 'class_1';
class2Dir = 'class_2';
unknownDir = 'unknown';

% Filter parameters (optional pre-processing)
applyFilter = true;
fsFilter = 44100;
f_low = 700;    
f_high = 2000;  
filterOrder = 4;
if applyFilter
    [b, a] = butter(filterOrder, [f_low, f_high] / (fsFilter / 2), 'bandpass');
end

% Weight factors for the hybrid feature components
wMFCC = 2;       % Weight for MFCC features
wFFT = 1;        % Weight for the average FFT magnitude
wBand = 1;       % Weight for the standard deviation of FFT

%% 1. Extract Training Features for class_1 and class_2
featuresClass1 = [];
featuresClass2 = [];

filesClass1 = dir(fullfile(class1Dir, '*.wav'));
filesClass2 = dir(fullfile(class2Dir, '*.wav'));

for i = 1:length(filesClass1)
    filePath = fullfile(class1Dir, filesClass1(i).name);
    feat = extractHybridFeatures(filePath, applyFilter, b, a, wMFCC, wFFT, wBand);
    featuresClass1 = [featuresClass1; feat];
end

for i = 1:length(filesClass2)
    filePath = fullfile(class2Dir, filesClass2(i).name);
    feat = extractHybridFeatures(filePath, applyFilter, b, a, wMFCC, wFFT, wBand);
    featuresClass2 = [featuresClass2; feat];
end

% Combine all features for normalization
allTrainFeatures = [featuresClass1; featuresClass2];
mu = mean(allTrainFeatures, 1);
sigma = std(allTrainFeatures, [], 1);
featuresClass1_norm = (featuresClass1 - mu) ./ sigma;
featuresClass2_norm = (featuresClass2 - mu) ./ sigma;

% Compute centroids (mean feature vector) for each class in the normalized space
centroidClass1 = mean(featuresClass1_norm, 1);
centroidClass2 = mean(featuresClass2_norm, 1);

%% 2. Process Unknown Files and Classify Based on Normalized Features
filesUnknown = dir(fullfile(unknownDir, '*.wav'));
results = struct('filename',{},'distanceClass1',{},'distanceClass2',{},'assignedClass',{});

for i = 1:length(filesUnknown)
    filePath = fullfile(unknownDir, filesUnknown(i).name);
    featUnknown = extractHybridFeatures(filePath, applyFilter, b, a, wMFCC, wFFT, wBand);
    featUnknown_norm = (featUnknown - mu) ./ sigma;
    
    % Compute Euclidean distances in normalized space using the average distance 
    % from the unknown feature to every sample of each class.
    d1 = mean(vecnorm(featuresClass1_norm - featUnknown_norm, 2, 2));
    d2 = mean(vecnorm(featuresClass2_norm - featUnknown_norm, 2, 2));
    
    if d1 < d2
        assigned = 'class_1';
    else
        assigned = 'class_2';
    end
    
    results(i).filename = filesUnknown(i).name;
    results(i).distanceClass1 = d1;
    results(i).distanceClass2 = d2;
    results(i).assignedClass = assigned;
end

%% Display the Classification Results
fprintf('--- Normalized Classification Results for Unknown Files ---\n');
for i = 1:length(results)
    fprintf('File: %s -> Assigned to: %s (d1=%.2f, d2=%.2f)\n', ...
        results(i).filename, results(i).assignedClass, results(i).distanceClass1, results(i).distanceClass2);
end

%% Local Function Definitions
function featVector = extractHybridFeatures(filePath, applyFilter, b, a, wMFCC, wFFT, wBand)
    % Reads an audio file, applies optional filtering, computes MFCCs and FFT features,
    % and returns a weighted, combined feature vector.
    
    [audio, fs] = audioread(filePath);
    
    % Convert stereo to mono if necessary
    if size(audio,2) > 1
       audio = mean(audio, 2);
    end
    
    % Optionally apply bandpass filter (if applyFilter is false, the raw audio is used)
    if applyFilter
        audio = filter(b, a, audio);
    end
    
    % --------- MFCC Feature Extraction -----------
    % Use 13 coefficients (as standard)
    coeffs = mfcc(audio, fs, 'NumCoeffs', 13);
    mfcc_mean = mean(coeffs, 1);
    
    % --------- FFT-Based Feature Extraction -----------
    N = length(audio);
    fftSpectrum = abs(fft(audio));
    fftSpectrum = fftSpectrum(1:floor(N/2));  % Take the first half (positive frequencies)
    fft_mean = mean(fftSpectrum);
    fft_std = std(fftSpectrum);
    
    % Apply weight factors and combine the features:
    featVector = [wMFCC * mfcc_mean, wFFT * fft_mean, wBand * fft_std];
end







