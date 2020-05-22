close all;
clear all;
 
% prepare MLP
inputLayerSize = 784 % input layer
hiddenLayerSize = 14 % hidden layer
outputLayerSize = 10 % output layer
rate = 0.3 % learning rate
epochs = 3
pc_count = 40 % number of principle components (dimensionality reduction)
training_size = 60000
test_size = 10000

% initialise MLP
mlp = MLP(inputLayerSize, hiddenLayerSize, outputLayerSize);
mlp.initWeight(1);

% load in dataset
training = loadMNISTImages('train-images-idx3-ubyte');
training_targets_raw = loadMNISTLabels('train-labels-idx1-ubyte');
testing = loadMNISTImages('t10k-images-idx3-ubyte');
testing_targets_raw = loadMNISTLabels('t10k-labels-idx1-ubyte');

% reduce dataset size
training = training(:,1:training_size);
training_targets_raw = training_targets_raw(1:training_size,:);
testing = testing(:,1:test_size);
testing_targets_raw = testing_targets_raw(1:test_size,:);
 
% create one-hot encoded vectors (for output values)
training_targets = zeros(length(training_targets_raw), 10);
for i = 1:length(training_targets_raw)
    training_targets(i, training_targets_raw(i,1)+1) = 1;
end
testing_targets = zeros(length(testing_targets_raw), 10);
for i = 1:length(testing_targets_raw)
    testing_targets(i, testing_targets_raw(i,1)+1) = 1;
end
 
% dimensionality reduction (training)
[pc_coeff,score,var,~,~,mu] = pca(training);
figure('Name','Variance by principle componeent')
bar(var) % show variance by principle component
xlabel('Principle component') 
ylabel('Variance') 
recon = score(:,1:pc_count) * pc_coeff(:,1:pc_count)' + repmat(mu, size(score, 1), 1);
training = recon;
 
% dimensionality reduction (testing)
[pc_coeff,score,var,~,~,mu] = pca(testing);
recon = score(:,1:pc_count) * pc_coeff(:,1:pc_count)' + repmat(mu, size(score, 1), 1);
testing = recon;
 
% begin training
total_training_error = [];
for l = 1:epochs
    fprintf('EPOCH %i \n', l)
    epoch_training_error = [];
    for k = 1:size(training, 2)
        fprintf('DATA %i \n', k)
        current = training(:,k);
        target = training_targets(k,:);
        current_training_error = mlp.adapt_to_target(current, target, rate);
        epoch_training_error = [epoch_training_error current_training_error];
    end
    total_training_error = [total_training_error mean(epoch_training_error)];
end
figure('Name', 'All training data points');
gscatter(training(1,:), training(1,:), training_targets)
title('(TRAINING) all training data points')
figure('Name', 'Error rate by epoch');
scatter(1:epochs, total_training_error)
title('(TRAINING) error rate by epoch')
mean_training_error = mean(total_training_error)
last_epoch_training_error = total_training_error(epochs)
 
% begin testing
total_testing_error = [];
for k = 1:size(testing, 2)
    fprintf('DATA %i \n', k)
    current = testing(:,k);
    target = testing_targets(k,:);
    output = mlp.compute_output(current);
    current_testing_error = abs(abs(output) - target);
    total_testing_error = [total_testing_error current_testing_error];
end
figure('Name', 'All testing data points');
gscatter(testing(1,:), testing(1,:), testing_targets)
title('(TESTING) all testing data points')
 
fprintf('Mean training error was: %f \n', mean(mean_training_error))
fprintf('Final training epoch error was: %f \n', last_epoch_training_error)
fprintf('Mean testing error was: %f \n', mean(total_testing_error))
