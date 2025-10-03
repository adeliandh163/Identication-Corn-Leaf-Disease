clear all; 
clc; 
klaster = '';
K = input('Masukkan nilai K = ');

folder_path = 'C:\Users\Adelia\Documents\MATLAB\skrispi\Program\Training';
test_folder = 'C:\Users\Adelia\Documents\MATLAB\skrispi\Program\Training2';

subfolders = dir(folder_path);
subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.'));
kel = length(subfolders);
nama = {};
y = {};

training_results = table('Size', [0, 6], 'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, 'VariableNames', {'Class', 'Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Label'});

for t = 1:kel
    folder_name = subfolders(t).name;
    nama{t} = folder_name;
    image_files = dir(fullfile(folder_path, folder_name, '*.*'));
    image_files = image_files(~[image_files.isdir]);
    
    for z = 1:length(image_files)
        file_path = fullfile(folder_path, folder_name, image_files(z).name);
        r = imresize(imread(file_path), [450 450]);
        i = rgb2gray(r);
        maks = max(max(i)); 
        mins = min(min(i));
        
        a = graycomatrix(i, 'Offset', [mins maks], 'Symmetric', true);
        b = graycomatrix(i, 'Offset', [-mins maks], 'Symmetric', true);
        c = graycomatrix(i, 'Offset', [mins -maks], 'Symmetric', true);
        d = graycomatrix(i, 'Offset', [-mins -maks], 'Symmetric', true);
        okuren = round(0.25 * (a + b + c + d));
        
        train_features = struct2cell(graycoprops(okuren, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'}));
        train_features = cell2mat(train_features');
        
        training_results = [training_results; {folder_name, train_features(1), train_features(2), train_features(3), train_features(4), t}];
        y = [y; {train_features(1), train_features(2), train_features(3), train_features(4), t}];
    end
end

disp('Hasil Training:');
disp(training_results);

y = cell2mat(y);
[brs, ~] = size(y);

subfolder_test = dir(test_folder);
subfolder_test = subfolder_test([subfolder_test.isdir] & ~startsWith({subfolder_test.name}, '.'));

results = table('Size', [0, 6], 'VariableTypes', {'string', 'string', 'double', 'double', 'double', 'double'}, 'VariableNames', {'ImageName', 'PredictedClass', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

actual_labels = {};

for sf = 1:length(subfolder_test)
    test_subfolder = subfolder_test(sf).name;
    test_files = dir(fullfile(test_folder, test_subfolder, '*.*'));
    test_files = test_files(~[test_files.isdir]);
    
    for f = 1:length(test_files)
        file_path = fullfile(test_folder, test_subfolder, test_files(f).name);
        r = imresize(imread(file_path), [450 450]);
        i = rgb2gray(r);
        maks = max(max(i)); 
        mins = min(min(i));
        
        a = graycomatrix(i, 'Offset', [mins maks], 'Symmetric', true);
        b = graycomatrix(i, 'Offset', [-mins maks], 'Symmetric', true);
        c = graycomatrix(i, 'Offset', [mins -maks], 'Symmetric', true);
        d = graycomatrix(i, 'Offset', [-mins -maks], 'Symmetric', true);
        okuren = round(0.25 * (a + b + c + d));
        
        test_features = struct2cell(graycoprops(okuren, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'}));
        test_features = cell2mat(test_features');
        
        j = zeros(brs, 2);
        for p = 1:brs
            j(p,:) = [sqrt(sum((test_features - y(p,1:4)).^2)), y(p,5)];
        end
        
        kn = topkrows(j, K, 'ascend');
        Kn = num2str(kn(:,2)');
        
        tot = zeros(1, kel);
        for k = 1:kel
            tot(k) = count(Kn, num2str(k));
        end
        
        [~, kelompok] = max(tot);
        predicted_class = string(nama(kelompok));
        
        results = [results; {test_files(f).name, predicted_class, test_features(1), test_features(2), test_features(3), test_features(4)}];
        actual_labels = [actual_labels; test_subfolder];
    end
end

disp('Hasil Klasifikasi:');
disp(results);

actual_labels = categorical(actual_labels);
predicted_labels = categorical(results.PredictedClass);

conf_mat = confusionmat(actual_labels, predicted_labels);
class_names = categories(actual_labels);

if size(conf_mat, 1) ~= length(class_names)
    class_names = unique([actual_labels; predicted_labels]);
end

accuracy = sum(diag(conf_mat)) / sum(conf_mat(:));

disp('===============================================');
fprintf('Nilai Akurasi: %.2f%%\n', accuracy * 100);
disp(' ');

valid_var_names = cellfun(@(x) matlab.lang.makeValidName(x), cellstr(class_names), 'UniformOutput', false);

conf_table = array2table(conf_mat, 'RowNames', valid_var_names, 'VariableNames', valid_var_names);
disp('Tabel Hasil Klasifikasi:');
disp(conf_table);

save('hasil_klasifikasi2.mat', 'training_results', 'results', 'conf_table');
