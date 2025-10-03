function image_classification_gui
    handles.f = figure('Position', [100, 100, 600, 400], 'MenuBar', 'none', ...
                       'Name', 'Klasifikasi Penyakit Tanaman Jagung', 'Resize', 'off', ...
                       'Color', [0.3294, 0.7686, 0.3294]);

    uicontrol('Style', 'text', 'String', 'Klasifikasi Penyakit Tanaman Jagung', ...
              'Position', [50, 350, 500, 30], 'FontSize', 16, 'FontWeight', 'bold', ...
              'BackgroundColor', [0.3294, 0.7686, 0.3294], 'HorizontalAlignment', 'center');

    uicontrol('Style', 'text', 'String', 'Nilai K:', ...
              'Position', [50, 300, 100, 30], 'FontSize', 12, ...
              'BackgroundColor', [0.3294, 0.7686, 0.3294]);
    handles.k_input = uicontrol('Style', 'edit', 'String', '1', ...
                                'Position', [150, 300, 50, 30], 'FontSize', 12);

    uicontrol('Style', 'pushbutton', 'String', 'Training', ...
              'Position', [220, 300, 100, 30], 'FontSize', 12, ...
              'Callback', @(src, event) train_model(handles));

    uicontrol('Style', 'pushbutton', 'String', 'Testing', ...
              'Position', [350, 300, 100, 30], 'FontSize', 12, ...
              'Callback', @(src, event) test_model(handles));

    handles.axes_handle = axes('Units', 'pixels', 'Position', [50, 50, 300, 200], 'Color', 'white');
    axis off;
    
    blank_image = uint8(255 * ones(450, 450)); % Gambar putih ukuran 450x450
    imshow(blank_image, 'Parent', handles.axes_handle);

    handles.result_text = uicontrol('Style', 'text', 'Position', [400, 250, 180, 30], ...
                                    'String', 'Hasil Klasifikasi: ', 'FontSize', 12, ...
                                    'BackgroundColor', [0.3294, 0.7686, 0.3294]);

    handles.status_text = uicontrol('Style', 'text', 'Position', [400, 190, 180, 60], ...
                                    'String', 'Status: Siap!', 'FontSize', 12, ...
                                    'BackgroundColor', [0.3294, 0.7686, 0.3294], 'HorizontalAlignment', 'left');

    guidata(handles.f, handles);
end

function train_model(handles)
    handles = guidata(handles.f);

    K = str2double(get(handles.k_input, 'String'));

    set(handles.status_text, 'String', 'Status: Proses training berlangsung...');
    pause(0.1);

    folder_path = 'C:\Users\Adelia\Documents\MATLAB\skrispi\Program\Training';

    subfolders = dir(folder_path);
    subfolders = subfolders([subfolders.isdir] & ~startsWith({subfolders.name}, '.'));

    global y nama;
    y = [];
    nama = {};
    kel = length(subfolders);

    for t = 1:kel
        folder_name = subfolders(t).name;
        nama{t} = folder_name;

        image_files = dir(fullfile(folder_path, folder_name, '*.*'));
        image_files = image_files(~[image_files.isdir]);

        for z = 1:length(image_files)
            file_path = fullfile(folder_path, folder_name, image_files(z).name);
            r = imresize(imread(file_path), [450 450]);
            i = rgb2gray(r);
            maks = max(i(:)); 
            mins = min(i(:));
            
            features = extract_features(i, mins, maks);
            y = [y; features, t];
        end
    end
    set(handles.status_text, 'String', 'Status: Training selesai!');
end

function test_model(handles)
    handles = guidata(handles.f);

    set(handles.status_text, 'String', 'Status: Proses testing berlangsung...');
    pause(0.1);

    [file, path] = uigetfile('*.*', 'Input file testing');
    if isequal(file, 0)
        set(handles.status_text, 'String', 'Status: Testing dibatalkan');
        return;
    end
    
    r = imresize(imread(fullfile(path, file)), [450 450]);
    i = rgb2gray(r);
    imshow(i, 'Parent', handles.axes_handle);

    maks = max(i(:)); 
    mins = min(i(:));
    tes_features = extract_features(i, mins, maks);
    
    global y nama;
    kelompok = classify_image(tes_features, y, str2double(get(handles.k_input, 'String')));
    
    set(handles.result_text, 'String', ['Hasil Klasifikasi: ', char(nama{kelompok})]);
    set(handles.status_text, 'String', 'Status: Testing selesai!');
end

function features = extract_features(image, mins, maks)
    a = graycomatrix(image, 'Offset', [mins maks], 'Symmetric', true);
    b = graycomatrix(image, 'Offset', [-mins maks], 'Symmetric', true);
    c = graycomatrix(image, 'Offset', [mins -maks], 'Symmetric', true);
    d = graycomatrix(image, 'Offset', [-mins -maks], 'Symmetric', true);
    okuren = round(0.25 * (a + b + c + d));

    features = struct2cell(graycoprops(okuren, 'Contrast'));
    features = [features; struct2cell(graycoprops(okuren, 'Correlation'))];
    features = [features; struct2cell(graycoprops(okuren, 'Energy'))];
    features = [features; struct2cell(graycoprops(okuren, 'Homogeneity'))];
    features = cell2mat(features)'; 
end

function kelompok = classify_image(test_features, training_data, K)
    brs = size(training_data, 1);
    j = zeros(brs, 2);
    
    for p = 1:brs
        j(p, 1) = sqrt(sum((test_features - training_data(p, 1:4)).^2));
        j(p, 2) = training_data(p, 5);
    end
    
    kn = sortrows(j, 1);
    kn = kn(1:K, :);
    
    kelas_unik = unique(kn(:, 2));
    jumlah_kelas = arrayfun(@(x) sum(kn(:, 2) == x), kelas_unik);
    
    [~, idx] = max(jumlah_kelas);
    kelompok = kelas_unik(idx);
end
