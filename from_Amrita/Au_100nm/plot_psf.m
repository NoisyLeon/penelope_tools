function plot_psf(filename)

    %filename = 'C:\Users\amrita.masurkar\Desktop\RAVEN\penelope\xray_source\psf-test1.dat';
    formatSpec = '%12s%21s%31s%26s%26s%21s%26s%26s%26s%17s%12s%12s%12s%s%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '',  'ReturnOnError', false);
    fclose(fileID);
    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = dataArray{col};
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));
    for col=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        % Converts strings in the input cell array to numbers. Replaced non-numeric
        % strings with NaN.
        rawData = dataArray{col};
        for row=1:size(rawData, 1);
            % Create a regular expression to detect and remove non-numeric prefixes and
            % suffixes.
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;

                % Detected commas in non-thousand locations.
                invalidThousandsSeparator = false;
                if any(numbers==',');
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(numbers, thousandsRegExp, 'once'));
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end
                % Convert numeric strings to numbers.
                if ~invalidThousandsSeparator;
                    numbers = textscan(strrep(numbers, ',', ''), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch me
            end
        end
    end
    R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
    raw(R) = {NaN}; % Replace non-numeric cells

    %% Allocate imported array to column variable names
    kpar = cell2mat(raw(:, 1));
    energy = cell2mat(raw(:, 2));
    x = cell2mat(raw(:, 3));
    y = cell2mat(raw(:, 4));
    z = cell2mat(raw(:, 5));
    u = cell2mat(raw(:, 6));
    v = cell2mat(raw(:, 7));
    w = cell2mat(raw(:, 8));
    VarName9 = cell2mat(raw(:, 9));
    VarName10 = cell2mat(raw(:, 10));
    VarName11 = cell2mat(raw(:, 11));
    VarName12 = cell2mat(raw(:, 12));
    VarName13 = cell2mat(raw(:, 13));
    VarName14 = cell2mat(raw(:, 14));

    %% PLOT SPECTRA
    figure(1)
    subplot(2,1,1)
    h = histogram(energy,15000,'FaceColor','none','EdgeColor','r');
    grid on
    
    for ii = 1:length(h.BinEdges)-1
        eng(ii) = (h.BinEdges(ii)+h.BinEdges(ii+1))/2;
    end
    subplot(2,1,2)
    semilogy(eng,h.Values)
    grid on
    
    %% PLOT PSF IN X-Y-Z
    figure(2)
    scatter3(x,y,z,'.')
    xlabel('X (cm)')
    ylabel('Y (cm)')
    zlabel('Z (cm)')
    axis([-1e-5 1e-5 -1e-5 1e-5 0 1e-5])
    grid on
    
    %% PLOT DEPTH PROFILE
    
    hh = 0:1e-7:1e-5;
    z_prof = [];
    for ii = 1:length(hh)-1
        loo = find((z > hh(ii)) && (z < hh(ii + 1)));
        z_prof(ii) = length(loo);
    end
    figure(3)
    plot(hh(1:end-1),z_prof)
    
    %% PLOT X-Y PROFILE @ Z = 0
    goo = find(z < 1e-7);
    figure(4);scatter(x(goo),y(goo))
    axis equal
    
    %% Clear temporary variables
    clearvars filename formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me R;

end