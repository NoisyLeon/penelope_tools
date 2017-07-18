function scan_I_E

%%
I = [0.01 0.05 0.10 0.50 1 2 5 8 10 100];
E = [30 25 20 15 10 5];

%% READ IN PROBE BEAM DIAMTER TABLE

    filename = 'beam_diameter_table_4mm.txt';
    delimiter = ' ';
    formatSpec = '%s%s%s%s%s%s%s%[^\n\r]';
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true,  'ReturnOnError', false);
    fclose(fileID);

    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = dataArray{col};
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));

    for col=[1,2,3,4,5,6,7]
        rawData = dataArray{col};
        for row=1:size(rawData, 1);
            regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
            try
                result = regexp(rawData{row}, regexstr, 'names');
                numbers = result.numbers;

                invalidThousandsSeparator = false;
                if any(numbers==',');
                    thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                    if isempty(regexp(numbers, thousandsRegExp, 'once'));
                        numbers = NaN;
                        invalidThousandsSeparator = true;
                    end
                end

                if ~invalidThousandsSeparator;
                    numbers = textscan(strrep(numbers, ',', ''), '%f');
                    numericData(row, col) = numbers{1};
                    raw{row, col} = numbers{1};
                end
            catch me
            end
        end
    end

    probe_current = cell2mat(raw(:, 1));
    spot_size_30keV = cell2mat(raw(:, 2));
    spot_size_25keV = cell2mat(raw(:, 3));
    spot_size_20keV = cell2mat(raw(:, 4));
    spot_size_15keV = cell2mat(raw(:, 5));
    spot_size_10keV = cell2mat(raw(:, 6));
    spot_size_5keV = cell2mat(raw(:, 7));

    spot_size_all = [spot_size_30keV spot_size_25keV spot_size_20keV spot_size_15keV spot_size_10keV spot_size_5keV];
    
    clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me;
    
%% CALCULATE CONICAL ANGLE
    wd = 4e6; %working distance (nm)

    con_angle_30keV = atan(spot_size_30keV/(2*wd)) * 180/pi; %conical angle (degrees)
    con_angle_25keV = atan(spot_size_25keV/(2*wd)) * 180/pi; %conical angle (degrees)
    con_angle_20keV = atan(spot_size_20keV/(2*wd)) * 180/pi; %conical angle (degrees)
    con_angle_15keV = atan(spot_size_15keV/(2*wd)) * 180/pi; %conical angle (degrees)
    con_angle_10keV = atan(spot_size_10keV/(2*wd)) * 180/pi; %conical angle (degrees)
    con_angle_5keV = atan(spot_size_5keV/(2*wd)) * 180/pi; %conical angle (degrees)

    format long

    display(con_angle_30keV);
    display(con_angle_25keV);
    display(con_angle_20keV);
    display(con_angle_15keV);
    display(con_angle_10keV);
    display(con_angle_5keV);
    
    con_ang_all = [con_angle_30keV con_angle_25keV con_angle_20keV con_angle_15keV con_angle_10keV con_angle_5keV];

for gg = 1:length(1:length(E))
%%
    for jj = 1:length(I)

        display(['I = ',num2str(I(jj))])

        %FIND conical angle IN TABLE
        conical_angle = con_ang_all(jj,gg);

        SENERG_new = ['SENERG ',num2str(E(gg)),'e3             [Initial energy (monoenergetic sources only)]'];
        SCONE_new = ['SCONE  0 0 ',num2str(conical_angle),'                   [Conical beam; angles in deg]'];
        NBE_new = ['NBE    0  ',num2str(E(gg)),'e3  100                   [Energy window and no. of bins]'];

        clear fid tline i

        % Read *.in line by line
        fid = fopen('au_disc.in','r');
        i = 1;
        tline = fgetl(fid);
        A{i} = tline;
        while ischar(tline)
            i = i+1;
            tline = fgetl(fid);
            A{i} = tline;
        end
        fclose(fid);

        % Replace line 5, 7, 18
        % Change cell A
        A{5} = SENERG_new;
        A{7} = SCONE_new;
        A{18} = NBE_new;

        clear fid i
        % Write cell A into txt
        fid = fopen('au_disc.in', 'w');
        for i = 1:numel(A)
            if A{i+1} == -1
                fprintf(fid,'%s', A{i});
                break
            else
                fprintf(fid,'%s\n', A{i});
            end
        end
        fclose(fid);


    %%    
        % Run PENELOPE

        system(['rm angle.dat dump.dat energy-down.dat energy-up.dat geometry.rep material.dat penmain.dat penmain-res.dat '...
            'polar-angle.dat psf-test.dat'])

        [status,cmdout] = system('penmain.exe < au_disc.in');
        dirname = ['I_',num2str(I(jj)),'nA_E_',num2str(E(gg)),'keV'];

        system(['mkdir ',dirname]);
        system(['mv angle.dat ',dirname])
        system(['mv energy-down.dat ',dirname])
        system(['mv energy-up.dat ',dirname])
        system(['mv geometry.rep ',dirname])
        system(['cp au_disc.in ',dirname])
        system(['mv material.dat ',dirname])
        system(['mv penmain.dat ',dirname])
        system(['mv penmain-res.dat ',dirname])
        system(['mv polar-angle.dat ',dirname])
        system(['mv psf-test.dat ',dirname])

    end
    
end

end
