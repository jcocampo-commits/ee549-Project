
% Name: John Ocampo
% Date: 2026/04/02
% Desc: Convert the 24ghz radar returns to RDMs
% Credit: modified and was originally based on: datToImage_Anchortech.m

%File Selection
%fNameIn='D:\MASS\Downloads\2026\24ghz\05_walking_towards\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\06_walking_away\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\07_picking_obj\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\08_Bending\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\09_sitting\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\10_kneeling\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\11_crawling\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\16_walking_on_toes\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\17_limping_RL_stiff\'
%fNameIn='D:\MASS\Downloads\2026\24ghz\18_short_steps\';
%fNameIn='D:\MASS\Downloads\2026\24ghz\19_scissor_gait\';

% files=ls(fNameIn);%select random files
% files(1:2,:)=[];%can't let it pick the first index
% randIndx=randi(size(files,1));
% fNameIn=[fNameIn,files(randIndx,:)]

fNameOut='D:\MASS\Downloads\2026\24ghz\zzz_outputs\';
mkdir(fNameOut)

dirs=dir('D:\MASS\Downloads\2026\24ghz\**\*.dat');
randIndx=randperm(length(dirs),1);

parfor i=1:length(dirs)%only use parfor to save on time, otherwise deubgging should be done with for
%fNameIn=[dirs(randIndx).folder, filesep, dirs(randIndx).name];
fNameIn=[dirs(i).folder, filesep, dirs(i).name];

%close all
%% Original Function

fileID = fopen(fNameIn, 'r');
dataArray = textscan(fileID, '%f');
fclose(fileID);
radarData = dataArray{1};
%clearvars fileID dataArray ans;
fc = radarData(1); % Center frequency
Tsweep = radarData(2); % Sweep time in ms
Tsweep=Tsweep/1000; %then in sec
NTS = radarData(3); % Number of time samples per sweep
Bw = radarData(4); % FMCW Bandwidth. For FSK, it is frequency step;
% For CW, it is 0.
Data = radarData(5:end); % raw data in I+j*Q format

fs=NTS/Tsweep; % sampling frequency ADC
record_length=length(Data)/NTS*Tsweep; % length of recording in s
nc=record_length/Tsweep; % number of chirps

% Reshape data into chirps and do range FFT (1st FFT)
Data_time=reshape(Data, [NTS nc]);% note: unwanted things past this point were deleted

%% Convolution => Matrix Organization => FFT => dB-Normalization
%refPulse= Data_time(:,1); % uses first pulse from each sample as reference
ldStruct=load('refPulse_sitting.mat'); %reference is taken from one of the sitting samples
refPulse=ldStruct.refPulse;
fullData= Data_time(:);

%Checking if data was flipped
% refPulse= Data_time(1,:);%the weird part is it looks more like FMCW now
% fullData= reshape(Data_time.', 1, []);

cmprsSig=xcorr(fullData,refPulse);
cmprsSig(1:length(fullData)-1)=[];
cmprsMat=reshape(cmprsSig, [NTS nc]);

rdm=abs(fft(cmprsMat,[],2));%Correct FFT to generate RDM
%rdm=abs(fft(cmprsMat,[],1));% appling FFT to range or fast time
rdm_norm=20*log10(rdm/max(max(rdm)));

rdm_limit=max(rdm_norm,-120); %acts as a floor to data
%rdm_limit=min(rdm_norm, -60); %acts as a ceil to data
%rdm_limit=rdm_norm; %no floor or ceil 

% Recenter
rdm_limit=circshift(rdm_limit,NTS/2,1);
rdm_limit=circshift(rdm_limit,nc/2,2);

% Test Plot
% mesh(rdm_limit)
% view(0,90)
% pause()

% Plot and Save
f = figure('Visible', 'off'); 
filesepIndx=strfind(fNameIn,filesep);
filesepIndx=filesepIndx(end-1);

finName=strcat([fNameOut,replace(fNameIn(filesepIndx+1:end-4),'\','_'),'.png'])
plotSave(rdm_limit, finName)
close(f);

end
return

%% test functions


%% Other Functions
function mplot(input)
plot(real(input)); hold on;
plot(imag(input)); hold off
end

function aplot(input)
plot(atan(imag(input)./real(input)))
end

function chkPulses(Data_time)
    for i =1:200:size(Data_time,2)
        plot(real(Data_time(:,i))); hold on
        plot(imag(Data_time(:,i))); hold off
        pause(.1)
    end
end

function plotSave(RDM, finName)
    colormap(parula(256));
    %mesh(RDM); view(0,90)
    ax=gca; imagesc(RDM); ax.YDir="normal";
    frame = frame2im(getframe(gca));
    imwrite(frame, finName);
end