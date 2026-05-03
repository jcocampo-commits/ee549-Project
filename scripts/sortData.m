%Name: John Ocampo
%Date: 2026/04/13
%Desc: Seperate both datasets to training and testing

prefixDir='zzz_outputs*'
path='C:\Users\Jeco\Documents\School\EE559\ee549-Project\';

dataDirs=dir([path, filesep, prefixDir]);
outPath=[path,'sorted'];

fileList=ls(fullfile(dataDirs(1).folder, dataDirs(1).name));
fileList(1:2,:)=[];% remove the . and .. directories

classList=extractBefore(string(fileList), '_');
classList=str2num(char(classList));

numFiles=size(fileList,1);
numList=1:numFiles;

classUniq=unique(classList);
testIndx=[];
for u= 1:length(classUniq)
    tempList= classUniq(u)==classList;
    tempFirst= find(tempList==1,1)-1;

    tempLen=sum(tempList);%
    trainSz=ceil(.8*tempLen); %80-percent
    testSz=tempLen-trainSz;%20-percent
    
    testList=randperm(tempLen,testSz)+ tempFirst;

    testIndx=[testIndx, numList(testList)];
end
trainIndx=numList(~ismember(numList,testIndx));%everything else

%%
for i= 1:length(dataDirs)
    
    tempPar=fullfile(dataDirs(i).folder, dataDirs(i).name);
    fileList=ls(tempPar);
    fileList(1:2,:)=[];
    outPathData=fullfile(outPath, dataDirs(i).name)
    
    outPathTest=fullfile(outPathData, 'test');
    mkdir(outPathTest)
    for j=testIndx
        copyfile(fullfile(tempPar, fileList(j,:)), fullfile(outPathTest, fileList(j,:)))
    end
    
    outPathTrain=fullfile(outPathData, 'train');
    mkdir(outPathTrain)
    for j=trainIndx
        copyfile(fullfile(tempPar, fileList(j,:)), fullfile(outPathTrain, fileList(j,:)))
    end

end

