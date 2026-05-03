%Name: John Ocampo
%Date: 2026/04/08
%Desc: Collect then check consistency of radar parameters


genPath='D:\MASS\Downloads\2026\24ghz\'
dirs=ls(genPath);
dirs(1:2,:)=[];%can't let it pick the first index

%%
store=[]
for i=1:length(dirs)
    tempPath=[genPath,dirs(i,:)];
    files=ls(tempPath);
    files(1:2,:)=[];
    
    
    indxDat=contains(string(files),'.dat')%filter by key file extension
    files=files(indxDat)

    for j=1:length(files)
        fNameIn=[tempPath,filesep,files(j,:)];
        fileID = fopen(fNameIn, 'r');
        dataArray = textscan(fileID, '%f');
        fclose(fileID);
        radarData = dataArray{1};
        store=[store,radarData(1:4)];
    end
end