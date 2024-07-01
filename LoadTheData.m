clc;clear;
FaultType = {'cf12','cf20','cf30','cf45','eo14','eo32',...
    'eo50','eo68--unsteady test1','fwc10','fwc20','fwc30',...
    'fwc40','fwe10','fwe20','fwe30','fwe40','nc1','nc2',...
    'nc3','nc5','rl10','rl20','rl30','rl40','ro10','ro20',...
    'ro30','ro40','normal r1','normal r','normal nc','normal eo',...
    'normal cf6','normal cf5','normal cf4','normal cf3','normal cf2',...
    'normal cf','normal2','normal1','normal'};
Threshold_Param = 7;

%% Remove Outliers
for i = 1:length(FaultType)
   
    Name = [FaultType{i},'.xls'];
    x = xlsread(Name,'Complete Data Set','B2:BN5192');
    Indicator = zeros(length(x(:,1)),1); % index for the removed outliers
    % remove the data whose deviation from the mean value > 7*std
    for j = 1:length(x(1,:))
        tempSeries = x(:,j);
        
        % first step: remove outliers according to the changing rate 
        ChangingRate = abs(tempSeries(2:end) - tempSeries(1:end-1));
        meanChanging = mean(ChangingRate);
        Thres = Threshold_Param * std(abs(ChangingRate - meanChanging));
        for k = 1:length(ChangingRate)-1
            if abs(ChangingRate(k)-meanChanging)>Thres && abs(ChangingRate(k+1)-meanChanging)>Thres
                Indicator(k+1) = 1;
            end          
        end
        
        % second step: remove outliers according to the value
        meanSeries = mean(tempSeries);
        stdSeries = std(abs(tempSeries - meanSeries));
        DiffSeries = abs(tempSeries - meanSeries);
        Indicator(DiffSeries>Threshold_Param*stdSeries) = 1;
        
    end
    x(Indicator==1,:) = [];
    save([FaultType{i},'.mat'],'x');
    disp(i)
    
end   


