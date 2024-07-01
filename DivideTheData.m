clc;clear;
Number_of_labeled_data = 100;
Number_of_unlabeled_data = 20000;
LabeledPercentage = Number_of_labeled_data/162379;
UnlabeledPercentage = Number_of_unlabeled_data/162379;
TestPercentage = 0.1;
FaultType = {'cf12','cf20','cf30','cf45','eo14','eo32',...
    'eo50','eo68--unsteady test1','fwc10','fwc20','fwc30',...
    'fwc40','fwe10','fwe20','fwe30','fwe40','nc1','nc2',...
    'nc3','nc5','rl10','rl20','rl30','rl40','ro10','ro20',...
    'ro30','ro40','normal r1','normal r','normal nc','normal eo',...
    'normal cf6','normal cf5','normal cf4','normal cf3','normal cf2',...
    'normal cf','normal2','normal1','normal'};
Inputx = [];
Labelx = [];
LabelxWithSeverity = [];
for i = 1:28
    Name = [FaultType{i},'.mat'];
    load(Name);
    Inputx = [Inputx;x];
    temp = ones(length(x(:,1)),1)*(floor((i-1)/4)+1); % the label
    Labelx = [Labelx;temp];
    temptemp = ones(length(x(:,1)),1)*i;
    LabelxWithSeverity = [LabelxWithSeverity;temptemp];
end

for i = [33,35,37,39,41] % 'normal' dataset
    Name = [FaultType{i},'.mat'];
    load(Name);
    Inputx = [Inputx;x];
    temp = ones(length(x(:,1)),1)*0; % the label
    Labelx = [Labelx;temp];
    LabelxWithSeverity = [LabelxWithSeverity;temp];
end

% 44th,54th column is constant value
Inputx(:,[44,45,54,57]) = [];
clear x_steady temp temptemp Name i FaultType

%% Randomize
rng('default')
rng(8)
N = length(Labelx);
%RandomIndex = randperm(N);
%Inputx = Inputx(RandomIndex,:);
%Labelx = Labelx(RandomIndex);
%LabelxWithSeverity = LabelxWithSeverity(RandomIndex);
%clear RandomIndex

%% Testing Set
N2 = floor(N*(1-TestPercentage));
TestSet = Inputx(N2+1:end,:);
TestLabel = Labelx(N2+1:end,:);
TestSeverity = LabelxWithSeverity(N2+1:end,:);

%% Unsupervised Set
N1 = floor(N*0.1); 
N_unsup = floor(UnlabeledPercentage*N);
UnsupSet = Inputx(N1+1:N1+N_unsup,:);

%% Supervised Set
N_sup = floor(LabeledPercentage*N);
SupSet = Inputx(1:N_sup,:);
SupLabel = Labelx(1:N_sup,:);
SupSeverity = LabelxWithSeverity(1:N_sup,:);

%% Normalization
ThisSet = [SupSet;UnsupSet];
MaxValue = max(ThisSet);
MinValue = min(ThisSet);
MaxMatrix = repmat(MaxValue,length(ThisSet(:,1)),1);
MinMatrix = repmat(MinValue,length(ThisSet(:,1)),1);
UnlabeledSetNor = 2*(ThisSet - MinMatrix)./(MaxMatrix - MinMatrix) - 1;

MaxMatrix2 = repmat(MaxValue,length(TestSet(:,1)),1);
MinMatrix2 = repmat(MinValue,length(TestSet(:,1)),1);
TestSetNor = 2*(TestSet - MinMatrix2)./(MaxMatrix2 - MinMatrix2) - 1;

MaxMatrix3 = repmat(MaxValue,length(SupSet(:,1)),1);
MinMatrix3 = repmat(MinValue,length(SupSet(:,1)),1);
LabeledSetNor = 2*(SupSet - MinMatrix3)./(MaxMatrix3 - MinMatrix3) - 1;

clear MinMatrix MinMatrix2 MinMatrix3 MaxMatrix MaxMatrix2 MaxMatrix3 TestSet SupSet ThisSet UnsupSet

save('ProcessData_100t.mat','TestLabel','UnlabeledSetNor','TestSetNor','TestSeverity',...
        'SupLabel','LabeledSetNor','SupSeverity')