% Set the file path of your CSV file
filename = 'C:/Users/User/Desktop/Code/TV/Data/2024_01_24/TV_JM_154213.dat.csv';

% Read the CSV file into a table
dataTable = readtable(filename);

%%

My = dataTable.My;
numColumns = width(dataTable);
column_Names = dataTable.Properties.VariableNames;
%%
x = dataTable{:,15};
temp = dataTable{:,4};
scatter(x,temp)

xlabel('torque')
ylabel('force')


%%
subplot 
for i = 1:numColumns-1
    x= dataTable{:,i};
    subplot(6,4,i)
    scatter(x,My)
    title(column_Names(i))
end

    