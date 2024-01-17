% Set the file path of your CSV file
filename = 'C:\Users\User\Desktop\Code\TV\Data\2024-01-09\Data.csv';

% Read the CSV file into a table
dataTable = readtable(filename);

%%

y =  dataTable.Car_FyFL + dataTable.Car_FyFR + dataTable.Car_FyRL + dataTable.Car_FyRR;

dataTable = removevars(dataTable, 'Car_FyFL');
dataTable = removevars(dataTable, 'Car_FyFR');
dataTable = removevars(dataTable, 'Car_FyRL');
dataTable = removevars(dataTable, 'Car_FyRR');

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
for i = 1:numColumns
    x= dataTable{:,i};
    subplot(6,4,i)
    scatter(x,y)
    title(column_Names(i))
end

    