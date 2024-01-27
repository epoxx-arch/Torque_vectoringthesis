% Set the file path of your CSV file
filename = 'C:/Users/User/Desktop/Code/TV/Data/ML/Custom_data/concat_random_torque_vectoring_v.csv';

% Read the CSV file into a table
dataTable = readtable(filename,VariableNamingRule="preserve");

%%

My = dataTable.My;
numColumns = width(dataTable);
column_Names = dataTable.Properties.VariableNames;



%%
subplot 
for i = 1:numColumns-2
    for j = i:numColumns-1
        x = dataTable{:,i};
        y = dataTable{:,j};
        figure (i)
        scatter3(x,y,My)
        xlabel(column_Names(i))
        ylabel(column_Names(j))
        title(column_Names(i),column_Names(j))
    end
end

    