% Set the file path of your CSV file
filename = 'C:/Users/User/Desktop/Code/TV/Data/ML/Custom_data/new_concat.csv';

% Read the CSV file into a table
dataTable = readtable(filename,VariableNamingRule="preserve");

%%

My = dataTable.My;
numColumns = width(dataTable);
column_Names = dataTable.Properties.VariableNames;



%%
for i = 1:numColumns-1

    x = dataTable{:,i};
    figure (i)
    yyaxis left;
    plot(x)
    yyaxis right ;
    plot(My)
    xlabel(column_Names(i))
    title(column_Names(i))

end

%%


for i = 1:numColumns-2
    for j = i:numColumns-1
        x = dataTable{:,i};
        y = dataTable{:,j};
        figure (i)
        scatter3(x,y,My)
        xlabel(column_Names(i))s
        ylabel(column_Names(j))
        title(column_Names(i),column_Names(j))
    end
end

    