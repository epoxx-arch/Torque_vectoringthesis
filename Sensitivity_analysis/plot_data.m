% Set the file path of your CSV file
filename = 'C:/Users/User/Desktop/Code/TV/Data/ML/Custom_data/new_concat2.csv';

% Read the CSV file into a table
dataTable = readtable(filename,VariableNamingRule="preserve");

%%

My = dataTable.My;
numColumns = width(dataTable);
column_Names = dataTable.Properties.VariableNames;



%%
for i = 1:numColumns-1
    x = dataTable{:,i};
    figure(i);
    hold on; % Add this line to overlay plots on the same figure
    yyaxis left;
    plot(x);
    yyaxis right;
    plot(My);
    legend(column_Names(i)); % Update legend to label the lines correctly
    xlabel(column_Names(i));
    title(column_Names(i));
    hold off; % Add this line to release the hold on the current figure
end

%%


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

    