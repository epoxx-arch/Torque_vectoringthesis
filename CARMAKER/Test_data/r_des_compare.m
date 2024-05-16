clc, clear all
% LSTM_TV = load('LSTM_Sinuns_30km_Test.mat');
% No_TV = load('No_LSTM_sinus_30km_Test.mat');
LSTM_TV = load('LSTM_track1.mat');
No_TV = load('No_LSTM_track1.mat');
%%
time_lstm = LSTM_TV.data{1}.Values.Time;
time_No = No_TV.data{2}.Values.Time;
r_des = LSTM_TV.data{3}.Values.Data;
r_lstm = LSTM_TV.data{4}.Values.Data;
r_No_TV = No_TV.data{4}.Values.Data;

plot(time_lstm, r_des, 'LineWidth', 1.5); % Plot desired signal
hold on
plot(time_lstm, r_lstm, 'LineWidth', 1.5); % Plot LSTM signal
plot(time_No, r_No_TV, 'LineWidth', 1.5); % Plot No TV signal
xlim([0, 100])


% Add legend
legend('Desired Signal', 'LSTM', 'No LSTM',"FontSize",20);

% Add labels and title
xlabel('Time',"FontSize",20);
ylabel('Yaw rate',"FontSize",20);
title('Comparison of Signals',"FontSize",20);

% Add grid
grid on;
%%

time_LSTM = LSTM_TV.data{7}.Values.Time;
predtiction = LSTM_TV.data{5}.Values.Data;
Actual = LSTM_TV.data{6}.Values.Data;

figure
plot(time_LSTM, predtiction, 'LineWidth', 1.5); % Plot desired signal
hold on
plot(time_LSTM, Actual, 'LineWidth', 1.5); % Plot LSTM signal

legend('LSTM', 'GT',"FontSize",20);

% Add labels and title
xlabel('Time',"FontSize",20);
ylabel('Moment',"FontSize",20);
title('LSTM and Ground Truth',"FontSize",20);

xlim([0, 100])
% Add grid
grid on;

%%
time_LSTM = LSTM_TV.data{7}.Values.Time;
dis_no = No_TV.data{8}.Values.Data;
dis_lstm = LSTM_TV.data{8}.Values.Data;

figure
plot(time_LSTM,dis_lstm)

hold on
plot(time_No,dis_no)
xlim([0, 100])