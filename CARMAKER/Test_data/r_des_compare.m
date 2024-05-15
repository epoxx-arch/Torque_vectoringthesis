% LSTM_TV = load('LSTM_Sinuns_30km_Test.mat');
% No_TV = load('No_LSTM_sinus_30km_Test.mat');
LSTM_TV = load('Track_Yes.mat');
No_TV = load('Track_No.mat');
%%
time_lstm = LSTM_TV.data{1}.Values.Time;
time_No = No_TV.data{2}.Values.Time;
r_des = LSTM_TV.data{1}.Values.Data;
r_lstm = LSTM_TV.data{2}.Values.Data;
r_No_TV = No_TV.data{2}.Values.Data;

plot(time_lstm, r_des, 'LineWidth', 1.5); % Plot desired signal
hold on
plot(time_lstm, r_lstm, 'LineWidth', 1.5); % Plot LSTM signal
plot(time_No, r_No_TV, 'LineWidth', 1.5); % Plot No TV signal

xlim([0, 40])

% Add legend
legend('Desired Signal', 'LSTM', 'No LSTM',"FontSize",20);

% Add labels and title
xlabel('Time',"FontSize",20);
ylabel('Amplitude',"FontSize",20);
title('Comparison of Signals',"FontSize",20);

% Add grid
grid on;