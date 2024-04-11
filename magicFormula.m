% MATLAB Code to Plot the Magic Formula for Lateral Force vs. Slip Angle

% Define the slip angle range (in degrees)
slip_angle = -15:0.1:15; % A range of slip angles from -15 to 15 degrees

% Magic Formula coefficients for an example tire
B = 10; % Stiffness factor
C = 1.9; % Shape factor
D = 1000; % Peak factor
E = 0.97; % Curvature factor

% Convert slip angle from degrees to radians for the calculation
slip_angle_rad = deg2rad(slip_angle);

% Calculate the lateral force using the Magic Formula
lateral_force = D * sin(C * atan(B * slip_angle_rad - E * (B * slip_angle_rad - atan(B * slip_angle_rad))));

% Plotting
figure;
plot(slip_angle, lateral_force);
grid on;
title('Lateral Force vs. Slip Angle - Magic Formula');
xlabel('Slip Angle (degrees)');
ylabel('Lateral Force (N)');