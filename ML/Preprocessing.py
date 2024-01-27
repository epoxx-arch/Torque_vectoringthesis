import numpy as np
import pandas as pd
import math as m
import os

def preprocessing(input_path,output_path):
    df = pd.read_csv(input_path, delimiter='\t',low_memory=False)
    # Drop first 500 rows and the '#Name' column
    df = df.drop(index=range(0, 500))
    df = df.drop('#Name', axis=1)

    # Convert all values to numeric
    df = df.apply(pd.to_numeric, errors='coerce').astype(float)

    # Extract specific columns and convert them to lists
    FyFL, FyFR, FyRL, FyRR = [df.pop(f'Car.Fy{i}') for i in ['FL', 'FR', 'RL', 'RR']]
    steer_angleFL, steer_angleFR = [df.pop(f'Car.SteerAngle{i}') for i in ['FL', 'FR']]
    Trq_alignFL, Trq_alignFR, Trq_alignRL, Trq_alignRR = [df.pop(f'Car.TrqAlign{i}') for i in ['FL', 'FR', 'RL', 'RR']]

    # Calculate 'steer' as the sum of 'steer_angleFL' and 'steer_angleFR'
    steer = steer_angleFL + steer_angleFR

    # Calculate 'My' using vectorized operations
    tr, lf, lr = 0.621, 0.813, 0.787
    My = (FyFL - FyFR) * np.sin(steer) * tr + (FyFR + FyFL) * np.cos(steer) * lf - (FyRR + FyRL) * lr - Trq_alignFL - Trq_alignFR - Trq_alignRL - Trq_alignRR

    # Control Values
    DM_steering = df.pop('DM.Steer.Ang')
    motor_speed_FL, motor_trq_FL = df.pop('PT.Motor.rotv'), df.pop('PT.Motor.Trq')
    motor_speed_FR, motor_trq_FR = df.pop('PT.Motor1.rotv'), df.pop('PT.Motor1.Trq')
    motor_speed_RL, motor_trq_RL = df.pop('PT.Motor2.rotv'), df.pop('PT.Motor2.Trq')
    motor_speed_RR, motor_trq_RR = df.pop('PT.Motor3.rotv'), df.pop('PT.Motor3.Trq')

    # Sensor Values
    ax, ay, az = df.pop('Car.ax'), df.pop('Car.ay'), df.pop('Car.az')
    Roll_Vel, Yaw_Vel, Pitch_Vel = df.pop('Car.RollVel'), df.pop('Car.YawVel'), df.pop('Car.PitchVel')

    # addition Values

    Vel = df.pop('Car.Con.v')
    # Create a new DataFrame for the extracted data
    input_data = pd.DataFrame({
        "DM.steering": DM_steering,
        "motor_speed_FL": motor_speed_FL,
        "motor_speed_FR": motor_speed_FR,
        "motor_speed_RL": motor_speed_RL,
        "motor_speed_RR": motor_speed_RR,
        "motor_trq_FL": motor_trq_FL,
        "motor_trq_FR": motor_trq_FR,
        "motor_trq_RL": motor_trq_RL,
        "motor_trq_RR": motor_trq_RR,
        "ax": ax,
        "ay": ay,
        "az": az,
        "Roll_Vel": Roll_Vel,
        "Yaw_Vel": Yaw_Vel,
        "Pitch_Vel": Pitch_Vel,
        "Car.Con.v" : Vel,
        "My" : My
    })

    # Export the new DataFrame to a CSV file
    input_data.to_csv(output_path, index=False)



if __name__ == "__main__":

    path = 'D:/TV/FCM_Projects_JM/FS_race/SimOutput/DESKTOP-00IBLK8/20240126/random_torque_vectoring'
    output_path = 'Data/2024_01_26/random_torque_vectoring'
    file_list = os.listdir(path)
    for file in file_list:
        file_name = file.split('.')
        print('Now processing :' + file_name[0])
        input_path = path + '/' + file
        output = output_path + '/' + file_name[0] + '.csv'
        preprocessing(input_path, output)
    print('End')