import numpy as np
import random
import math as m
import sys


def generate_period_list(end_time,dt):
    '''output is period list. this contains period sampling time nums. if 15sec, 150'''
    time_sample_num = int(end_time/dt) + 1
    period_list =[]
    while True :
        period = random.randint(50,200)
        time_sample_num = time_sample_num-period
        if time_sample_num < 0 :
            period_list.append(time_sample_num+period)
            break
        else:
            period_list.append(period)

    return period_list



def generate_longitudinal_control(vel,end_time, time_step):
    period_list = generate_period_list(end_time,time_step)
    speed_list = []
    for period in period_list:
        speed_list = speed_list + [vel for i in range(0,period)]

    return speed_list          



def generate_lateral_control(steer, end_time, time_step):
    period_list = generate_period_list(end_time,time_step)
    steer_list = []
    for period in period_list:
        steer_rad = steer * m.pi/180
        steer_list = steer_list + [steer_rad for i in range(0,period)]


    return steer_list


def make_path(steer, vel, end_time=500 ):
    start_time = 0
    time_step = 0.01
    time_sample_num = end_time/time_step + 1

    # Rest of your code...
    # Modify the function to use the external functions
    long_control = generate_longitudinal_control(vel, end_time, time_step)
    lat_control = generate_lateral_control(steer, end_time, time_step)
    time = list(range(int(0),int(100*end_time),int(100*time_step)))
    time = list(map(lambda x: round(x*0.01,3),time))
    for i in range(200):
        lat_control[i] =0

    time.append(end_time)

 
    # Write to file
    value_string ='#TIme steer velocity\n'
    for i in range(len(time)):
        value_string = value_string + str(time[i]) + ' ' +str(lat_control[i]) + ' ' + str(long_control[i])+'\n'

    with open('D:/TV/FCM_Projects_JM/FS_race/Siminput/circles'+str(vel)+'.csv','w') as file:
        file.write(value_string)

if __name__ == "__main__":
    # Use default values or specify your own

    steer = 100
    vel = 15
    end_time = 200

    
    make_path(steer, vel, end_time)