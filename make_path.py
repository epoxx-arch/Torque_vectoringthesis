import numpy as np
import random
import math as m
import sys


def generate_period_list(end_time,dt):
    '''output is period list. this contains period sampling time nums. if 15sec, 150'''
    time_sample_num = int(end_time/dt) + 1
    period_list =[]
    while True :
        period = random.randint(10,150)
        time_sample_num = time_sample_num-period
        if time_sample_num < 0 :
            period_list.append(time_sample_num+period)
            break
        else:
            period_list.append(period)

    return period_list



def generate_longitudinal_control(start_vel, vel_interval,end_time, time_step,longitutal_list):
    period_list = generate_period_list(end_time,time_step)
    control = random.choice(longitutal_list)
    speed_list = []
    for period in period_list:
        if control == 'brake':
            scale = random.uniform(1,vel_interval)
            sens = scale/period
            speed_list = speed_list + [start_vel+ scale - sens * time_step * i for i in range(0,period+1)]


        elif control == 'acc':
            scale = random.uniform(1,vel_interval)
            sens = scale/period
            speed_list = speed_list + [start_vel + sens * time_step * i for i in range(0,period)]

        elif control == 'sin':
            amplitude = random.uniform(0,vel_interval/2)
            speed_list = speed_list + [start_vel + vel_interval/2 + amplitude * m.sin(i/period*2*m.pi) for i in range(0,period)] 
        
        elif control =='fix':
            speed_list = speed_list + [random.uniform(start_vel,start_vel+vel_interval) for i in range(0,period)]

    return speed_list          

    

def generate_lateral_control(max_steer, end_time, time_step, lateral_list):
    period_list = generate_period_list(end_time,time_step)
    control = random.choice(lateral_list)
    steer_list = []
    for period in period_list:
        if control == 'fix':
            steer = random.randint(-max_steer,max_steer)
            steer_rad = steer * m.pi/180
            steer_list = steer_list + [steer_rad for i in range(0,period)]
        elif control == 'sin':
            amplitude = random.randint(-max_steer,max_steer)
            amplitude_rad = max_steer * m.pi/180
            steer_list = steer_list + [amplitude_rad * m.sin(i/period*2*m.pi) for i in range(0,period)]

    return steer_list


def make_path(start_vel,vel_interval, end_time=200, longitutal_list=None, lateral_list=None):
    start_time = 0
    time_step = 0.1
    max_steer = 30
    time_sample_num = end_time/time_step + 1
    if longitutal_list is None:
        longitutal_list = ['brake', 'acc', 'fix', 'sin']
    if lateral_list is None:
        lateral_list = ['fix', 'sin']

    # Rest of your code...

    # Modify the function to use the external functions
    long_control = generate_longitudinal_control(start_vel, vel_interval,end_time, time_step, longitutal_list)
    lat_control = generate_lateral_control(max_steer, end_time, time_step, lateral_list)
    time = list(range(int(0),int(10*end_time),int(10*time_step)))
    time = list(map(lambda x: round(x*0.1,2),time))
    time.append(end_time)
    # Write to file
    value_string ='#TIme steer velocity\n'
    for i in range(len(time)):
        value_string = value_string + str(time[i]) + ' ' +str(lat_control[i]) + ' ' + str(long_control[i])+'\n'

    with open('D:/TV/FCM_Projects_JM/FS_race/Siminput/values_'+str(start_vel)+'.csv','w') as file:
        file.write(value_string)

if __name__ == "__main__":
    # Use default values or specify your own
    vel_interval = 6
    cnt = 1 
    max_vel = 0



    while True :
        
        temp_max_vel = vel_interval * cnt
        if temp_max_vel > 30 :
            break
        else:
            max_vel = temp_max_vel
        
        cnt += 1

    
    start_vel_range = range(vel_interval, max_vel+1, vel_interval)
    
    for start_vel in start_vel_range:
        make_path(start_vel, vel_interval)