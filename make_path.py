import numpy as np
import random
import math as m
import sys

def make_path(start_vel=10,end_time=300):
    start_time=0
    time_step = 0.1

    longitutal_list = ['brake','acc','fix','sin']
    lateral_list = ['fix','sin']

    long_control=[start_vel]
    lat_control = [0]
    time = []
    now_time = start_time
    next_time = 0
    if start_vel < 10:
        print("Start velocity is lower than 10. Shutting down the program.")
        sys.exit()
    while next_time < end_time:
        
        long = random.choice(longitutal_list)
        lat = random.choice(lateral_list)
        last_vel = lat_control[-1]
        last_steer = long_control[-1]
        period = random.randint(1,15)
        now_time = next_time
        next_time = now_time + period
        if next_time > end_time:
            next_time = end_time
            period = end_time - now_time

        timelist = list(range(int(10*now_time),int(10*next_time),int(10*time_step)))
        timelist = list(map(lambda x: round(x*0.1,2),timelist))
        time = time + timelist

        if long == 'brake':
                sens = 10/period
                speed_list = [start_vel+ 10 - sens * time_step * i for i in range(0,len(timelist))]

        elif long == 'acc':
                sens = 10/period
                speed_list = [start_vel + sens * time_step * i for i in range(0,len(timelist))]
        elif long =='fix':
            speed_list = [last_vel for i in range(0,len(timelist))]

        elif long == 'sin':
            amplitude = random.uniform(0,5)
            speed_list = [start_vel + 5 + amplitude * m.sin(i/len(timelist)*2*m.pi) for i in range(0,len(timelist))]
        long_control = long_control + speed_list

        if lat == 'fix':
            steer = random.randint(-30,30)
            steer_rad = steer * m.pi/180
            steer_list = [steer_rad for i in range(0,len(timelist))]
        elif lat == 'sin':
            amplitude = random.randint(-30,30)
            amplitude_rad = amplitude * m.pi/180
            steer_list =[amplitude_rad * m.sin(i/len(timelist)*2*m.pi) for i in range(0,len(timelist))]

        lat_control = lat_control + steer_list
    time.append(end_time)



    value_string ='#TIme steer velocity\n'

    for i in range(len(time)):
        value_string = value_string + str(time[i]) + ' ' +str(lat_control[i]) + ' ' + str(long_control[i])+'\n'



    with open('D:/TV/FCM_Projects_JM/FS_race/Siminput/values_'+str(start_vel)+'.csv','w') as file:
        file.write(value_string)

if __name__ == "__main__":
    ## parser setting
    start_vel = range(10,60,5)
    for i in start_vel:
        make_path(start_vel=i)
