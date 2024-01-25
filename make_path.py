import numpy as np
import random
import math as m

def make_path(start_vel=0,end_time=300):
    start_time=0
    time_step = 0.1

    longitutal_list = ['brake','acc','fix','sin']
    lateral_list = ['fix','sin']

    long_control=[start_vel]
    lat_control = [0]
    time = []
    now_time = start_time
    next_time = 0
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
            if last_vel <= 0:
                speed_list = [5 for i in range(0,len(timelist))]
            else:
                max_sens = last_vel/period
                sens = random.uniform(0,float(max_sens))
                speed_list = [last_vel - sens * time_step * i for i in range(0,len(timelist))]

        elif long == 'acc':
                if last_vel > 100:
                    speed_list = [last_vel for i in range(0,len(timelist))]
                else:
                    sens = random.uniform(0,10)
                    speed_list = [last_vel + sens * time_step * i for i in range(0,len(timelist))]
        elif long =='fix':
            speed_list = [last_vel for i in range(0,len(timelist))]

        elif long == 'sin':
            if last_vel < 5:
                speed_list = [last_vel for i in range(0,len(timelist))]
            else:  
                amplitude = random.uniform(0,5)
                speed_list = [last_vel + amplitude * m.sin(i/len(timelist)*2*m.pi) for i in range(0,len(timelist))]
        long_control = long_control + speed_list

        if lat == 'fix':
            steer = random.randint(-30,30)
            steer_rad = steer * m.pi/180
            steer_list = [steer_rad for i in range(0,len(timelist))]
        elif lat == 'sin':
            amplitude = random.randint(-45,45)
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
    start_vel = range(0,40,5)
    for i in start_vel:
        make_path(start_vel=i)
