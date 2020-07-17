#!/usr/bin/env python3
import socket
import pickle

from ev3dev.ev3 import *
from ev3dev2.motor import SpeedPercent, MoveTank
from time import sleep

#socket created
print("Waiting to be connected......")
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)            
s.bind((socket.gethostname(),3000))
s.listen(1)
conn,addr = s.accept()
true=True
addr = str(addr)
print('Connecting by : %s ' %addr )

cl = ColorSensor('in3')
us = UltrasonicSensor('in4')

us.mode='US-DIST-CM'

blender = LargeMotor('outA')
belt = LargeMotor('outC')
door = MediumMotor('outD')

while True:
    data = pickle.loads(conn.recv(1024))
    if data == 1:
        distance = us.value()/10  # convert mm to cm
        if distance < 100:
            Sound.beep()
            cl.mode='COL-COLOR'
            Sound.beep()
            door.run_timed(time_sp=3000, speed_sp=-300)
            sleep(1)
            blender.run_timed(time_sp=9000, speed_sp=180)
            belt.run_timed(time_sp=12000, speed_sp=360)
            sleep(10)
            door.run_timed(time_sp=3000, speed_sp=300)
            cl.mode='COL-REFLECT'
            break
        else:
            cl.mode='COL-REFLECT'

    