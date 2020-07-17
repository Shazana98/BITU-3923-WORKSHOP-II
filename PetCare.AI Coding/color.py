#!/usr/bin/env python3
from ev3dev.ev3 import *

cl = ColorSensor('in3')
us = UltrasonicSensor('in4')

us.mode='US-DIST-CM' # Put the US sensor into distance mode.

while True:
    distance = us.value()/10  # convert mm to cm
    if distance < 50:
        Sound.beep()
        cl.mode='COL-COLOR'
    else:
        cl.mode='COL-REFLECT'