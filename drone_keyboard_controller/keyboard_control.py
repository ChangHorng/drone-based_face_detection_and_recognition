from djitellopy import tello

import keyboard as kp


def get_keyboard_input():
    # Initialise the displacement of the drone in all directions to 0
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    # Updates the displacement of drone in the direction respective to the key press
    if kp.get_key("LEFT"):
        lr = -speed
    elif kp.get_key("RIGHT"):
        lr = speed
    if kp.get_key("UP"):
        fb = speed
    elif kp.get_key("DOWN"):
        fb = -speed
    if kp.get_key("w"):
        ud = speed
    elif kp.get_key("s"):
        ud = -speed
    if kp.get_key("a"):
        yv = -speed
    elif kp.get_key("d"):
        yv = speed
    if kp.get_key("e"):
        me.takeoff()
    if kp.get_key("q"):
        me.land()

    return [lr, fb, ud, yv]


kp.init()
me = tello.Tello()
me.connect()

# Keep looping to get keyboard input from user
while True:
    vals = get_keyboard_input()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])