import keyboard as kp


def start_control():

    window = kp.init()

    return window


def get_keyboard_input():
    """
    This function detects the key press from users to carry out the respective movements.
    Here are the controls:
    - W -> Drone moves upwards vertically
    - A -> Drone turns to the left
    - S -> Drone moves downwards vertically
    - D -> Drone turns to the right
    - Up (Arrow Key) -> Drone moves forwards horizontally
    - Down (Arrow Key) -> Drone moves backwards horizontally
    - Left (Arrow Key) -> Drone moves to the left
    - Right (Arrow Key) -> Drone moves to the right
    - E -> Drone takes off (flight mode)
    - Q -> Drone lands
    - O -> Drone's camera starts streaming
    - P -> Drone's camera stops streaming
    """
    # Initialise the displacement of the drone in all directions to 0
    lr, fb, ud, yv = 0, 0, 0, 0

    take_off = None
    open_cam = None
    # Initialize drone movement speed to 25 units per key press
    speed = 25

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
        take_off = True
    if kp.get_key("q"):
        take_off = False
    if kp.get_key("o"):
        open_cam = True
    if kp.get_key("p"):
        open_cam = False

    return [lr, fb, ud, yv, take_off, open_cam]
