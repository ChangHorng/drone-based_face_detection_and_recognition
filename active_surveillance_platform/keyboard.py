import pygame


def init():

    pygame.init()

    # Create a pygame window to get key input from user for drone movement
    window = pygame.display.set_mode((400, 400))
    pygame.display.set_caption('Drone Keyboard Controller Window')

    return window


def get_key(key_name):

    ans = False

    for eve in pygame.event.get():
        pass

    # Get the key that is pressed by user
    key_input = pygame.key.get_pressed()
    my_key = getattr(pygame, 'K_{}'.format(key_name))

    if key_input[my_key]:
        ans = True

    pygame.display.update()

    return ans
