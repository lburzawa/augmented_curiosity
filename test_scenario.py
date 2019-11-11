from vizdoom import *
import numpy as np
import random
import time
import itertools as it
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Scale
import cv2

game = DoomGame()
game.load_config("../ViZDoom/scenarios/health_gathering_supreme.cfg")
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)
game.set_screen_format(ScreenFormat.CRCGCB)
game.set_screen_resolution(ScreenResolution.RES_640X480)
#game.clear_available_game_variables()
#game.add_available_game_variable(GameVariable.POSITION_X)
#game.add_available_game_variable(GameVariable.POSITION_Y)
game.init()

#shoot = [0, 0, 1]
#left = [1, 0, 0]
#right = [0, 1, 0]
#actions = [shoot, left, right]
n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]
#some_action = [0]*n; some_action[0]=1
#actions=[list(b) for b in list(set([a for a in it.permutations(some_action)]))]

episodes = 10
for i in range(0,episodes):
    game.new_episode()
    t = 1
    state = game.get_state()
    #misc = state.game_variables
    #curr_x=misc[0]
    #curr_y=misc[1]
    while not game.is_episode_finished():
        #state = game.get_state().screen_buffer
        #state=np.moveaxis(state, [0,1,2], [2,0,1])
        #print(state.shape)
        #img=Image.fromarray(state)
        #img.save('./garbage/image.png')
        #misc = state.game_variables
        #dist=np.sqrt(np.power(misc[0]-curr_x,2)+np.power(misc[1]-curr_y,2))
        #if t%20==0:
        #    print(dist)
        #curr_x = misc[0]
        #curr_y = misc[1]
        misc = game.get_game_variable(GameVariable.HEALTH)
        #print(misc)
        game.advance_action(5)
        #reward = game.make_action(random.choice(actions))
        #print("\treward:", reward)
        #time.sleep(0.02)
        t+=1

    #print(t)
    print("Result:", game.get_total_reward())
    time.sleep(2)

