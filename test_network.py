from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
import random
from random import sample, randint
from time import time, sleep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import trange
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import cv2
import shutil
import math
from model import DoomNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.benchmark=True
random.seed(0)
torch.manual_seed(0)

# Other parameters
frame_repeat = 5
resolution = (42, 42)
episodes_to_watch = 10
model_loadfile = "./save/model.pth"
# Configuration file path
#config_file_path = "../ViZDoom/scenarios/my_way_home.cfg"
config_file_path = "./scenarios/my_way_home_sparse.cfg"
#config_file_path = "../ViZDoom/scenarios/basic.cfg"
#config_file_path = "../ViZDoom/scenarios/defend_the_center.cfg"

# Converts and down-samples the input image
def preprocess(state):
    img = state.screen_buffer
    img = np.moveaxis(img, [0,1,2], [2,0,1])
    img = cv2.resize(img, resolution)
    #img = Image.fromarray(img)
    #img = Resize(tupleself.resolution) (img)
    img = ToTensor() (img)
    img = img.unsqueeze(0).cuda()
    return img

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

if __name__ == '__main__':

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]

    print("Loading model from: ", model_loadfile)
    model = DoomNet(len(actions))
    my_sd=torch.load(model_loadfile)
    model.load_state_dict(my_sd)
    model=model.cuda()
    model=model.eval()

    count_fr=0
    all_scores=np.zeros((episodes_to_watch),dtype=np.float32)

    for j in range(episodes_to_watch):
        game.new_episode()
        count_fr=0
        state=model.init_hidden(1)
        while not game.is_episode_finished():
            s1 = preprocess(game.get_state())
            (actual_q, _, state) = model(s1, state)
            m, index = torch.max(actual_q, 1)
            a = index.item()

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[a])
            for _ in range(frame_repeat):
                #if not game.is_episode_finished():
                #    s1, frame = preprocess(game.get_state().screen_buffer)
                    #out.write(frame)
                game.advance_action()
                sleep(0.02)

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        all_scores[j]=score
        print("Total score: ", score)

    final_score=all_scores.mean()
    print('Final scores is ', final_score)

