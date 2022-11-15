# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import argparse
import time
from components.base_blocks import trans_complex_to_real
import  numpy as np
from components.wireless_env import IrsCompMISOEnv
from implementation.data_loader import getDataset
from implementation.solver_gan_ofdm import Solver
from components.config import  TrainSettings









if __name__ =="__main__":

    # a= np.array([1,1,1,1])
    # b= np.array([2j,2j,2j,2j])
    # fft_rs = a+b
    # val = trans_complex_to_real(fft_rs)
    # MISO-OFDM
    config = TrainSettings().parse_args()
    # Extract the settings
    train,test,valid = getDataset(config)
    model = Solver(config=config,
                   voc_train_loader=train,
                   voc_test_loader=test,
                   voc_valid_loader=valid).cuda()      # create a model given setting.model and other settings           # regular setup: load and print networks; create schedulers
    #model.record_H()
    model.train()








