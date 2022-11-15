import os
from os.path import dirname, abspath

import numpy as np
from  tqdm import  tqdm
from wireless_blocks import dist_calc_x, dist_calc_y

np.random.seed(1)

# from env import IrsNoma
dir_path = dirname(abspath(__file__))



# def user_move():
#     for i in range(self.gfu):
#
#         speed_lst=np.random.uniform(0,self.max_speed)
#         angle_lst=np.random.uniform(0,2*np.pi)
#         self.gfu_coord[i,0] = dist_calc_x(0,self.gfu_coord[i,0],speed_lst,angle_lst,[],self.gfu_min_x,self.gfu_max_x,self.max_speed)
#         self.gfu_coord[i,1] = dist_calc_y(0,self.gfu_coord[i,1],speed_lst,angle_lst,[],self.y_min_edge,self.y_max_edge,self.max_speed)
#     for j in range(self.gbu):
#         speed_lst=np.random.uniform(0,self.max_speed)
#         angle_lst=np.random.uniform(0,2*np.pi)
#         self.gbu_coord[j,0] = dist_calc_x(0,self.gbu_coord[j,0],speed_lst,angle_lst,[],self.gbu_min_x,self.x_max_edge,self.max_speed)
#         self.gbu_coord[j,1] = dist_calc_y(0,self.gbu_coord[j,1] ,speed_lst,angle_lst,[],self.y_min_edge,self.y_max_edge,self.max_speed)
#

if __name__ == "__main__":
    max_step = 1000
    max_gfu = 1024
    coord_cue = np.zeros((max_step,max_gfu, 3))
    cue_generate_limit = 20
    max_speed = 2
    x_min_edge = -35
    y_min_edge = -10
    x_max_edge = -25
    y_max_edge = 200
    gfu_min_x = 0
    gfu_max_x = 200
    user_angle = "random"

    ''' 先生成近处的 '''
    for i in range(max_gfu):
            coord_cue[0,i, 0] = 8 + (2*np.random.rand()-1)*5
            coord_cue[0, i,1] = 11+(2*np.random.rand()-1)*5
            coord_cue[0, i,-1] = 0
    loader =  tqdm(range(1,max_step))
    for k in loader:
            loader.set_description_str( ("[step %d| %d]"%(k,max_step)))

            for i in range(max_gfu):

                speed_lst = np.random.uniform(0, max_speed)
                angle_lst = np.random.uniform(0, 2 * np.pi)
                coord_cue[k, i, 0] = dist_calc_x(0, coord_cue[k - 1, i, 0], speed_lst, angle_lst, [], gfu_min_x,
                                                 gfu_max_x, max_speed)
                coord_cue[k, i, 1] = dist_calc_y(0, coord_cue[k - 1, i, 1], speed_lst, angle_lst, [], y_min_edge,
                                                 y_max_edge, max_speed)
                coord_cue[k, i, -1] = 0

    print(dir_path)
    np.save(dir_path + '/cue.npy', coord_cue)


