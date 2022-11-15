from math import exp, cos, sin
from os.path import dirname, abspath
import  math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
from numpy import random
from components.wireless_blocks import *
import torch
dir_path = dirname(abspath(__file__)) 
CUE_all_coord = np.load(dir_path + '/cue.npy')
# coord_cue_near_random = np.load(dir_path + '/cue_near.npy')
# coord_cue_far_random = np.load(dir_path + '/cue_far.npy')
coord_cue_random = np.load(dir_path + '/cue.npy')
# GBU_all_coord = np.load(dir_path + '/gbu.npy')
# coord_gbu_random = np.load(dir_path + '/gbu_100001.npy')

class IrsCompMISOEnv:
    def __init__(self,bs_num, ue_num, mec_p_max,transmit_p_max, irs_units_num,antenna_num,fov_patch_num,reflect_max,
                 r_min,BW,with_CoMP=False,open_matlab=False,train= True,mec_rule = "default",rand_omega=False):
        '''
        :param bs_num: 基站数目
        :param ue_num: 用户设备数目
        :param mec_p_max: MEC最大功率限制
        :param transmit_p_max: 传输最大功率限制
        :param irs_units_num: IRS反射单元数目
        :param antenna_num: 发射天线数
        :param fov_patch_num: Fov数，目前fov_patch_num=ue_num
        :param reflect_max: 最大反射偏转角度
        :param r_min: 最小传输速率
        :param BW: 带宽
        :param with_CoMP: 是否使用多点协作传输
        :param open_matlab: 是否使用matlab
        :param train_mode: 是否训练
        :param mec_rule: 为缓存指配mec的策略
        :param rand_omega: 是否随机预编码矩阵
        '''
        self.mec_p_max =mec_p_max
        self.transmit_p_max = transmit_p_max
        self.train = train
        self.mec_rule = mec_rule
        self.init_bs_power=0
        self.init_bs_rates=0
        self.opt_bs_power= 0
        self.opt_bs_rates=0
        self.rs=0
        self.cr=1.5
        self.Kb=10**(-9)
        self.Ub=10**5
        self.ub=15
        self.epsilon_fix = None
        self.rand_omega = rand_omega
        self.with_CoMP = with_CoMP
        self.mec_storage=generate_storage_mec(bs_num,80,80)
        self.mec_max_computing_resources = generate_max_computing_resources_mec(bs_num,3200,3200)
        #这里改一下，直接定死每个MEC能缓存的内容
        self.fov_sizes = generate_fov_size(fov_patch_num,1,1)
        self.available_space = available_space(bs_num,fov_patch_num)
        self.epsilon = np.zeros([bs_num,fov_patch_num],np.int)
        self.epsilon_noCoMP = np.zeros([bs_num,fov_patch_num],np.int)
        self.bs_num = int(bs_num)
        self.ue_num = int(ue_num)
        self.fov_patch_num = int(fov_patch_num)
        self.action_table= []
        self.irs_units_num = int(irs_units_num)
        self.antenna_num = int(antenna_num)
        self.storage_limit = gen_mec_store_limit(5e-3, self.bs_num, self.mec_max_computing_resources,self.mec_p_max)


        self.action_table = gen_action_table_v2(bs_num,fov_patch_num)
        self.uefov_table = generate_uefov_table(self.ue_num)
        self.bsfov_table = generate_bsfov_table(self.epsilon)
        self.omegas =generate_omega_fixed(self.bs_num, self.ue_num, self.antenna_num) #初始化一个omega
        self.total_power_record = []
        self.total_power_record_NoRIS = []
        self.init_bs_power_record = []
        self.total_init_power_record=[]
        self.bs_power_record_NoRIS=[]
        self.bs_power_record=[]
        self.bs_power_randOmega_record=[]
        self.available_action = []

        self.ue_avg_rates_record = []
        self.ue_avg_rates_record_NoRIS= []

        for bs in range (self.bs_num):
            self.available_action.append(np.ones(len(self.action_table)))
        self.available_action = np.array(self.available_action)
        self.reflect_max = reflect_max
        self.r_min = r_min
        self.rendered_fov_sizes = cal_total_rendered_fov_sizes(self.fov_sizes,self.cr)
        self.total_computing_resources = cal_total_computing_resources(self.fov_sizes,self.Kb,self.Ub,self.ub,self.cr)
        self.BW = BW
        N_0_dbm = -174 + 10 * np.log10(1e7)
        self.N_0= np.power(10,((N_0_dbm - 30) / 10))

        self.cue_coord = coord_cue_random
        self.ch_space = np.zeros(self.ue_num)
        self.bs_coord = None
        self.gfu_max = 1
        self.engine = 0
        self.action = np.zeros(self.bs_num)
        # self.n_reflect = 5 #将反射矩阵的系数划分成几等级
        self.action_irs = 0
        self.reflect = np.ones((self.irs_units_num, self.irs_units_num))
        # self.action_c_p = np.zeros((self.ue_num+self.antenna_num, self.antenna_num))
        self._coord_set()
        # self._gain_calculate()

        self.G,self.G2,self.g_ue_ris,self.g_bs_ris,self.g_bs_ue \
            = all_G_gain_cal_MISO_splitI(0, self.bs_num, self.ue_num,
                                         self.antenna_num, self.irs_coord,
                                         self.cue_coord, self.bs_coord,
                                         self.reflect, self.irs_units_num)
        self.states = np.concatenate([np.array(self.G).flatten(), self.epsilon.flatten()], axis=0)

        # self.states=self._gain_contact()+self.ch_add_states()+self.p_add_states()+self.reflect_amp_add_states()
        # self.states = self._gain_contact()
        print("MISO环境创建完毕！")

    def _coord_set(self):
        '''
        :return: 根据预先的坐标按照不同数量进行选择
        '''
        # self.bs_coord = np.matrix([[0, 0, 0]]).getA()
        # self.cue_coord = CUE_all_coord[:self.ue_num+self.antenna_num, :]
        self.bs_coord = np.array([[0,0,0],[5,20,0],[20,10,0],[15,15,0]])
        # self.cue_coord = coord_cue_random[:self.ue_num, :]
        # a=GBU_all_coord[:self.antenna_num, :]
        # self.cue_coord = np.r_[self.cue_coord,a]
        # self.irs_coord = np.matrix([[31, 6, 0]])
        self.irs_coord = np.array([[6, 12, 0]])

        # #将位置plot出来
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
        #
        # plt.xlabel('X')
        # plt.ylabel('Y')
        #
        # print('基站用户位置分布图')
        #
        # colors1 = '#00CED1'  # 点的颜色
        # colors2 = '#DC143C'
        # colors3 = '#7FFFD4'
        # colors4 = '#A52A2A'
        # colors5 = '#008000'
        # area = np.pi ** 2  # 点面积
        # # 画散点图
        # plt.scatter(self.bs_coord[:, 0], self.bs_coord[:, 1], s=area * 2, marker='o', c=colors1, alpha=0.4, label='基站')
        # plt.scatter(self.irs_coord[:, 0], self.irs_coord[:, 1], s=area * 2, marker='s', c=colors2, alpha=0.4,
        #             label='反射面')
        # plt.scatter(self.cue_coord[0,:, 0], self.cue_coord[0,:, 1], s=area*2, marker='v', c=colors3, alpha=0.4, label='CUE用户')
        # #             label='D2DR')
        # plt.legend()
        # plt.show()
        #

        # if self.point>0:
        #     new_coord_lst = []
        #     random_count = self.point*15+1
        #     for g0 in range(self.ue_num):
        #         new_coord_lst.append(coord_cue_random[random_count,g0,:])
        #     for g1 in range(self.antenna_num):
        #         new_coord_lst.append(coord_gbu_random[random_count,g1+self.ue_num,:])
        #     self.cue_coord = np.array(new_coord_lst).reshape(self.ue_num+self.antenna_num,3)
        # else:
        # self._gain_calculate()


    def plot_location(self):
        #将位置plot出来
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
        plt.xlabel('X')
        plt.ylabel('Y')
        print('位置分布图')
        colors1 = '#00CED1'  # 点的颜色
        colors2 = '#DC143C'
        colors3 = '#7FFFD4'
        colors4 = '#A52A2A'
        colors5 = '#008000'
        area = np.pi ** 2  # 点面积
        # 画散点图
        plt.scatter(self.irs_coord[:, 0], self.irs_coord[:, 1], s=area * 2, marker='s', c=colors2, alpha=0.4,
                    label='反射面')
        plt.scatter(self.cue_coord[:, 0], self.cue_coord[:, 1], s=area, marker='v', c=colors3, alpha=0.4, label='CUE用户')
        plt.legend()
        plt.savefig(dir_path + '/location.png', dpi=300)
        plt.show()

    def _gain_contact(self):
        #将计算出来的信道增益进行拼接作为state
        a=[]
        for ue_num in range(self.ue_num):
            for bs_i in range(self.bs_num):
                for ch_i in range(self.antenna_num):
                    which_bs = self.ch_space[ue_num]/self.antenna_num
                    which_ch = self.ch_space[ue_num]%self.antenna_num
                    if bs_i==which_bs and ch_i == which_ch:
                        a.append(self.G[ue_num])
                    else:
                        a.append(0)
        return a
    def reset(self):
        # 重新设置环境
        # if stat=="all":
        #     self._coord_set()
        # self.G, self.G2, self.small_irs_record, self.ad_irs_record, self.small_ue_record, self.ad_ue_record, self.g_ue_ris, self.g_bs_ris, self.g_bs_ue \
        #     = all_G_gain_cal_MISO_splitI(0, self.bs_num, self.ue_num,
        #                                  self.antenna_num, self.irs_coord,
        #                                  self.cue_coord, self.bs_coord,
        #                                  self.reflect, self.irs_units_num)
        # self.states = np.concatenate([np.array(self.G).flatten() ,self.epsilon.flatten()],axis=0)
        # return self.states
        return  self.step(0)
    # def user_location_random(self):
    #     #随机生成下一步的位置
    #     limit = bs_dist_limit-100
    #     limit_1 = bs_dist_limit-50
    #     zeros_arr = np.array([0]).reshape(-1,1)
    #     for i in range(self.ue_num):
    #         cx = (-1 + 2*np.random.random())* limit
    #         cy = (-1 + 2*np.random.random())* limit
    #         cxy = np.array([cx,cy]).reshape(1,2)
    #         while np.linalg.norm(cxy, axis=1, keepdims=True) > limit_1:
    #             cx = (-1 + 2*np.random.random())* limit
    #             cy = (-1 + 2*np.random.random())* limit
    #             cxy = np.array([cx,cy]).reshape(1,2)
    #         self.cue_coord[i,:] = np.hstack((cxy,zeros_arr))
    #     print('cue新位置随机成功')

    def step(self,time_slot):
        new_coord_lst =[]
        self.G,self.G2,self.g_ue_ris,self.g_bs_ris,self.g_bs_ue \
            = all_G_gain_cal_MISO_splitI(time_slot, self.bs_num, self.ue_num,
                                         self.antenna_num, self.irs_coord,
                                         self.cue_coord, self.bs_coord,
                                         self.reflect, self.irs_units_num)
        # states_ = self.states = np.concatenate([np.array(self.G).flatten(), self.epsilon.flatten()], axis=0)
        return self.G2
    def action_states(self):
        p = []
        for i in range(self.bs_num):
            p.append(self.action[i])
        return p
    def reflect_amp_add_states(self):
        reflect_amp= []
        for i in range(self.irs_units_num):
            reflect_amp.append(self.reflect[i][i])
        return reflect_amp
    def G_tau__add_states(self):
        g_tau= []
        for i in range(self.ue_num):
            for j in range(self.antenna_num):
                if self.G[i][j]**2>=self.tau:
                    g_tau.append(1)
                else:
                    g_tau.append(0)
        return g_tau

