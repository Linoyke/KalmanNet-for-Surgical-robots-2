# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:45:16 2025

@author: User
"""

########
##Kalmanet
#########

import torch
import torch.nn as nn
import math
from datetime import datetime
import Filters.EKF_test_withbias_IK as EKF_test
#import Filters.EKF_test_withbias_IK as EKF_test
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from Simulations.Extended_sysmdl import SystemModel
import Simulations.config as config
from Simulations.parameters_withbias_trainIK import m1x_0, m2x_0, m, n, f, h, fInacc, Q_structure, R_structure, f_with_action
from KNet.KalmanNet_nn_withbias_IK import InverseKinematicsNN
from Pipelines.Pipeline_EKF_withbias_IK import Pipeline_EKF
from KNet.KalmanNet_nn_withbias_IK import KalmanNetNN
import winsound
# from Plot import Plot_extended as Plot
import torch.nn.functional as F
from Utils import read_data, data_from_dict, stack_tensors, split_into_trajectories, split_trajectories, split_traj_by_given_idx, load_indices_and_split_tensor, \
    add_bias_dimensions, plot_trajectories,plot_predictions_vs_ground_truth, load_from_dict, calc_action, plot_predictions_vs_ground_truth_1, plot_predictions_vs_ground_truth_diff_colors, \
        plot_predictions_vs_ground_truth_diff_colors_withraven,plot_predictions_vs_ground_truth_diff_colors_withraven_axesequal,\
            save_full_run_state, plot_xyz_trajectories_from_four_sources, center_trajectories_by_mean, plot_xyz_trajectories_from_sources,plot_sensor_differences_all_trajectories,\
                plot_predictions_vs_ground_truth_diff_colors_withraven_axesequal_2, plot_xyz_trajectories_from_sources2, data_from_dict_fixed, integrate_position_from_gt, plot_predictions_vs_ground_truth_with_integrated,\
                    plot_xyz_trajectories_from_sources3,integrate_position_from_gt_shifted, compute_rmse_stats_and_save,plot_3d_valid_vs_train,save_execution_times, compute_mean_std_execution_times, plot_xyz_trajectories_from_sources4

print("Pipeline Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%d.%m.%y")
strNow = now.strftime("%H_%M_%S")
strTime = strToday + "__" + strNow
print("Current Time =", strTime)

###################
###  Settings   ###
###################
args = config.general_settings()
### dataset parameters
args.N_E = 457
args.N_CV = 81
args.N_T = 7
args.T = 85#30 #40#50 #30
args.T_test = 85#30#40#50 #30
### training parameters
args.use_cuda = False # use GPU or not
args.n_steps = 550 #250 # used to be 200
args.n_steps_EKF = 65
args.n_batch = 40 #20 #120#20 #used to be 128
args.lr = 1e-4 #0.0001 #1e-4#0.0001 #1e-4 #0.00002 - R=10 #1e-4 #used to be 1e-5 # when worked well on training was 1e-3
args.wd = 0.02 #1.5 #0.01 #0.1 #0.5 #0.5 #1e-1 #1e-4 # was 1e-2
args.to_center_trajs = False
args.CompositionLoss2 = True
args.alpha = 0.009 #0.009 #0.001 #0.003 #0.15 #0.0000099- R=10 #0.15#0.05#0.05

## Flags ##
EKF = False
if args.use_cuda:
    if torch.cuda.is_available():
      device = torch.device('cuda')
      args.device = torch.device('cuda')
      print("Using GPU")
    else:
      raise Exception("No GPU found, please set args.use_cuda = False")
else:
    device = torch.device('cpu')
    print("Using CPU")

offset = 0 # offset for the data
chop = False # whether to chop the dataset sequences into smaller ones
path_results = 'KNet/'
DatafolderName = 'Simulations/Raven_ii_matlab/data/'
DatafileName = 'processed_trajectories_devide.pt'


path_results2 = 'Runs/' + strTime

Q = 0.1*Q_structure #0.1* Q_structure
R = 10*R_structure #10*R_structure 


################# Load Data #################
pt_files = ["data/train_and_valid_1.pt", "data/train_and_valid_2.pt", "data/train_and_valid_3.pt"]  # Replace if needed
time, raven_x, raven_y, raven_z, marker_x, marker_y, marker_z, joint_pos_1, joint_pos_2, joint_pos_3 ,des_x, des_y, des_z = load_from_dict(pt_files,"data")

# correct desired pos
des_x = des_x/1000
des_y = des_y/1000
des_z = des_z/1000

# stack tensors 
raven_data = stack_tensors(raven_x,raven_y,raven_z).to(args.device)
marker_data = stack_tensors(marker_x,marker_y,marker_z).to(args.device)
observations = stack_tensors(joint_pos_1,joint_pos_2,joint_pos_3).to(args.device)
desired_position = stack_tensors(des_x,des_y,des_z).to(args.device)

# split data into traj of desired length
raven_data_traj = split_into_trajectories(raven_data,args.T)
marker_data_traj = split_into_trajectories(marker_data,args.T)
observations_traj = split_into_trajectories(observations,args.T)
desired_position_traj = split_into_trajectories(desired_position, args.T)

# if we want to shift by mean
if args.to_center_trajs:
    marker_data_traj, raven_data_traj = center_trajectories_by_mean(marker_data_traj, raven_data_traj)
    marker_data_traj.to(args.device)
    desired_position_traj.to(args.device)
    raven_data_traj.to(args.device)

#split raven data, ground truth and observations into train validation and test
set_train_cv_test_idx_from_new = False
if set_train_cv_test_idx_from_new:
    print('Splitting data into train validation and test according to new permutation.')
    train_input,cv_input, test_input, train_idx, cv_idx, test_idx, path_to_idx = split_trajectories(observations_traj,train_ratio=0.84, val_ratio=0.15, test_ratio=0.01)#train_ratio=0.79, val_ratio=0.2, test_ratio=0.01 # train_ratio=0.89, val_ratio=0.1, test_ratio=0.01 # used to be train_ratio=0.88, val_ratio=0.1, test_ratio=0.02
    train_target, cv_target, test_target = split_traj_by_given_idx(marker_data_traj,train_idx,cv_idx,test_idx)
    train_raven, cv_raven, test_raven = split_traj_by_given_idx(raven_data_traj,train_idx,cv_idx,test_idx)
    train_des, cv_des, test_des = split_traj_by_given_idx(desired_position_traj, train_idx,cv_idx,test_idx)
else:
    #path_to_idx = 'data/train_valid_test_indices_2025-03-03_14-35-22.txt' #'data/train_valid_test_indices_2025-02-17_14-53-12.txt'
    path_to_idx = 'data/train_valid_test_indices_2025-06-29_11-41-01.txt'
    print(f'Splitting data into train validation and test according to: {path_to_idx}')
    train_input,cv_input, test_input = load_indices_and_split_tensor(observations_traj, path_to_idx) # observations
    train_target, cv_target, test_target = load_indices_and_split_tensor(marker_data_traj, path_to_idx) # GT, target
    train_raven, cv_raven, test_raven = load_indices_and_split_tensor(raven_data_traj, path_to_idx) #raven data to compare to
    train_des, cv_des, test_des = load_indices_and_split_tensor(desired_position_traj, path_to_idx)

plot_3d_valid_vs_train(train_target,cv_target)
# calc delta distance 
train_des = calc_action(train_des).to(args.device)
cv_des = calc_action(cv_des).to(args.device)
test_des = calc_action(test_des).to(args.device)

##### add bias dimentions #####
train_target = add_bias_dimensions(train_target, n).to(args.device)
cv_target = add_bias_dimensions(cv_target, n).to(args.device)
test_target = add_bias_dimensions(test_target, n).to(args.device)

##### print sizes ######
print("load dataset to device:", train_input.device)
print("testset size:", test_target.size())
print("trainset size:", train_target.size())
print("cvset size:", cv_target.size())


# intiate m1 and m2
m1x_0 = train_target[:,:,0]
m2x_0 = Q

# # True Model
sys_model_true = SystemModel(f_with_action, Q, h, R, args.T, args.T_test,m,n)
sys_model_true.InitSequence(m1x_0, m2x_0)

######### EKF ###########

if EKF:
    batch_size = args.n_batch
    num_epochs = args.n_steps
    # Initialize the IK Model
    IK_model = InverseKinematicsNN()

    # If using CUDA
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    IK_model = IK_model.to(device)

    # Optimizer for training the IK model
    optimizer = torch.optim.Adam(IK_model.parameters(), lr=args.lr , weight_decay=args.wd)

    # Loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # To store the concatenated outputs across all trajectories
    all_EKF_out = []
    all_MSE = []

            # Run EKF and train the IK model for this batch
    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = EKF_test.EKF_Train(
          args, sys_model_true, IK_model,cv_input, cv_target, cv_des, train_input, train_target, train_des, optimizer, path_results)

    [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out,t] = EKF_test.EKFTest(args, sys_model_true, test_input, test_target, test_des, path_results) 


# Extract the first elements from the third dimension
test_init = test_target[:, :, 0]  
train_init = train_target[:, :, 0]  
cv_init = cv_target[:, :, 0]  

#  ensure correctness of m1
m1x_0 = train_target[:,:,0]
sys_model_true.m1x_0=m1x_0
cv_init=cv_target[:,:,0]
#train_init=train_target[:,:,0]
# # =============================================================================
# # # KalmanNet
# # =============================================================================
## Build Neural Network
KNet_model = KalmanNetNN() # builds init and IK networks
KNet_model.NNBuild(sys_model_true, args)
KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KNet_Pipeline.setModel(KNet_model)
KNet_Pipeline.setssModel(sys_model_true)
print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
# Train Neural Network
KNet_Pipeline.setTrainingParams(args)
Train_knet = False
if Train_knet:
    if(chop):
        KNet_Pipeline.NNTrain(sys_model_true,cv_input,cv_target,train_input,train_target,path_results,\
          randomInit=False,train_init=train_init)
    else:
        KNet_Pipeline.NNTrain(sys_model_true,cv_input,cv_target,train_input,train_target,cv_des,train_des,path_results)
    # Test Neural Network
    m1x_0 = test_target[:,:,0] #### change
    sys_model_true.m1x_0=m1x_0
    sys_model_true.InitSequence(m1x_0, m2x_0)
    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out,t] = KNet_Pipeline.NNTest(sys_model_true,test_input,test_target,test_des,path_results,MaskOnState=False, randomInit=False)
    is_equal = torch.equal(knet_out[:,:,0], test_target[:, :, 0])


test_model = torch.load(path_results + 'best-model.pt', map_location=device)
for name, param in test_model.named_parameters():
    print(f"{name}: Mean={param.data.mean():.4f}, Std={param.data.std():.4f}")
    
################# Test ##############
path_to_save = "data/Test_Data.pt"
loaded_tensors = torch.load(path_to_save)
time, raven_x, raven_y, raven_z, marker_x, marker_y, marker_z, joint_pos_1, joint_pos_2, joint_pos_3, des_x, des_y, des_z = data_from_dict_fixed(loaded_tensors) # was data_from_dict
raven_data = stack_tensors(raven_x,raven_y,raven_z).to(args.device)
marker_data = stack_tensors(marker_x,marker_y,marker_z).to(args.device)
observations = stack_tensors(joint_pos_1,joint_pos_2,joint_pos_3).to(args.device)
desired_position = stack_tensors(des_x,des_y,des_z).to(args.device)
desired_position = desired_position/1000

if args.to_center_trajs:
    marker_data, raven_data = center_trajectories_by_mean(marker_data, raven_data) # , desired_position
    marker_data.to(args.device)
    desired_position.to(args.device)
    raven_data.to(args.device)
    
desired_position = calc_action(desired_position)
marker_data = add_bias_dimensions(marker_data, n)
sys_model_true = SystemModel(f_with_action, Q, h, R, args.T, 1000,m,n) #3000:5000 12000:13000 1000:3000
# sys_model_true = SystemModel(f, Q, h, R, args.T, 300,m,n)
m1x_0 = marker_data[:,:,12000] #### change
sys_model_true.m1x_0=m1x_0
sys_model_true.InitSequence(m1x_0, m2x_0)
#knet
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out,t] = KNet_Pipeline.NNTest(sys_model_true,observations[:,:,12000:13000],marker_data[:,:,12000:13000],desired_position[:,:,12000:13000],path_results,MaskOnState=False, randomInit=False)

# Pt+1 = Pt + dist_t
pos_with_action = integrate_position_from_gt_shifted(marker_data[:,:,12000:13000],desired_position[:,:,12000:13000])
# EKF
EKF = True
if EKF:
    [MSE_EKF_linear_arr_test, MSE_EKF_linear_avg_test, MSE_EKF_dB_avg_test, KG_array_test, EKF_out_test, execution_time_EKF] = EKF_test.EKFTest(args, sys_model_true, observations[:,:,12000:13000], marker_data[:,:,12000:13000], desired_position[:,:,12000:13000], path_results) 
    labels= ["Ground Truth", "Algorithm 2", "Raven II", "Algorithm 1"] #,"State Evolution"
    mse_report = plot_xyz_trajectories_from_sources4(marker_data[:,0:3,12000:13000], knet_out[:,0:3,:], raven_data[:,:,12000:13000],EKF_out_test[:,0:3,:],labels=labels,colors=None, path_results2=path_results2)# ,pos_with_action
    plot_predictions_vs_ground_truth_diff_colors_withraven_axesequal_2(marker_data[:,:,12000:13000], knet_out, raven_data[:,:,12000:13000],EKF_out_test[:,0:3,:],labels = labels )
    
#plot_predictions_vs_ground_truth_diff_colors(marker_data[:,:,1000:3000], knet_out)
else: 
    plot_predictions_vs_ground_truth_diff_colors_withraven_axesequal(marker_data[:,:,12000:13000], knet_out, raven_data[:,:,12000:13000])# knet_out
    labels= ["Ground Truth (Markers)", "Knet", "Raven"]
    mse_report = plot_xyz_trajectories_from_sources(marker_data[:,:,12000:13000], knet_out, raven_data[:,:,12000:13000],labels= labels,colors=None, path_results2=path_results2)
    ####### save everything needed ##### 
save_full_run_state(folder_path=path_results2,pt_file_path= path_results + 'best-model.pt',EKF_pt_file_path= path_results + 'best-model-IKModel.pt', script_path="Main_KalmanNet_Raven.py", idx_file_path = path_to_idx)#__file__
print('====== sumary ==========')
if EKF:
    print('linear MSE EKF: ',MSE_EKF_linear_avg_test)
print('linear MSE Knet: ', MSE_test_linear_avg)

print("knet time:", t)
print("EKF time:", execution_time_EKF)


# Play default system sound when finished
winsound.MessageBeep()

# Or a custom frequency + duration beep:
winsound.Beep(frequency=1000, duration=800)
