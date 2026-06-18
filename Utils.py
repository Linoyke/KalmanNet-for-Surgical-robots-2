# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:57:21 2025

@author: User
"""

# utils for main_KalmanNet_Raven

import pandas as pd
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import pickle
import os
from datetime import date
from shutil import copy
import inspect
import types
import torch.nn as nn
import io
import re
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.collections import LineCollection


def read_data(path_to_file,path_to_save):
    # Load CSV file into a DataFrame
    df = pd.read_csv(path_to_file)
    
    # Convert DataFrame columns to a dictionary of PyTorch tensors
    tensors = {col: torch.tensor(df[col].values, dtype=torch.float32) for col in df.columns}
    
    #current_time
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%d_%m_%y")
    strNow = now.strftime("%H_%M_%S")
    strTime = strToday + "_" + strNow
    path_to_save = f'{path_to_save}/data_{strTime}.pt'
    # Save the dictionary of tensors to a file
    torch.save(tensors, path_to_save)
    print(f"pt file was save to: {path_to_save}")
    # Load it back to simulate reading directly using torch.load()
    loaded_tensors = torch.load(path_to_save)
    return loaded_tensors,path_to_save

def data_from_dict(loaded_tensors):
    
    # save each df column
    time = loaded_tensors['Raven_time']
    raven_x = loaded_tensors['raven_x_raven_frame']
    raven_y = loaded_tensors['raven_y_raven_frame']
    raven_z = loaded_tensors['raven_z_raven_frame']
    marker_x = loaded_tensors['marker_x_raven_frame_int']
    marker_y = loaded_tensors['marker_y_raven_fram_int'] ##### change to frame after fixing the code #######
    marker_z = loaded_tensors['marker_z_raven_frame_int']
    joint_pos_1 = loaded_tensors['Joint_pos_1_']
    joint_pos_2 = loaded_tensors['Joint_pos_1__1'] ####### cahnge to Joint_pos_2_ when working with data from the fixed code ########
    joint_pos_3 = loaded_tensors['Joint_pos_2_'] ####### cahnge to Joint_pos_3_ when working with data from the fixed code ########
    des_x = loaded_tensors['des_x']
    des_y = loaded_tensors['des_y'] 
    des_z = loaded_tensors['des_z'] 
    
    return time, raven_x, raven_y, raven_z, marker_x, marker_y, marker_z, joint_pos_1, joint_pos_2, joint_pos_3, des_x, des_y, des_z

def load_from_dict(pt_files,path_to_save):
    # Initialize an empty dictionary to store concatenated tensors
    merged_tensors = {
        "Raven_time": None,
        "raven_x_raven_frame": None,
        "raven_y_raven_frame": None,
        "raven_z_raven_frame": None,
        "marker_x_raven_frame_int": None,
        "marker_y_raven_fram_int": None,  # Change to 'marker_y_raven_frame_int' after fixing
        "marker_z_raven_frame_int": None,
        "Joint_pos_1_": None,
        "Joint_pos_2_": None,  # Change to 'Joint_pos_2_' after fixing
        "Joint_pos_3_": None,  # Change to 'Joint_pos_3_' after fixing
        # "motor_pos_1_": None,
        # "motor_pos_2_": None,  # Change to 'Joint_pos_2_' after fixing
        # "motor_pos_3_": None,  # Change to 'Joint_pos_3_' after fixing
        "des_x": None,
        "des_y": None, 
        "des_z": None,
    }

    for file in pt_files:
        loaded_tensors = torch.load(file)

        for key in merged_tensors.keys():
            if key in loaded_tensors:
                if merged_tensors[key] is None:
                    merged_tensors[key] = loaded_tensors[key].clone()
                else:
                    merged_tensors[key] = torch.cat((merged_tensors[key], loaded_tensors[key]))
    #current_time
    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%d_%m_%y")
    strNow = now.strftime("%H_%M_%S")
    strTime = strToday + "_" + strNow
    path_to_save = f'{path_to_save}/merged_data_{strTime}.pt'
    # Save the dictionary of tensors to a file
    torch.save(merged_tensors, path_to_save)
    print(f"pt file was save to: {path_to_save}")

    return (
        merged_tensors["Raven_time"],
        merged_tensors["raven_x_raven_frame"],
        merged_tensors["raven_y_raven_frame"],
        merged_tensors["raven_z_raven_frame"],
        merged_tensors["marker_x_raven_frame_int"],
        merged_tensors["marker_y_raven_fram_int"],  # Change to 'marker_y_raven_frame_int' after fixing
        merged_tensors["marker_z_raven_frame_int"],
        merged_tensors["Joint_pos_1_"],
        merged_tensors["Joint_pos_2_"],  # Change to 'Joint_pos_2_' after fixing
        merged_tensors["Joint_pos_3_"],  # Change to 'Joint_pos_3_' after fixing
        # merged_tensors["motor_pos_1_"],
        # merged_tensors["motor_pos_2_"],  # Change to 'Joint_pos_2_' after fixing
        # merged_tensors["motor_pos_3_"],  # Change to 'Joint_pos_3_' after fixing
        merged_tensors["des_x"],
        merged_tensors["des_y"],
        merged_tensors["des_z"],
    )

def data_from_dict_fixed(loaded_tensors):
    
    # save each df column
    time = loaded_tensors['Raven_time']
    raven_x = loaded_tensors['raven_x_raven_frame']
    raven_y = loaded_tensors['raven_y_raven_frame']
    raven_z = loaded_tensors['raven_z_raven_frame']
    marker_x = loaded_tensors['marker_x_raven_frame_int']
    marker_y = loaded_tensors['marker_y_raven_fram_int'] ##### change to frame after fixing the code #######
    marker_z = loaded_tensors['marker_z_raven_frame_int']
    joint_pos_1 = loaded_tensors['Joint_pos_1_']
    joint_pos_2 = loaded_tensors['Joint_pos_2_'] ####### cahnge to Joint_pos_2_ when working with data from the fixed code ########
    joint_pos_3 = loaded_tensors['Joint_pos_3_'] ####### cahnge to Joint_pos_3_ when working with data from the fixed code ########
    des_x = loaded_tensors['des_x']
    des_y = loaded_tensors['des_y'] 
    des_z = loaded_tensors['des_z'] 
    
    return time, raven_x, raven_y, raven_z, marker_x, marker_y, marker_z, joint_pos_1, joint_pos_2, joint_pos_3, des_x, des_y, des_z


def stack_tensors(t1,t2,t3):
    # Stack along dimension 0 to get shape (3, timesteps)
    stacked = torch.stack([t1, t2, t3], dim=0)  # Shape: (3, timeste[s])

    # Add a batch dimension (unsqueeze at dim=0) to get shape (1, 3, timesteps)
    final_tensor = stacked.unsqueeze(0)  # Shape: (1, 3, timesteps)
    return final_tensor

def split_into_trajectories(tensor, num_steps):
    """
    Efficiently splits a trajectory tensor of shape (1, 3, timesteps) into multiple sub-trajectories
    using PyTorch's unfold to avoid loops.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (1, 3, timesteps).
        num_steps (int): Number of timesteps per trajectory.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, num_steps) containing multiple trajectories.
    """
    # Ensure input shape is (1, 3, timesteps)
    if tensor.dim() != 3 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (1, 3, timesteps)")

    # Extract the timesteps
    timesteps = tensor.shape[2]

    # Compute number of full trajectories that fit
    N = timesteps // num_steps

    # Slice to only include full sequences
    tensor = tensor[:, :, :N * num_steps]  # Ensure full sequences fit

    # Use `reshape` to split without looping
    result_tensor = tensor.reshape(1, 3, N, num_steps).squeeze(0).permute(1, 0, 2)

    return result_tensor

def split_trajectories(tensor, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits the input tensor of trajectories into train, validation, and test sets.
    Saves the indices of each set into a text file with the current date and time.
    
    Parameters:
        tensor (torch.Tensor): Input tensor of shape (N, 3, num_steps).
        train_ratio (float): The ratio of trajectories for the train set.
        val_ratio (float): The ratio of trajectories for the validation set.
        test_ratio (float): The ratio of trajectories for the test set.

    Returns:
        train_tensor (torch.Tensor): The tensor for the train set.
        val_tensor (torch.Tensor): The tensor for the validation set.
        test_tensor (torch.Tensor): The tensor for the test set.
        train_indices (list): List of indices for the train set.
        val_indices (list): List of indices for the validation set.
        test_indices (list): List of indices for the test set.
    """
    
    # Ensure the input tensor is of shape (N, 3, num_steps)
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D (N, 3, num_steps)")

    N = tensor.shape[0]  # Number of trajectories
    
    # Create a list of indices [0, 1, ..., N-1]
    indices = np.arange(N)
    
    # Shuffle indices randomly
    np.random.shuffle(indices)
    
    # Split the indices according to the ratios
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create the train, validation, and test tensors using the split indices
    train_tensor = tensor[train_indices]
    val_tensor = tensor[val_indices]
    test_tensor = tensor[test_indices]
    
    # Save the indices to a file with current date and time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/train_valid_test_indices_{now}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Train indices: {train_indices.tolist()}\n")
        f.write(f"Validation indices: {val_indices.tolist()}\n")
        f.write(f"Test indices: {test_indices.tolist()}\n")
    
    print(f"Saved indices of train validation and test to: {filename}")
    
    return train_tensor, val_tensor, test_tensor, train_indices.tolist(), val_indices.tolist(), test_indices.tolist(), filename

def split_traj_by_given_idx(tensor, train_idx, cv_idx, test_idx):
    train = tensor[train_idx]
    cv = tensor[cv_idx]
    test = tensor[test_idx]
    return train, cv, test

def load_indices_and_split_tensor(tensor, indices_file_path):
    """
    Loads the indices from the specified text file and splits the input tensor into
    train, validation, and test tensors based on these indices.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (N, 3, num_steps).
        indices_file_path (str): Path to the text file containing the indices of
                                  train, validation, and test sets.

    Returns:
        train_tensor (torch.Tensor): The tensor for the train set.
        val_tensor (torch.Tensor): The tensor for the validation set.
        test_tensor (torch.Tensor): The tensor for the test set.
    """
    
    # Load indices from the file
    with open(indices_file_path, 'r') as f:
        lines = f.readlines()

    # Parse the indices from the file
    train_indices = eval(lines[0].split(":")[1].strip())  # Remove "Train indices: " and parse the list
    val_indices = eval(lines[1].split(":")[1].strip())    # Remove "Validation indices: " and parse the list
    test_indices = eval(lines[2].split(":")[1].strip())   # Remove "Test indices: " and parse the list

    # Convert indices lists to numpy arrays for easier indexing
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Split the tensor based on the loaded indices
    train_tensor = tensor[train_indices]
    val_tensor = tensor[val_indices]
    test_tensor = tensor[test_indices]

    return train_tensor, val_tensor, test_tensor

def add_bias_dimensions(tensor, n, fill_value=0.02):
    """
    Adds `n` new dimensions filled with `fill_value` to a tensor of shape [N, 3, timesteps],
    resulting in a tensor of shape [N, 3+n, timesteps].

    Parameters:
        tensor (torch.Tensor): Input tensor of shape [N, 3, timesteps].
        n (int): Number of new dimensions to add.
        fill_value (float, optional): Value to fill the new dimensions with. Default is 0.2.

    Returns:
        torch.Tensor: Tensor of shape [N, 3+n, timesteps].
    """
    N, _, timesteps = tensor.shape  # Extract original shape

    # Create new dimensions filled with `fill_value`
    new_dims = torch.full((N, n, timesteps), fill_value, dtype=tensor.dtype, device=tensor.device)

    # Concatenate along the second dimension (feature dimension)
    return torch.cat([tensor, new_dims], dim=1)

def calc_action(desired_positions):
    delta = torch.diff(desired_positions, dim=2, prepend=torch.zeros_like(desired_positions[:, :, :1]))
    delta[:,:,0] = 0
    return delta
# def project_to_box(position, box_dims):
#     """
#     Projects a 3D point into a fixed bounding box centered at (0,0,0).
    
#     Parameters:
#         position (tuple): (x, y, z) coordinates of the point.
#         box_dims (tuple): (x_box, y_box, z_box) defining the box dimensions.
    
#     Returns:
#         tuple: Projected (x, y, z) coordinates within the box.
#     """
#     x, y, z = position
#     x_box, y_box, z_box = box_dims

#     # Define box bounds (half-lengths)
#     x_min, x_max = -x_box / 2, x_box / 2
#     y_min, y_max = -y_box / 2, y_box / 2
#     z_min, z_max = -z_box / 2, z_box / 2

#     # Project (clamp) the point inside the box
#     x_proj = np.clip(x, x_min, x_max)
#     y_proj = np.clip(y, y_min, y_max)
#     z_proj = np.clip(z, z_min, z_max)

#     return (x_proj, y_proj, z_proj)

def project_to_box(tensor, x_limits, y_limits, z_limits):
    """
    Projects the x, y, z values in the tensor to stay within the given limits.
    
    Parameters:
    - tensor: torch.Tensor of shape [N_trajectories, 6, timesteps]
    - x_limits: tuple (x_min, x_max)
    - y_limits: tuple (y_min, y_max)
    - z_limits: tuple (z_min, z_max)

    Returns:
    - Modified tensor with x, y, z values clamped within the limits.
    """
    # Apply clamping without modifying the tensor in-place
    tensor_x = torch.clamp(tensor[:, 0], x_limits[0], x_limits[1]).unsqueeze(1)  # Keep shape [N_batch, 1]
    tensor_y = torch.clamp(tensor[:, 1], y_limits[0], y_limits[1]).unsqueeze(1)
    tensor_z = torch.clamp(tensor[:, 2], z_limits[0], z_limits[1]).unsqueeze(1)
    
    # Preserve other dimensions (assuming tensor has 6 features)
    tensor_rest = tensor[:, 3:].clone()  # Copy the last 3 dimensions to avoid in-place modification
    
    # Concatenate everything back together
    return torch.cat([tensor_x, tensor_y, tensor_z, tensor_rest], dim=1)


def plot_training_curves(MSE_train_dB_epoch, MSE_cv_dB_epoch, save_path=None):
    """
    # Plots the training and validation loss curves over training steps.

    # Parameters:
    # - MSE_train_dB_epoch: List or tensor of training loss in dB over epochs.
    # - MSE_cv_dB_epoch: List or tensor of validation loss in dB over epochs.
    # - save_path: (Optional) Base path to save the plot as an image. The function will append a timestamp to the filename.
    # """
    # plt.figure(figsize=(10, 5))
    # plt.plot(MSE_train_dB_epoch, label="Training Loss (dB)", marker="o", linestyle="-")
    # plt.plot(MSE_cv_dB_epoch, label="Validation Loss (dB)", marker="s", linestyle="--")

    # plt.xlabel("Training Steps")
    # plt.ylabel("Loss (dB)")
    # plt.title("Training and Validation Loss vs Training Steps")
    # plt.legend()
    # plt.grid(True)

    # # Generate timestamped filename
    # if save_path:
    #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     save_path = f"{save_path}_{timestamp}.png"
    #     plt.savefig(save_path, dpi=300)
    #     print(f"Plot saved as {save_path}")

    # plt.show()
    """
    Plots the training and validation loss curves over training steps.
    
    Parameters:
    - MSE_train_dB_epoch: List or tensor of training loss in dB over epochs.
    - MSE_cv_dB_epoch: List or tensor of validation loss in dB over epochs.
    - save_path: (Optional) Base path to save the plot as an image. The function will append a timestamp to the filename.
    """
    # Default title
    title = "Training and Validation Loss vs Training Steps"
    
    # Try to extract method name from save_path
    if save_path:
        base_name = os.path.basename(save_path)  # Strip path
        if base_name.startswith("loss_curve_") and base_name.endswith(".png"):
            method_name = base_name[len("loss_curve_"):-len(".png")]
            title += f" ({method_name})"
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(MSE_train_dB_epoch, label="Training Loss (dB)", marker="o", linestyle="-")
    plt.plot(MSE_cv_dB_epoch, label="Validation Loss (dB)", marker="s", linestyle="--")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss (dB)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save figure
    if save_path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path_with_time = f"{save_path[:-4]}_{timestamp}.png"  # Insert timestamp before .png
        plt.savefig(save_path_with_time, dpi=300)
        print(f"Plot saved as {save_path_with_time}")
    
    plt.show()
    
def plot_trajectories(train_target, cv_target, test_target):
    """
    Plots 3D trajectories for train, validation (cv), and test sets.
    
    Parameters:
    - train_target: Tensor of shape (N_train, 6, 30)
    - cv_target: Tensor of shape (N_cv, 6, 30)
    - test_target: Tensor of shape (N_test, 6, 30)
    
    The first three variables (x, y, z) are used for plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each dataset
    colors = {
        "train": "blue",
        "cv": "green",
        "test": "red"
    }

    # Function to plot trajectories
    def plot_tensor_trajectories(tensor, color, label):
        N = tensor.shape[0]  # Number of trajectories
        for i in range(N):
            x = tensor[i, 0, :].cpu().numpy()  # Extract X coordinates
            y = tensor[i, 1, :].cpu().numpy()  # Extract Y coordinates
            z = tensor[i, 2, :].cpu().numpy()  # Extract Z coordinates
            alpha = 0.6  # Transparency for better visibility

            # Use varying line styles for better distinction
            line_styles = ['-', '--', '-.', ':']
            linestyle = line_styles[i % len(line_styles)]

            ax.plot(x, y, z, linestyle=linestyle, color=color, alpha=alpha, label=label if i == 0 else None)

    # Plot each set with its assigned color
    plot_tensor_trajectories(train_target, colors["train"], "Train")
    plot_tensor_trajectories(cv_target, colors["cv"], "Validation")
    plot_tensor_trajectories(test_target, colors["test"], "Test")

    # Labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories of Train, Validation, and Test Sets")
    ax.legend()

    plt.show()
    
def plot_predictions_vs_ground_truth(test_target, model_output):
    """
    Plots 3D trajectories of test_target (ground truth) and model_output (predictions).
    
    Parameters:
    - test_target: Tensor of shape (N, 6, 30) -> Ground truth trajectories.
    - model_output: Tensor of shape (N, 6, 30) -> Model-predicted trajectories.
    
    The first three variables (x, y, z) are used for plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Number of trajectories
    colormap = cm.get_cmap("tab10", N)  # Use a color map for N different colors

    for i in range(N):
        # Extract X, Y, Z coordinates and detach from computation graph
        x_true, y_true, z_true = test_target[i, 0, :].detach().cpu().numpy(), \
                                 test_target[i, 1, :].detach().cpu().numpy(), \
                                 test_target[i, 2, :].detach().cpu().numpy()

        x_pred, y_pred, z_pred = model_output[i, 0, :].detach().cpu().numpy(), \
                                 model_output[i, 1, :].detach().cpu().numpy(), \
                                 model_output[i, 2, :].detach().cpu().numpy()

        color = colormap(i)  # Assign a unique color to each trajectory

        # Plot ground truth with a solid line
        ax.plot(x_true, y_true, z_true, linestyle='-', color=color, alpha=0.8, label=f"Ground Truth {i}" if i == 0 else None)

        # Plot model predictions with a dashed line
        ax.plot(x_pred, y_pred, z_pred, linestyle='--', color=color, alpha=0.8, label=f"Prediction {i}" if i == 0 else None)

    # Labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories: Ground Truth vs Predictions")
    ax.legend()
    
    plt.show()

def plot_predictions_vs_ground_truth_1(test_target, model_output):
    """
    Plots 3D trajectories of test_target (ground truth) and model_output (predictions).
    
    Parameters:
    - test_target: Tensor of shape (N, 6, 30) -> Ground truth trajectories.
    - model_output: Tensor of shape (N, 6, 30) -> Model-predicted trajectories.
    
    The first three variables (x, y, z) are used for plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Number of trajectories
    colormap = cm.get_cmap("tab10", N)  # Use a color map for N different colors

    for i in range(N):
        # Extract X, Y, Z coordinates and detach from computation graph
        x_true, y_true, z_true = test_target[ 0, :].detach().cpu().numpy(), \
                                 test_target[ 1, :].detach().cpu().numpy(), \
                                 test_target[ 2, :].detach().cpu().numpy()

        x_pred, y_pred, z_pred = model_output[ 0, :].detach().cpu().numpy(), \
                                 model_output[ 1, :].detach().cpu().numpy(), \
                                 model_output[ 2, :].detach().cpu().numpy()

        color = colormap(i)  # Assign a unique color to each trajectory

        # Plot ground truth with a solid line
        ax.plot(x_true, y_true, z_true, linestyle='-', color=color, alpha=0.8, label=f"Ground Truth {i}" if i == 0 else None)

        # Plot model predictions with a dashed line
        ax.plot(x_pred, y_pred, z_pred, linestyle='--', color=color, alpha=0.8, label=f"Prediction {i}" if i == 0 else None)

    # Labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories: Ground Truth vs Predictions")
    ax.legend()
    
    plt.show()
    
def plot_predictions_vs_ground_truth_diff_colors(test_target, model_output):
    """
    Plots 3D trajectories of test_target (ground truth) and model_output (predictions).
    
    Parameters:
    - test_target: Tensor of shape (N, 6, T) -> Ground truth trajectories.
    - model_output: Tensor of shape (N, 6, T) -> Model-predicted trajectories.
    
    The first three variables (x, y, z) are used for plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Number of trajectories

    for i in range(N):
        # Extract X, Y, Z coordinates and detach from computation graph
        x_true, y_true, z_true = test_target[i, 0, :].detach().cpu().numpy(), \
                                 test_target[i, 1, :].detach().cpu().numpy(), \
                                 test_target[i, 2, :].detach().cpu().numpy()

        x_pred, y_pred, z_pred = model_output[i, 0, :].detach().cpu().numpy(), \
                                 model_output[i, 1, :].detach().cpu().numpy(), \
                                 model_output[i, 2, :].detach().cpu().numpy()

        # Ground Truth -> Blue Solid Line
        ax.plot(x_true, y_true, z_true, linestyle='-', color='blue', alpha=0.8, label="Ground Truth" if i == 0 else None)

        # Prediction -> Red Dashed Line
        ax.plot(x_pred, y_pred, z_pred, linestyle='--', color='red', alpha=0.8, label="Prediction" if i == 0 else None)

    # Labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories: Ground Truth vs Predictions")
    ax.legend()
    
    plt.show()
    
def plot_predictions_vs_ground_truth_diff_colors_withraven(test_target, model_output, raven_data):
    """
    Plots 3D trajectories of test_target (ground truth) and model_output (predictions).
    
    Parameters:
    - test_target: Tensor of shape (N, 6, T) -> Ground truth trajectories.
    - model_output: Tensor of shape (N, 6, T) -> Model-predicted trajectories.
    
    The first three variables (x, y, z) are used for plotting.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Number of trajectories

    for i in range(N):
        # Extract X, Y, Z coordinates and detach from computation graph
        x_true, y_true, z_true = test_target[i, 0, :].detach().cpu().numpy(), \
                                 test_target[i, 1, :].detach().cpu().numpy(), \
                                 test_target[i, 2, :].detach().cpu().numpy()

        x_pred, y_pred, z_pred = model_output[i, 0, :].detach().cpu().numpy(), \
                                 model_output[i, 1, :].detach().cpu().numpy(), \
                                 model_output[i, 2, :].detach().cpu().numpy()
        x_r, y_r, z_r = raven_data[i, 0, :].detach().cpu().numpy(), \
                                 raven_data[i, 1, :].detach().cpu().numpy(), \
                                 raven_data[i, 2, :].detach().cpu().numpy()
        # Ground Truth -> Blue Solid Line
        ax.plot(x_true, y_true, z_true, linestyle='-', color='blue', alpha=0.8, label="Ground Truth" if i == 0 else None)

        # Prediction -> Red Dashed Line
        ax.plot(x_pred, y_pred, z_pred, linestyle='--', color='red', alpha=0.8, label="Prediction" if i == 0 else None)
        
        ax.plot(x_r, y_r, z_r, linestyle='dashdot', color='green', alpha=0.8, label="Raven data" if i == 0 else None)

    # Labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories: Ground Truth vs Predictions")
    ax.legend()
    
    plt.show()
    
def plot_predictions_vs_ground_truth_diff_colors_withraven_axesequal(test_target, model_output, raven_data):
    """
    Plots 3D trajectories of test_target (ground truth), model_output (predictions), and raven_data.
    Sets axes to equal scale.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Number of trajectories

    # Collect all points for equal axis scaling
    all_x, all_y, all_z = [], [], []

    for i in range(N):
        x_true, y_true, z_true = test_target[i, 0, :].detach().cpu().numpy(), \
                                 test_target[i, 1, :].detach().cpu().numpy(), \
                                 test_target[i, 2, :].detach().cpu().numpy()
        x_pred, y_pred, z_pred = model_output[i, 0, :].detach().cpu().numpy(), \
                                 model_output[i, 1, :].detach().cpu().numpy(), \
                                 model_output[i, 2, :].detach().cpu().numpy()
        x_r, y_r, z_r = raven_data[i, 0, :].detach().cpu().numpy(), \
                        raven_data[i, 1, :].detach().cpu().numpy(), \
                        raven_data[i, 2, :].detach().cpu().numpy()

        # Plot lines
        ax.plot(x_true, y_true, z_true, linestyle='-', color='blue', alpha=0.8, label="Ground Truth" if i == 0 else None)
        ax.plot(x_pred, y_pred, z_pred, linestyle='--', color='red', alpha=0.8, label="Prediction" if i == 0 else None)
        ax.plot(x_r, y_r, z_r, linestyle='dashdot', color='green', alpha=0.8, label="Raven data" if i == 0 else None)

        # Accumulate points
        all_x.extend(np.concatenate([x_true, x_pred, x_r]))
        all_y.extend(np.concatenate([y_true, y_pred, y_r]))
        all_z.extend(np.concatenate([z_true, z_pred, z_r]))

    # Set equal axes
    x_range = np.array(all_x)
    y_range = np.array(all_y)
    z_range = np.array(all_z)
    max_range = np.array([x_range.max()-x_range.min(),
                          y_range.max()-y_range.min(),
                          z_range.max()-z_range.min()]).max() / 2.0

    mid_x = (x_range.max()+x_range.min()) * 0.5
    mid_y = (y_range.max()+y_range.min()) * 0.5
    mid_z = (z_range.max()+z_range.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories: Ground Truth vs Predictions")
    ax.legend()

    plt.show()
    
def save_plots(path):
    today_str = date.today().isoformat()  # Format: YYYY-MM-DD
    folder_path = os.path.join(path, today_str)
    os.makedirs(folder_path, exist_ok=True)
    
    # Get all open figure numbers
    fig_nums = plt.get_fignums()
    
    # Save each figure
    for i in fig_nums:
        fig = plt.figure(i)
        filename = os.path.join(folder_path, f'figure_{i}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(fig, f)
    
    print(f"Saved {len(fig_nums)} figure(s) to {folder_path}")    
    

def save_full_run_state(folder_path,pt_file_path,EKF_pt_file_path, script_path=None, idx_file_path=None):
    """
    Saves variables, all open figures, and a backup of the current script to the folder.
    """
    # 1. Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)
    print(f"[INFO] Using run folder: {folder_path}")

    # 2. Save all non-callable, non-built-in global variables
    variables_to_save = {
    k: v for k, v in globals().items()
    if not k.startswith('__')
    and not callable(v)
    and not isinstance(v, types.ModuleType)
    }
    vars_path = os.path.join(folder_path, 'variables.pkl')
    with open(vars_path, 'wb') as f:
        pickle.dump(variables_to_save, f)
    print(f"[INFO] Saved variables to {vars_path}")

    # 3. Save all open matplotlib figures
    figs = [plt.figure(i) for i in plt.get_fignums()]
    base_time = datetime.now().strftime('%d_%m_%Y_%H_%M')
    for i, fig in enumerate(figs, start=1):
        fig_filename = f"{base_time}_fig_{i}.png"
        fig_path = os.path.join(folder_path, fig_filename)
        fig.savefig(fig_path)
        print(f"[INFO] Saved figure to {fig_path}")

    # 4. Save a snapshot of the running script
    if script_path is None:
        # Try to automatically get the current script path
        try:
            script_path = inspect.getfile(inspect.currentframe().f_back)
        except Exception as e:
            print(f"[WARNING] Could not infer script path: {e}")
            script_path = None

    if script_path and os.path.isfile(script_path):
        dest_path = os.path.join(folder_path, os.path.basename(script_path))
        copy(script_path, dest_path)
        print(f"[INFO] Script backup saved to {dest_path}")
    else:
        print("[WARNING] Script path not found. Backup not saved.")
        
    if pt_file_path and os.path.isfile(pt_file_path):
        dest_path = os.path.join(folder_path, os.path.basename(pt_file_path))
        copy(pt_file_path, dest_path)
        print(f"[INFO] KNET .pt file copied to {dest_path}")
    elif pt_file_path:
        print(f"[WARNING] KNET .pt file not found at: {pt_file_path}")
        
    if EKF_pt_file_path and os.path.isfile(EKF_pt_file_path):
        dest_path = os.path.join(folder_path, os.path.basename(EKF_pt_file_path))
        copy(EKF_pt_file_path, dest_path)
        print(f"[INFO] EKF .pt file copied to {dest_path}")
    elif EKF_pt_file_path:
        print(f"[WARNING] EKF .pt file not found at: {EKF_pt_file_path}")
        
    if idx_file_path and os.path.isfile(idx_file_path):
        dest_path = os.path.join(folder_path, os.path.basename(idx_file_path))
        copy(idx_file_path, dest_path)
        print(f"[INFO] Index file copied to {dest_path}")
    elif idx_file_path:
        print(f"[WARNING] Index file not found at: {idx_file_path}")
        
        ## TODO:
            # add saving MSE of knet,EKF,raven total and each axis in a txt file
            # plot of MSE x y z
            # plot each result in x y z seperatly
def plot_xyz_trajectories_from_four_sources(tensor1, tensor2, tensor3, tensor4,
                                             labels=None, colors=None,
                                             path_results2=None):
    """
    Plot X, Y, Z components over time for 4 input tensors and compute axis-wise and total MSE.
    Saves a report to 'path_results2/mse_report.txt' if path_results2 is provided.

    Each tensor should have shape [N, D, T], where D ≥ 3 (X, Y, Z in dims 0, 1, 2).

    Args:
        tensor1, tensor2, tensor3, tensor4: tensors of shape [N, D, T]
        labels: list of 4 strings for legend labels
        colors: list of 4 strings for line colors
        path_results2: path to save the report (e.g., 'Runs/2025-05-04_17-52-00')
    
    Returns:
        mse_report (str): Formatted MSE report text
    """
    tensors = [tensor1, tensor2, tensor3, tensor4]
    assert all(t.shape[1] >= 3 for t in tensors), "Each tensor must have at least 3 features (X, Y, Z)"
    
    N = tensor1.shape[0]
    T = tensor1.shape[2]

    if labels is None:
        labels = [f"Tensor {i+1}" for i in range(4)]
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple']

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    dim_names = ['X', 'Y', 'Z']
    loss_fn = nn.MSELoss(reduction='mean')

    # Plotting
    for d in range(3):  # For X, Y, Z
        ax = axs[d]
        for i, tensor in enumerate(tensors):
            for n in range(N):
                values = tensor[n, d, :].detach().cpu().numpy()
                ax.plot(range(T), values, color=colors[i], alpha=0.6,
                        label=labels[i] if n == 0 else None)
        ax.set_ylabel(f"{dim_names[d]} Position")
        ax.legend()
        ax.grid(True)

    axs[2].set_xlabel("Timestep")
    fig.suptitle("Position Components over Time (X, Y, Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # MSE Reporting
    buffer = io.StringIO()
    buffer.write("Mean Squared Error vs Ground Truth:\n")
    for i, tensor in enumerate(tensors[1:], start=1):  # Skip tensor1
        buffer.write(f"\n{labels[i]}:\n")
        total_mse = 0.0
        for d, dim_name in enumerate(dim_names):
            gt = tensor1[:, d, :].reshape(-1)
            pred = tensor[:, d, :].reshape(-1)
            mse = loss_fn(pred, gt).item()
            total_mse += mse
            buffer.write(f"  {dim_name} MSE: {mse:.6f}\n")
        buffer.write(f"  Total MSE: {total_mse:.6f}\n")

    mse_report = buffer.getvalue()
    buffer.close()

    print(mse_report)

    # Save to file if path_results2 is provided
    if path_results2 is not None:
        os.makedirs(path_results2, exist_ok=True)
        file_path = os.path.join(path_results2, "mse_report.txt")
        with open(file_path, "w") as f:
            f.write(mse_report)
        print(f"[Saved MSE report to: {file_path}]")

    return mse_report

def plot_xyz_trajectories_from_sources(*tensors, labels=None, colors=None, path_results2=None):
    """
    Plot X, Y, Z components over time for up to 4 input tensors and compute axis-wise and total MSE.
    Saves a report to 'path_results2/mse_report.txt' if path_results2 is provided.

    Each tensor should have shape [N, D, T], where D ≥ 3 (X, Y, Z in dims 0, 1, 2).

    Args:
        *tensors: 2 to 4 tensors of shape [N, D, T]
        labels: list of legend labels (optional; auto-generated from variable names)
        colors: list of line colors (optional)
        path_results2: path to save the report
    
    Returns:
        rmse_report (str): Formatted MSE report text
    """
    assert 2 <= len(tensors) <= 4, "Provide between 2 and 4 tensors."
    assert all(t.shape[1] >= 3 for t in tensors), "Each tensor must have at least 3 features (X, Y, Z)"

    N, _, T = tensors[0].shape
    dim_names = ['X', 'Y', 'Z']
    loss_fn = nn.MSELoss(reduction='mean')

    # Get variable names if labels are not provided
    if labels is None:
        frame = inspect.currentframe().f_back
        labels = [name for name, val in frame.f_locals.items() if val in tensors][:len(tensors)]
        while len(labels) < len(tensors):
            labels.append(f"Tensor {len(labels) + 1}")

    if colors is None:
        default_colors = ['blue', 'red', 'green', 'purple']
        colors = default_colors[:len(tensors)]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for d in range(3):  # X, Y, Z
        ax = axs[d]
        for i, tensor in enumerate(tensors):
            for n in range(N):
                values = tensor[n, d, :].detach().cpu().numpy()
                ax.plot(range(T), values, color=colors[i], alpha=0.6,
                        label=labels[i] if n == 0 else None)
        ax.set_ylabel(f"{dim_names[d]} Position [mm]")
        ax.legend()
        ax.grid(True)
    axs[2].set_xlabel("Timestep")
    fig.suptitle("Position Components over Time (X, Y, Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # RMSE Reporting
    buffer = io.StringIO()
    buffer.write("Mean Squared Error vs Ground Truth:\n")
    for i, tensor in enumerate(tensors[1:], start=1):  # Compare to tensor[0]
        buffer.write(f"\n{labels[i]}:\n")
        total_mse = 0.0
        total_rmse = 0.0
        for d, dim_name in enumerate(dim_names):
            gt = tensors[0][:, d, :].reshape(-1)
            pred = tensor[:, d, :].reshape(-1)
            mse = loss_fn(pred, gt)
            rmse = torch.sqrt(mse).item()  # Convert to RMSE
            total_mse += mse
            # total_rmse += rmse
        
            buffer.write(f"  {dim_name} RMSE: {rmse:.6f}\n")
        total_rmse = torch.sqrt(total_mse).item
        buffer.write(f"  Total RMSE: {total_rmse:.6f}\n")

    rmse_report = buffer.getvalue()
    buffer.close()

    print(rmse_report)

    # Save report
    if path_results2 is not None:
        os.makedirs(path_results2, exist_ok=True)
        file_path = os.path.join(path_results2, "mse_report.txt")
        with open(file_path, "w") as f:
            f.write(rmse_report)
        print(f"[Saved MSE report to: {file_path}]")

    return rmse_report

def plot_xyz_trajectories_from_sources2(*tensors, labels=None, colors=None, path_results2=None):
    """
    Plot X, Y, Z components over time for up to 4 input tensors and compute axis-wise and total RMSE.
    Saves a report to 'path_results2/mse_report.txt' if path_results2 is provided.

    Each tensor should have shape [N, D, T], where D ≥ 3 (X, Y, Z in dims 0, 1, 2).

    Args:
        *tensors: 2 to 4 tensors of shape [N, D, T]
        labels: list of legend labels (optional; auto-generated from variable names)
        colors: list of line colors (optional)
        path_results2: path to save the report
    
    Returns:
        rmse_report (str): Formatted RMSE report text
    """
    assert 2 <= len(tensors) <= 4, "Provide between 2 and 4 tensors."
    assert all(t.shape[1] >= 3 for t in tensors), "Each tensor must have at least 3 features (X, Y, Z)"

    N, _, T = tensors[0].shape
    dim_names = ['X', 'Y', 'Z']
    loss_fn = nn.MSELoss(reduction='mean')

    if labels is None:
        frame = inspect.currentframe().f_back
        labels = [name for name, val in frame.f_locals.items() if val in tensors][:len(tensors)]
        while len(labels) < len(tensors):
            labels.append(f"Tensor {len(labels) + 1}")

    if colors is None:
        default_colors = ['blue', 'red', 'green', 'purple']
        colors = default_colors[:len(tensors)]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for d in range(3):  # X, Y, Z
        ax = axs[d]
        for i, tensor in enumerate(tensors):
            for n in range(N):
                values = tensor[n, d, :].detach().cpu().numpy()
                ax.plot(range(T), values, color=colors[i], alpha=0.6,
                        label=labels[i] if n == 0 else None)
        ax.set_ylabel(f"{dim_names[d]} Position [mm]")
        ax.legend()
        ax.grid(True)
    axs[2].set_xlabel("Timestep")
    fig.suptitle("Position Components over Time (X, Y, Z)", fontsize=14)
    plt.tight_layout()
    plt.show()

    # RMSE Reporting
    buffer = io.StringIO()
    buffer.write("Root Mean Squared Error vs Ground Truth:\n")

    for i, tensor in enumerate(tensors[1:], start=1):  # Compare to tensor[0]
        buffer.write(f"\n{labels[i]}:\n")
        axis_rmses = []
        all_squared_errors = []

        for d, dim_name in enumerate(dim_names):
            gt = tensors[0][:, d, :].reshape(-1)
            pred = tensor[:, d, :].reshape(-1)

            squared_errors = (pred - gt) ** 2
            mse = torch.mean(squared_errors)
            rmse = torch.sqrt(mse).item()
            axis_rmses.append(rmse)

            all_squared_errors.append(squared_errors)

            buffer.write(f"  {dim_name} RMSE: {rmse:.6f}\n")

        # Compute total RMSE correctly
        all_squared_errors_cat = torch.cat(all_squared_errors)
        total_mse = torch.mean(all_squared_errors_cat)
        total_rmse = torch.sqrt(total_mse).item()
        buffer.write(f"  Total RMSE: {total_rmse:.6f}\n")

    rmse_report = buffer.getvalue()
    buffer.close()
    print(rmse_report)

    # Save report
    if path_results2 is not None:
        os.makedirs(path_results2, exist_ok=True)
        file_path = os.path.join(path_results2, "rmse_report.txt")
        with open(file_path, "w") as f:
            f.write(rmse_report)
        print(f"[Saved RMSE report to: {file_path}]")

    return rmse_report

def plot_xyz_trajectories_from_sources3(*tensors, labels=None, colors=None, path_results2=None):
    """
    Plot X, Y, Z components over time for any number of input tensors and compute axis-wise and total RMSE.
    Saves a report to 'path_results2/mse_report.txt' if path_results2 is provided.

    Each tensor should have shape [N, D, T], where D ≥ 3 (X, Y, Z in dims 0, 1, 2).

    Args:
        *tensors: 2 or more tensors of shape [N, D, T]
        labels: list of legend labels (optional)
        colors: list of line colors (optional)
        path_results2: path to save the RMSE report

    Returns:
        rmse_report (str): Formatted RMSE report text
    """
    assert len(tensors) >= 2, "Provide at least 2 tensors."
    assert all(t.shape[1] >= 3 for t in tensors), "Each tensor must have at least 3 features (X, Y, Z)."
    assert all(t.shape == tensors[0].shape for t in tensors), "All tensors must have the same shape."

    N, D, T = tensors[0].shape
    dim_names = ['X', 'Y', 'Z']
    loss_fn = nn.MSELoss(reduction='mean')

    # Auto-labeling if none provided
    if labels is None:
        frame = inspect.currentframe().f_back
        labels = [name for name, val in frame.f_locals.items() if val in tensors][:len(tensors)]
        while len(labels) < len(tensors):
            labels.append(f"Tensor {len(labels) + 1}")

    if colors is None:
        # Extend matplotlib color cycle if needed
        import itertools
        default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'brown', 'pink', 'gray']
        color_cycle = itertools.cycle(default_colors)
        colors = [next(color_cycle) for _ in range(len(tensors))]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for d in range(3):  # X, Y, Z
        ax = axs[d]
        for i, tensor in enumerate(tensors):
            for n in range(N):
                values = tensor[n, d, :].detach().cpu().numpy()
                ax.plot(range(T), values, color=colors[i], alpha=0.6,
                        label=labels[i] if n == 0 else None)
        ax.set_ylabel(f"{dim_names[d]} Position [mm]")
        ax.legend()
        ax.grid(True)
    axs[2].set_xlabel("Time Step")
    # fig.suptitle("Per-Axis End-Effector Position Tracking Over Timesteps", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Optional RMSE report (between first tensor and others)
    rmse_report = ""
    ref = tensors[0]
    for i in range(1, len(tensors)):
        comp = tensors[i]
        rmse_axes = []
        for d in range(3):
            mse = loss_fn(ref[:, d, :], comp[:, d, :]).item()
            rmse_axes.append(mse ** 0.5)
        total_rmse = torch.sqrt(loss_fn(ref[:, :3, :], comp[:, :3, :])).item() # avaraged over all sample (1/ NX3XT) - this is not euclidian 3D RMSE
        err = ref[:, :3, :] - comp[:, :3, :]
        sq_dist = (err ** 2).sum(dim=1)  # sum over x,y,z
        rmse_3d = torch.sqrt(sq_dist.mean()).item()
        total_rmse = rmse_3d # euclidian distance RMSE
        rmse_report += f"RMSE between {labels[0]} and {labels[i]}:\n"
        rmse_report += f"  X: {rmse_axes[0]:.4f}, Y: {rmse_axes[1]:.4f}, Z: {rmse_axes[2]:.4f}, Total: {total_rmse:.4f}\n"

    print(rmse_report)
    # Save RMSE report if path provided
    if path_results2 is not None:
        os.makedirs(path_results2, exist_ok=True)
        with open(os.path.join(path_results2, "rmse_report.txt"), "w") as f:
            f.write(rmse_report)

    return rmse_report

def plot_xyz_trajectories_from_sources4(
    *tensors,
    labels=None,
    colors=None,
    path_results2=None,
    y_span_mm=None,   # NEW: desired equal y-range (in mm) for all subplots; if None, use max data span
    clip=False        # NEW: if True, enforce y_span_mm strictly (may clip). If False, expand to fit data.
):
    """
    Plot X, Y, Z components over time for any number of input tensors and compute axis-wise and total RMSE.
    Optionally enforces the same y-axis span across X/Y/Z subplots (e.g., 100 mm).
    Saves a report to 'path_results2/rmse_report.txt' if path_results2 is provided.

    Each tensor should have shape [N, D, T], where D ≥ 3 (X, Y, Z in dims 0, 1, 2).

    Args:
        *tensors: 2 or more tensors of shape [N, D, T]
        labels: list of legend labels (optional)
        colors: list of line colors (optional)
        path_results2: path to save the RMSE report
        y_span_mm: float or None. If provided, sets the same y-range for all subplots.
        clip: bool. If False and y_span_mm is too small to contain the data, it will be increased.

    Returns:
        rmse_report (str): Formatted RMSE report text
    """
    assert len(tensors) >= 2, "Provide at least 2 tensors."
    assert all(t.shape[1] >= 3 for t in tensors), "Each tensor must have at least 3 features (X, Y, Z)."
    assert all(t.shape == tensors[0].shape for t in tensors), "All tensors must have the same shape."

    N, D, T = tensors[0].shape
    dim_names = ['X', 'Y', 'Z']
    loss_fn = nn.MSELoss(reduction='mean')

    # Auto-labeling if none provided
    if labels is None:
        frame = inspect.currentframe().f_back
        labels = [name for name, val in frame.f_locals.items() if val in tensors][:len(tensors)]
        while len(labels) < len(tensors):
            labels.append(f"Tensor {len(labels) + 1}")

    if colors is None:
        import itertools
        default_colors = ['blue', 'red', 'darkgreen', 'purple', 'orange', 'black', 'brown', 'pink', 'gray']
        color_cycle = itertools.cycle(default_colors)
        colors = [next(color_cycle) for _ in range(len(tensors))]
        markers = ['o','s', '^', 'd']

    # Compute per-dimension min/max across all tensors and samples
    # Shape unification: stack tensors along a new leading axis for easy global min/max
    stacked = torch.stack(tensors, dim=0)  # [K, N, D, T]
    mins = stacked[:, :, :3, :].amin(dim=(0,1,3))  # [3]
    maxs = stacked[:, :, :3, :].amax(dim=(0,1,3))  # [3]

    # Decide a common y-span for all three subplots
    data_spans = (maxs - mins).detach().cpu().numpy()  # [3]
    if y_span_mm is None:
        common_span = float(max(data_spans))  # largest data span across X/Y/Z
        if common_span == 0.0:
            common_span = 1.0  # avoid zero-span edge case
    else:
        common_span = float(y_span_mm)
        # If not clipping, ensure common_span is at least as large as the largest data span
        if not clip:
            common_span = max(common_span, float(max(data_spans)))

    # Centers per dimension so plots are nicely centered on the data
    centers = ((mins + maxs) / 2.0).detach().cpu().numpy()  # [3]
    y_lims = [(c - common_span/2.0 -4, c + common_span/2.0 + 4) for c in centers]  # list of 3 (low, high)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for d in range(3):  # X, Y, Z
        ax = axs[d]
        for i, tensor in enumerate(tensors):
            for n in range(N):
                values = tensor[n, d, :].detach().cpu().numpy()
                ax.plot(range(T), values, color=colors[i],marker=markers[i],markersize=4,markevery=150,
                        label=labels[i] if n == 0 else None)
        ax.set_ylabel(f"{dim_names[d]} Position [mm]")
        ax.set_ylim(*y_lims[d])  # <-- enforce SAME span for all three axes
        axs[0].legend(loc='upper right',fontsize=14)
        order = (0, 3, 1, 2)
        axs[0].legend(*[[seq[i] for i in order] for seq in ax.get_legend_handles_labels()], loc='upper right', fontsize=14)
        ax.set_xlim(0, T-1)
        ax.grid(True)
        
        
        # --- ADD MAGNIFIED INSET ---
        # One box per subplot (X, Y, Z). Use None to skip a box for that axis.
        # data for traj 12000L13000
        # zoom_boxes = [
        #     {"x": (260, 280), "y": (-72, -68)},   # for X subplot
        #     {"x": (230, 250), "y": (2, 10)},    # for Y subplot
        #     {"x": (0, 25), "y": (-84, -80)},    # for Z subplot
        # ]
        zoom_boxes = [
            {"x": (32, 55), "y": (-62, -57)},   # for X subplot
            {"x": (230, 250), "y": (2, 10)},    # for Y subplot
            {"x": (0, 25), "y": (-85, -81)},    # for Z subplot
        ]
        # data for traj 8000:9000
        # zoom_boxes = [
        #     {"x": (400, 430), "y": (-72, -68)},   # for X subplot
        #     {"x": (100, 120), "y": (15, 20)},    # for Y subplot
        #     {"x": (0, 25), "y": (-93, -89)},    # for Z subplot
        # ]
        # Optional: different inset positions per subplot
        inset_locs = ['center left', 'upper center', 'upper left']
        # Create inset axis
        box = zoom_boxes[d]
        if box is not None:
            axins = inset_axes(ax, width="16%", height="25%", loc=inset_locs[d], borderpad=0.5)
            for i, tensor in enumerate(tensors):
                for n in range(N):
                    vals = tensor[n, d, :].detach().cpu().numpy()
                    axins.plot(range(T), vals, color=colors[i], marker=markers[i],markevery=100, alpha=0.6)
    
            # Apply per-subplot limits
            axins.set_xlim(*box["x"])
            axins.set_ylim(*box["y"])
            axins.set_xticks([]); axins.set_yticks([])
    
            # Draw rectangle on the parent axes + connecting lines
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1.4)

    axs[2].set_xlabel("Time Step")
    # fig.suptitle("Per-Axis End-Effector Position Tracking Over Timesteps", fontsize=14)
    plt.rcParams.update({
    "axes.labelsize": 16,   # axis labels
    "xtick.labelsize": 14,  # x-axis numbers
    "ytick.labelsize": 14,  # y-axis numbers
    })
    plt.tight_layout()
    plt.show()

    # Optional RMSE report (between first tensor and others)
    rmse_report = ""
    ref = tensors[0]
    for i in range(1, len(tensors)):
        comp = tensors[i]
        rmse_axes = []
        for d in range(3):
            mse = loss_fn(ref[:, d, :], comp[:, d, :]).item()
            rmse_axes.append(mse ** 0.5)
        err = ref[:, :3, :] - comp[:, :3, :]
        sq_dist = (err ** 2).sum(dim=1)  # sum over x,y,z
        rmse_3d = torch.sqrt(sq_dist.mean()).item()  # Euclidean 3D RMSE
        total_rmse = rmse_3d
        rmse_report += f"RMSE between {labels[0]} and {labels[i]}:\n"
        rmse_report += f"  X: {rmse_axes[0]:.4f}, Y: {rmse_axes[1]:.4f}, Z: {rmse_axes[2]:.4f}, Total: {total_rmse:.4f}\n"

    print(rmse_report)
    # Save RMSE report if path provided
    if path_results2 is not None:
        os.makedirs(path_results2, exist_ok=True)
        with open(os.path.join(path_results2, "rmse_report.txt"), "w") as f:
            f.write(rmse_report)

    return rmse_report

def plot_xyz_trajectories_colored_by_error(
    *tensors,
    labels=None,
    cmap_name='jet',
    y_span_mm=None,   # desired equal y-range (in mm) for all subplots; if None, use max data span
    clip=False        # if True, enforce y_span_mm strictly. If False, expand to fit data.
):
    """
    Plots X, Y, Z components over time for tracking algorithms. 
    The FIRST tensor passed in *tensors is treated as the Ground Truth (GT) and is ONLY used 
    to calculate absolute tracking errors. It is NOT plotted on the axes.
    """
    assert len(tensors) >= 2, "Provide at least 2 tensors (The first must be Ground Truth)."
    assert all(t.shape[1] >= 3 for t in tensors), "Each tensor must have at least 3 features (X, Y, Z)."
    assert all(t.shape == tensors[0].shape for t in tensors), "All tensors must have the same shape."

    # Extract dimensions
    N, D, T = tensors[0].shape
    dim_names = ['X', 'Y', 'Z']
    time_steps = np.arange(T)

    # Separate Ground Truth from evaluated algorithms
    gt_tensor = tensors[0]
    algo_tensors = tensors[1:]

    # Auto-labeling if none provided
    if labels is None:
        frame = inspect.currentframe().f_back
        labels = [name for name, val in frame.f_locals.items() if val in tensors][:len(tensors)]
        while len(labels) < len(tensors):
            labels.append(f"Tensor {len(labels) + 1}")
    
    # Isolate algorithm labels safely
    algo_labels = labels[1:] if len(labels) == len(tensors) else labels[:len(algo_tensors)]

    # --- FIX: Explicitly map specific marker styles to exact labels ---
    # This prevents any loop iteration or tensor ordering mismatches.
    marker_mapping = {
        'Algorithm 1': 'd',  # Diamond
        'Algorithm 2': 's',  # Square
        'Raven II': '^'      # Triangle
    }

    # Compute per-dimension min/max across evaluated algorithms (excluding GT) for robust layout scaling
    stacked_algos = torch.stack(algo_tensors, dim=0)  
    mins = stacked_algos[:, :, :3, :].amin(dim=(0, 1, 3))  
    maxs = stacked_algos[:, :, :3, :].amax(dim=(0, 1, 3))  

    # Decide a common y-span for all three subplots
    data_spans = (maxs - mins).detach().cpu().numpy()  
    if y_span_mm is None:
        common_span = float(max(data_spans))
        if common_span == 0.0:
            common_span = 1.0
    else:
        common_span = float(y_span_mm)
        if not clip:
            common_span = max(common_span, float(max(data_spans)))

    # Centers per dimension so plots stay aligned
    centers = ((mins + maxs) / 2.0).detach().cpu().numpy()  
    y_lims = [(c - common_span/2.0 - 4, c + common_span/2.0 + 4) for c in centers]

    # Global font configurations matching original layout style
    plt.rcParams.update({
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    # Plotting Initialization
    fig, axs = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    for d in range(3):  # Loop through X, Y, Z components
        ax = axs[d]
        dim_key = dim_names[d]
        
        # Calculate absolute errors across ALL tracking algorithms instantly
        axis_errors = torch.abs(stacked_algos[:, :, d, :] - gt_tensor[:, d, :]).detach().cpu().numpy()
        
        err_min = axis_errors.min()
        err_max = axis_errors.max()

        # Create ONE shared LINEAR normalization scale for this axis subplot
        norm = plt.Normalize(vmin=err_min, vmax=err_max)
        last_lc = None  

        # 2. Plot the Tracking Algorithms (Dynamic error color lines)
        for i, tensor in enumerate(algo_tensors):
            current_label = algo_labels[i]
            
            # Fallback to diamond if user string doesn't explicitly match dictionary keys
            chosen_marker = marker_mapping.get(current_label, 'd')

            for n in range(N):
                traj_values = tensor[n, d, :].detach().cpu().numpy()
                gt_values = gt_tensor[n, d, :].detach().cpu().numpy()
                abs_error = np.abs(traj_values - gt_values)

                # Format spatial data coordinates into discrete path segments
                points = np.array([time_steps, traj_values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create LineCollection and place on top layer (zorder=3) so color stays fully visible
                lc = LineCollection(segments, cmap=cmap_name, norm=norm, alpha=0.85, zorder=3)
                
                error_segments = (abs_error[:-1] + abs_error[1:]) / 2.0
                lc.set_array(error_segments)
                lc.set_linewidth(3)
                
                ax.add_collection(lc)
                last_lc = lc  

                # Overlay reference markers (White filled, black outlines, placed under lines)
                mark_indices = np.arange(0, T, 100)
                ax.plot(time_steps[mark_indices], traj_values[mark_indices], 
                        linestyle='None', marker=chosen_marker, markersize=10, 
                        markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.2, 
                        alpha=0.7, zorder=2)

        # --- Render exactly ONE linear colorbar per subplot ---
        if last_lc is not None:
            cbar = fig.colorbar(last_lc, ax=ax, orientation='vertical', pad=0.02, fraction=0.02)
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(f"{dim_key} Absolute Error [mm]", fontsize=11)

        # Formatting configurations per subplot row
        ax.set_ylabel(f"{dim_names[d]} Position [mm]",fontsize=18)
        ax.set_ylim(*y_lims[d])
        ax.set_xlim(0, T - 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # --- Only render the legend block for the FIRST subplot (d == 0) ---
        if d == 0:
            hl_a1 = ax.plot([], [], color='gray', marker=marker_mapping['Algorithm 1'], markersize=12, 
                            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, linewidth=2, linestyle='-')[0]
            
            hl_a2 = ax.plot([], [], color='gray', marker=marker_mapping['Algorithm 2'], markersize=12, 
                            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, linewidth=2, linestyle='-')[0]
            
            hl_rv = ax.plot([], [], color='gray', marker=marker_mapping['Raven II'], markersize=12, 
                            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5, linewidth=2, linestyle='-')[0]
            
            desired_labels = ['Algorithm 1', 'Algorithm 2', 'Raven II']
            ax.legend([hl_a1, hl_a2, hl_rv], desired_labels, loc='upper right', fontsize=18, prop={'size': 18})

    axs[2].set_xlabel("Time Step",fontsize=18)
    plt.tight_layout()
    plt.show()

def center_trajectories_by_mean(GT, raven_data): # (GT, desired_pos, raven_data)
    """
    Center trajectories by subtracting the mean of each dimension per trajectory.

    Parameters:
    - GT: Tensor of shape [N, D, T] (ground truth trajectories)
    # - desired_pos: Tensor of shape [N, D, T] (desired position data)
    - raven_data: Tensor of shape [N, D, T] (measured data from method 2)

    Returns:
    - centered_GT: GT with each trajectory reduced by its own mean (per dimension)
    # - centered_desired_pos: desired_pos with each trajectory reduced by GT mean (per dimension)
    - centered_raven_data: raven_data with each trajectory reduced by its own mean (per dimension)
    """

    # Compute per-trajectory mean across time for GT and raven_data: shape [N, D, 1]
    mean_GT = GT.mean(dim=2, keepdim=True)
    mean_raven = raven_data.mean(dim=2, keepdim=True)

    # Subtract the mean from each trajectory (broadcasted over time)
    centered_GT = GT - mean_GT
    # centered_desired_pos = desired_pos - mean_GT  # use GT mean for desired positions
    centered_raven_data = raven_data - mean_raven

    return centered_GT, centered_raven_data #centered_GT, centered_desired_pos, centered_raven_data



def plot_sensor_differences_all_trajectories(sensor1: torch.Tensor, sensor2: torch.Tensor, title="Sensor Difference Over Time"):
    """
    Plots the per-trajectory difference between two sensor measurements over time for each axis.

    Parameters:
        sensor1 (torch.Tensor): shape (N, 3, T)
        sensor2 (torch.Tensor): shape (N, 3, T)
        title (str): Title of the figure
    """
    assert sensor1.shape == sensor2.shape, "Sensors must have the same shape (N, 3, T)"
    assert sensor1.ndim == 3 and sensor1.shape[1] == 3, "Expected input shape (N, 3, T)"

    diff = (sensor1 - sensor2).detach().cpu().numpy()  # shape (N, 3, T)
    N, _, T = diff.shape

    axis_labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    time = np.arange(T)

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title)

    for axis in range(3):
        for n in range(N):
            axs[axis].plot(time, diff[n, axis], alpha=0.6)
        axs[axis].set_ylabel(f"Δ{axis_labels[axis]} (units)")
        axs[axis].grid(True)

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def plot_predictions_vs_ground_truth_diff_colors_withraven_axesequal_2(test_target, model_output, raven_data, ekf_output, labels):
    """
    Plots 3D trajectories of test_target (ground truth), model_output (predictions),
    raven_data, and ekf_output. Sets axes to equal scale.

    Args:
        test_target: Ground truth tensor [N, 3, T]
        model_output: Model predictions [N, 3, T]
        raven_data: Raw observations [N, 3, T]
        ekf_output: EKF estimations [N, 3, T]
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Number of trajectories

    # Collect all points for equal axis scaling
    all_x, all_y, all_z = [], [], []

    for i in range(N):
        x_true, y_true, z_true = test_target[i, 0, :].detach().cpu().numpy(), \
                                 test_target[i, 1, :].detach().cpu().numpy(), \
                                 test_target[i, 2, :].detach().cpu().numpy()
        x_pred, y_pred, z_pred = model_output[i, 0, :].detach().cpu().numpy(), \
                                 model_output[i, 1, :].detach().cpu().numpy(), \
                                 model_output[i, 2, :].detach().cpu().numpy()
        x_r, y_r, z_r = raven_data[i, 0, :].detach().cpu().numpy(), \
                        raven_data[i, 1, :].detach().cpu().numpy(), \
                        raven_data[i, 2, :].detach().cpu().numpy()
        x_ekf, y_ekf, z_ekf = ekf_output[i, 0, :].detach().cpu().numpy(), \
                              ekf_output[i, 1, :].detach().cpu().numpy(), \
                              ekf_output[i, 2, :].detach().cpu().numpy()

        # Plot lines
        colors = ['blue', 'red', 'darkgreen', 'purple']
        markers = ['o','s', '^', 'd']
        ax.plot(x_true, y_true, z_true, color=colors[0], alpha=0.8, label=labels[0] if i == 0 else None)
        ax.plot(x_pred, y_pred, z_pred, color=colors[1], alpha=0.8, label=labels[1] if i == 0 else None)
        ax.plot(x_r, y_r, z_r, color=colors[2], alpha=0.8, label=labels[2] if i == 0 else None)
        ax.plot(x_ekf, y_ekf, z_ekf, color=colors[3], alpha=0.8, label=labels[3] if i == 0 else None)

        # Accumulate points for scaling
        all_x.extend(np.concatenate([x_true, x_pred, x_r, x_ekf]))
        all_y.extend(np.concatenate([y_true, y_pred, y_r, y_ekf]))
        all_z.extend(np.concatenate([z_true, z_pred, z_r, z_ekf]))

    # Set equal axis scaling
    x_range = np.array(all_x)
    y_range = np.array(all_y)
    z_range = np.array(all_z)
    max_range = np.array([x_range.max() - x_range.min(),
                          y_range.max() - y_range.min(),
                          z_range.max() - z_range.min()]).max() / 2.0

    mid_x = (x_range.max() + x_range.min()) * 0.5
    mid_y = (y_range.max() + y_range.min()) * 0.5
    mid_z = (z_range.max() + z_range.min()) * 0.5

    ax.set_xlim(mid_x - max_range , mid_x + max_range )
    ax.set_ylim(mid_y - max_range , mid_y + max_range )
    ax.set_zlim(mid_z - max_range , mid_z + max_range )

    # Labels and legend
    ax.set_xlabel("X Position [mm]",labelpad=15)
    ax.set_ylabel("Y Position [mm]",labelpad=15)
    ax.set_zlabel("Z Position [mm]",labelpad=15)
    # ax.set_title("3D End-Effector Trajectory Tracking: Comparison of Methods")
    ax.legend(fontsize=14)
    order = (0, 3, 1, 2)
    ax.legend(*[[seq[i] for i in order] for seq in ax.get_legend_handles_labels()],fontsize=14)
    plt.tight_layout()
    plt.show()

def integrate_position_from_gt(gt_position: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """
    Computes x,y,z positions over time from initial GT position and per-timestep distances.
    
    Args:
        gt_position (torch.Tensor): Tensor of shape (1, 6, T) — first 3 channels are x, y, z.
        distance (torch.Tensor): Tensor of shape (1, 3, T) — distance at each timestep for x, y, z.
        
    Returns:
        torch.Tensor: Integrated position tensor of shape (1, 3, T+1)
    """
    # Extract the initial position at t=0 (x, y, z)
    initial_pos = gt_position[:, :3, 0]  # shape: (1, 3)

    # Cumulative sum of distances over time (integration)
    cumsum_dist = torch.cumsum(distance, dim=2)  # shape: (1, 3, T)

    # Add initial position to each timestep
    positions = initial_pos.unsqueeze(2) + cumsum_dist  # shape: (1, 3, T)

    # Prepend initial position at t=0
    full_positions = torch.cat([initial_pos.unsqueeze(2), positions], dim=2)  # shape: (1, 3, T+1)

    return full_positions

def plot_predictions_vs_ground_truth_with_integrated(
    test_target, model_output, raven_data, ekf_output, integrated_position
):
    """
    Plots 3D trajectories of test_target (ground truth), model_output (predictions),
    raven_data, ekf_output, and integrated_position. Sets axes to equal scale.

    Args:
        test_target: Ground truth tensor [N, 3, T]
        model_output: Model predictions [N, 3, T]
        raven_data: Raw observations [N, 3, T]
        ekf_output: EKF estimations [N, 3, T]
        integrated_position: Integrated position from GT + distance [N, 3, T+1]
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    N = test_target.shape[0]  # Batch size (number of sequences)

    all_x, all_y, all_z = [], [], []

    for i in range(N):
        x_true, y_true, z_true = test_target[i, 0, :].detach().cpu().numpy(), \
                                 test_target[i, 1, :].detach().cpu().numpy(), \
                                 test_target[i, 2, :].detach().cpu().numpy()
        x_pred, y_pred, z_pred = model_output[i, 0, :].detach().cpu().numpy(), \
                                 model_output[i, 1, :].detach().cpu().numpy(), \
                                 model_output[i, 2, :].detach().cpu().numpy()
        x_r, y_r, z_r = raven_data[i, 0, :].detach().cpu().numpy(), \
                        raven_data[i, 1, :].detach().cpu().numpy(), \
                        raven_data[i, 2, :].detach().cpu().numpy()
        x_ekf, y_ekf, z_ekf = ekf_output[i, 0, :].detach().cpu().numpy(), \
                              ekf_output[i, 1, :].detach().cpu().numpy(), \
                              ekf_output[i, 2, :].detach().cpu().numpy()
        x_int, y_int, z_int = integrated_position[i, 0, :].detach().cpu().numpy(), \
                              integrated_position[i, 1, :].detach().cpu().numpy(), \
                              integrated_position[i, 2, :].detach().cpu().numpy()

        # Plot lines
        ax.plot(x_true, y_true, z_true, linestyle='-', color='blue', alpha=0.8, label="Ground Truth" if i == 0 else None)
        ax.plot(x_pred, y_pred, z_pred, linestyle='--', color='red', alpha=0.8, label="KalmanNet" if i == 0 else None)
        ax.plot(x_r, y_r, z_r, linestyle='dashdot', color='green', alpha=0.8, label="Raven II" if i == 0 else None)
        ax.plot(x_ekf, y_ekf, z_ekf, linestyle=':', color='purple', alpha=0.8, label="Learned EKF" if i == 0 else None)
        ax.plot(x_int, y_int, z_int, linestyle='-', color='orange', alpha=0.8, label="State Evolution" if i == 0 else None)

        # Accumulate points for scaling
        all_x.extend(np.concatenate([x_true, x_pred, x_r, x_ekf, x_int]))
        all_y.extend(np.concatenate([y_true, y_pred, y_r, y_ekf, y_int]))
        all_z.extend(np.concatenate([z_true, z_pred, z_r, z_ekf, z_int]))

    # Set equal axis scaling
    x_range = np.array(all_x)
    y_range = np.array(all_y)
    z_range = np.array(all_z)
    max_range = np.array([
        x_range.max() - x_range.min(),
        y_range.max() - y_range.min(),
        z_range.max() - z_range.min()
    ]).max() / 2.0

    mid_x = (x_range.max() + x_range.min()) * 0.5
    mid_y = (y_range.max() + y_range.min()) * 0.5
    mid_z = (z_range.max() + z_range.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Trajectories: GT vs Predictions vs EKF vs Integrated")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def integrate_position_from_gt_shifted(gt_position: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """
    Integrates position from GT initial position plus distance at t=0, using p(t+1) = p(t) + distance(t).
    
    Args:
        gt_position (torch.Tensor): Tensor of shape (1, 6, T), first 3 dims are x, y, z.
        distance (torch.Tensor): Tensor of shape (1, 3, T)
        
    Returns:
        torch.Tensor: Integrated positions of shape (1, 3, T)
    """
    B, C, T = distance.shape  # (1, 3, 1000)
    
    # Initialize result tensor
    positions = torch.zeros((B, C, T), device=distance.device, dtype=distance.dtype)
    
    # Set position at t=0 as gt_position[:, :3, 0] + distance[:, :, 0]
    positions[:, :, 0] = gt_position[:, :3, 0] + distance[:, :, 0]
    
    # Iterate over time to compute position
    for t in range(1, T):
        positions[:, :, t] = positions[:, :, t-1] + distance[:, :, t]
    
    return positions

def compute_rmse_stats_and_save(directory_path): # this function was checked
    rmse_pattern = re.compile(
        r"RMSE between Ground Truth \(Markers\) and (\w[\w ]*\w):\s+X:\s+([\d.]+),\s+Y:\s+([\d.]+),\s+Z:\s+([\d.]+),\s+Total:\s+([\d.]+)"
    )

    rmse_data = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                content = file.read()
                for match in rmse_pattern.finditer(content):
                    method = match.group(1).strip()
                    values = list(map(float, match.groups()[1:]))

                    if method not in rmse_data:
                        rmse_data[method] = []

                    rmse_data[method].append(values)

    stats = {}
    for method, values in rmse_data.items():
        arr = np.array(values)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        stats[method] = {
            'mean': {'X': mean[0], 'Y': mean[1], 'Z': mean[2], 'Total': mean[3]},
            'std': {'X': std[0], 'Y': std[1], 'Z': std[2], 'Total': std[3]}
        }

    # Create filename with current datetime
    now = datetime.now().strftime("%d.%m.%Y__%H_%M_%S")
    output_filename = f"mean and std results_{now}.txt"
    output_path = os.path.join(directory_path, output_filename)

    # Save to file
    with open(output_path, 'w') as out_file:
        for method, data in stats.items():
            out_file.write(f"Method: {method}\n")
            out_file.write("  Mean RMSE:\n")
            for axis, val in data['mean'].items():
                out_file.write(f"    {axis}: {val:.4f}\n")
            out_file.write("  Std RMSE:\n")
            for axis, val in data['std'].items():
                out_file.write(f"    {axis}: {val:.4f}\n")
            out_file.write("\n")

    print(f"Results saved to {output_path}")
    return output_path

def plot_3d_valid_vs_train(train_traj: torch.Tensor, val_traj: torch.Tensor):
    """
    Plots 3D trajectories for training and validation sets.

    Parameters:
    - train_traj: torch.Tensor of shape (N_train, 3, t)
    - val_traj: torch.Tensor of shape (N_val, 3, t)
    """
    assert train_traj.ndim == 3 and train_traj.shape[1] == 3
    assert val_traj.ndim == 3 and val_traj.shape[1] == 3
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0

    # Plot training trajectories in red
    for traj in train_traj:
        x, y, z = traj[0], traj[1], traj[2]
        if i == 0:
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), color='red', alpha=0.6, label='Training set')
            i = 1
        else:
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), color='red', alpha=0.6)
            
    # Plot validation trajectories in blue
    i = 0
    for traj in val_traj:
        x, y, z = traj[0], traj[1], traj[2]
        if i == 0:
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), color='blue', alpha=0.6, label='Validation set')
            i = 1
        else:
            ax.plot(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), color='blue', alpha=0.6)
    #ax.set_title("3D Trajectories")
    ax.set_xlabel("X [mm]",labelpad=15)
    ax.set_ylabel("Y [mm]",labelpad=15)
    ax.set_zlabel("Z [mm]",labelpad=15)

    # Make axes equal
    all_points = torch.cat([train_traj, val_traj], dim=0)  # (N_total, 3, t)
    all_points = all_points.permute(0, 2, 1).reshape(-1, 3).cpu().numpy()  # (N_total * t, 3)
    x_min, x_max = np.min(all_points[:,0]), np.max(all_points[:,0])
    y_min, y_max = np.min(all_points[:,1]), np.max(all_points[:,1])
    z_min, z_max = np.min(all_points[:,2]), np.max(all_points[:,2])
    
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(fontsize=14)
    plt.show()
    
    
def save_execution_times(time1, time2, save_dir):
    """
    Saves two execution time values to 'execution_time.txt' in the specified directory.

    Parameters:
    - time1: Execution time of the first method (in seconds)
    - time2: Execution time of the second method (in seconds)
    - save_dir: Directory path where the file will be saved
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Full path to the file
    file_path = os.path.join(save_dir, "execution_time.txt")

    # Write execution times to the file
    with open(file_path, "w") as f:
        f.write(f"Execution time Knet: {time1:.6f} seconds\n")
        f.write(f"Execution time Learned EKF: {time2:.6f} seconds\n")
        
def compute_mean_std_execution_times(folder_path):
    """
    Reads all .txt files in the folder, identifies those with the correct structure,
    computes mean and std for 'Knet' and 'Learned EKF' methods, and saves results.

    Parameters:
    - folder_path: Path to the folder containing text files.
    """
    knet_times = []
    learned_ekf_times = []

    # Search for all .txt files in folder (and subfolders)
    file_list = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)

    if not file_list:
        print("No .txt files found.")
        return

    for file_path in file_list:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        # Check if file matches expected structure
        if len(lines) >= 2 and \
           lines[0].startswith("Execution time Knet:") and \
           lines[1].startswith("Execution time Learned EKF:"):
            try:
                knet_time = float(lines[0].split(":")[1].split()[0])
                learned_ekf_time = float(lines[1].split(":")[1].split()[0])

                knet_times.append(knet_time)
                learned_ekf_times.append(learned_ekf_time)
            except (ValueError, IndexError) as e:
                print(f"Skipping file {file_path} due to parsing error: {e}")
        else:
            # Not matching structure — skip silently or print a message
            pass

    if not knet_times:
        print("No matching structured files found.")
        return

    knet_times = np.array(knet_times)
    learned_ekf_times = np.array(learned_ekf_times)

    knet_mean, knet_std = np.mean(knet_times), np.std(knet_times)
    learned_mean, learned_std = np.mean(learned_ekf_times), np.std(learned_ekf_times)

    # Include date and time in output filename
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(folder_path, f"mean_and_std_execution_time_{datetime_str}.txt")

    with open(output_file, "w") as f:
        f.write(f"Knet - Mean: {knet_mean:.6f} sec, Std: {knet_std:.6f} sec\n")
        f.write(f"Learned EKF - Mean: {learned_mean:.6f} sec, Std: {learned_std:.6f} sec\n")

    print(f"Results saved to: {output_file}")