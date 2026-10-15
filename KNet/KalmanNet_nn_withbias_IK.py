
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import project_to_box
from Simulations.parameters_withbias_trainIK import x_limits, y_limits, z_limits

"""class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super(InverseKinematicsNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x"""
class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super(InverseKinematicsNN, self).__init__() # inherit from torch.nn.module
        # Define fully connected layers
        # self.norm = nn.LayerNorm(3) # delte
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)
        
        # Optional: Define activation functions as modules
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply activation functions between layers
        # x = self.norm(x) # delete
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Final layer without activation (assume a regression task)
        return x

class KalmanNetNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.h_nn = InverseKinematicsNN().to(self.device)
        # self.best_h_nn = InverseKinematicsNN()
        self.best_loss = float('inf')

    def NNBuild(self, SysModel, args):
        self.device = torch.device('cuda' if args.use_cuda else 'cpu')
        self.h_nn = InverseKinematicsNN().to(self.device)
        self.SysModel_h = SysModel.h
        self.InitSystemDynamics(SysModel.f, self.h_nn, SysModel.m, SysModel.n)
        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, args)
        self.args = args

    def InitSystemDynamics(self, f, h_nn, m, n):
        self.f = f
        self.h_nn = h_nn
        self.m = m
        self.n = n

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, args):
        self.seq_len_input = 1
        self.batch_size = args.n_batch
        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)

        self.d_input_Q = self.m * args.in_mult_KNet
        self.d_hidden_Q = self.m ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        self.d_input_Sigma = self.d_hidden_Q + self.m * args.in_mult_KNet
        self.d_hidden_Sigma = self.m ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        self.d_input_S = self.n ** 2 + 2 * self.n * args.in_mult_KNet
        self.d_hidden_S = self.n ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU()
        ).to(self.device)

        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * args.out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2)
        ).to(self.device)

        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU()
        ).to(self.device)

        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU()
        ).to(self.device)

        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * args.in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU()
        ).to(self.device)

        self.d_input_FC6 = self.m
        self.d_output_FC6 = self.m * args.in_mult_KNet
        self.FC6 = nn.Sequential(
            nn.Linear(self.d_input_FC6, self.d_output_FC6),
            nn.ReLU()
        ).to(self.device)

        self.d_input_FC7 = 2 * self.n
        self.d_output_FC7 = 2 * self.n * args.in_mult_KNet
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU()
        ).to(self.device)

    def InitSequence(self, M1_0, T):
        self.T = T
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m1x_posterior = M1_0.squeeze(-1).to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior

        # Separate position and bias for initialization
        self.position_prior_sq = self.m1x_posterior[:, :3].squeeze(-1).to(self.device)  # First 3 dimensions (position)
        self.bias_prior_sq = self.m1x_posterior[:, 3:].squeeze(-1).to(self.device)  # Last 3 dimensions (bias)

        # Use only the position in h_nn
        self.y_previous = self.h_nn(self.position_prior_sq) + self.bias_prior_sq

    def step_prior(self,des):
        self.position_prior = self.f(self.m1x_posterior[:, :3], des )  # First 3 dimensions (position)
        self.bias_prior = self.m1x_posterior[:, 3:]  # Last 3 dimensions (bias)
        self.m1x_prior = torch.cat((self.position_prior,self.bias_prior),dim =1)
        self.position_prior_sq = self.position_prior.squeeze(-1)  # Ensure it's in the right shape
        self.bias_prior_sq = self.bias_prior.squeeze(-1)

        # Forward pass through the neural network for position and add bias
        self.m1y_nn = self.h_nn(self.position_prior_sq) + self.bias_prior_sq

    def KNet_step(self, y, des):
        self.step_prior(des)
        dy_nn = y - self.m1y_nn

        # Update Kalman gain and perform state estimation
        self.KGain_step(y)
        # self.KGain_step(dy_nn)
        INOV_nn = torch.bmm(self.KGain_nn, dy_nn.unsqueeze(-1)).squeeze(-1)
        self.m1x_posterior_previous = self.m1x_posterior
        # Update the posterior state estimate
        self.m1x_posterior_nn = torch.cat([self.position_prior_sq, self.bias_prior_sq], dim=1) + INOV_nn

        # Concatenate the bias back to the updated position
        self.m1x_posterior = project_to_box(self.m1x_posterior_nn,x_limits, y_limits, z_limits)
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        if self.args.CompositionLoss2:
            return self.m1x_posterior, self.m1y_nn, self.bias_prior_sq
        else:   
            return self.m1x_posterior

    def KGain_step(self, y):
    # Assuming dy_nn contains the position differences (first 3 dimensions)
        # obs_diff = dy_nn
        obs_diff = y - self.y_previous 
        
        obs_innov_diff =  y - self.m1y_nn
            
        fw_evol_diff = self.m1x_posterior - self.m1x_posterior_previous
       
        
        # fw_update_diff = self.position_prior - self.m1x_prior_previous[:, :3]
        fw_update_diff = self.m1x_posterior - self.m1x_prior_previous
 
        # normalization
        
        obs_diff = F.normalize(obs_diff, p=2, dim=1, eps=1e-12, out=None)
        obs_innov_diff = F.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_evol_diff = F.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12, out=None)
        fw_update_diff = F.normalize(fw_update_diff, p=2, dim=1, eps=1e-12, out=None)

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff)
        obs_innov_diff = expand_dim(obs_innov_diff)
        fw_evol_diff = expand_dim(fw_evol_diff)
        fw_update_diff = expand_dim(fw_update_diff)

        out_FC5 = self.FC5(fw_evol_diff)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)

        out_FC6 = self.FC6(fw_update_diff)
        in_Sigma = torch.cat((out_Q, out_FC6), dim=2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        out_FC1 = self.FC1(out_Sigma)
    
    # Now pass the correctly shaped obs_diff through FC7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), dim=2)
        out_FC7 = self.FC7(in_FC7)  # obs_diff should now have the correct dimensions

        in_S = torch.cat((out_FC1, out_FC7), dim=2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)

        in_FC2 = torch.cat((out_Sigma, out_S), dim=2)
        out_FC2 = self.FC2(in_FC2)
        
        #####################
        ### Backward Flow ###
        #####################


                # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4
        
        #self.KGain_nn = out_FC2.view(self.batch_size, self.m, self.n)
        self.KGain_nn = torch.reshape(out_FC2, (self.batch_size, self.m, self.n))
        
    def make_state_valid(self, state, epsilon=0.01, cone_axis=None, cone_angle=None):
        if cone_axis is None:
            cone_axis = torch.tensor([1.0, 0.0, 0.0], device=state.device, dtype=state.dtype, requires_grad=False)
        if cone_angle is None:
            cone_angle = torch.tensor(30.0, device=state.device, dtype=state.dtype, requires_grad=False) * torch.pi / 180.0

        new_state = state.clone()
        for i in range(state.shape[0]):
            x = state[i, 0]
            y = state[i, 1]
            z = state[i, 2]
            b1 = state[i, 3]
            b2 = state[i, 4]
            b3 = state[i, 5]
            x = torch.clamp(x, -0.8, 0.8)
            y = torch.clamp(y, -0.25, 0.25)
            z = torch.clamp(z, -0.2, 0.55)
            b1 = torch.clamp(b1, -0.5, 0.5)
            b2 = torch.clamp(b2, -0.5, 0.5)
            b3 = torch.clamp(b3, -0.5, 0.5)
            # position_vector = torch.tensor([x, y, z], device=state.device, dtype=state.dtype)
            # angle_with_axis = torch.acos(torch.dot(position_vector, cone_axis) / position_vector.norm())
            # if angle_with_axis > cone_angle:
            #     projection_length = torch.dot(position_vector, cone_axis)
            #     projected_point = projection_length * cone_axis / torch.cos(cone_angle)
            #     x, y, z = projected_point
            new_state[i, 0] = x
            new_state[i, 1] = y
            new_state[i, 2] = z
            new_state[i, 3] = b1
            new_state[i, 4] = b2
            new_state[i, 5] = b3
        return new_state

    def forward(self, y, des):
        return self.KNet_step(y,des)

    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        self.h_S = weight.new_zeros(self.seq_len_input, self.batch_size, self.d_hidden_S)
        self.h_Sigma = weight.new_zeros(self.seq_len_input, self.batch_size, self.d_hidden_Sigma)
        self.h_Q = weight.new_zeros(self.seq_len_input, self.batch_size, self.d_hidden_Q)
    
    def compute_loss(self, y, target):
        # Compute loss between predicted position and target
        m1x_posterior_nn = self.forward(y)
        loss_fn = nn.MSELoss()
        return loss_fn(m1x_posterior_nn[:, :3], target[:, :3])

    def train_step(self, y, target, optimizer):
        optimizer.zero_grad()
        loss = self.compute_loss(y, target)
        loss.backward()
        optimizer.step()
        return loss.item()





