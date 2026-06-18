import torch
#torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
import torch.nn as nn
import time
from Filters.EKF_withbias_IK import ExtendedKalmanFilterWithIK
import random
from Utils import plot_training_curves

"""def EKFTest(args, SysModel, IKModel, test_input, test_target, optimizer, allStates=True, randomInit=False, test_init=None, test_lengthMask=None):
    N_T = test_target.size()[0]
    loss_fn = nn.MSELoss(reduction='mean')
    MSE_EKF_linear_arr = torch.zeros(N_T)
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]])
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]])

    start = time.time()

    # Instantiate the EKF with the IK neural network as the observation model
    EKF = ExtendedKalmanFilterWithIK(SysModel, IKModel, args)

    # Initialize EKF


    if randomInit:
        EKF.Init_batched_sequence(test_init, SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1))
    else:
        # Check if the shape is [150, 6] (which is already batch size x state dimension)
        if SysModel.m1x_0.shape == (N_T, SysModel.m):
            # Reshape m1x_0 to [N_T, m, 1] without expanding
            EKF.Init_batched_sequence(
                SysModel.m1x_0.unsqueeze(-1),  # Add a new dimension to get [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand m2x_0 across batch size
            )
        else:
            # Handle other cases (e.g., unexpected shape)
            EKF.Init_batched_sequence(
                SysModel.m1x_0.view(1, SysModel.m, 1).expand(N_T, -1, -1),  # Correctly reshape and expand to [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand across batch size
            )

    EKF.GenerateBatch(test_input)
    end = time.time()
    t = end - start

    KG_array = EKF.KG_array
    EKF_out = EKF.x

    for j in range(N_T):
    # Print current iteration
        print(f"Iteration {j + 1}/{N_T}")

        if allStates:
            if args.randomLength:
                loss = loss_fn(EKF.x[j, :, test_lengthMask[j]], test_target[j, :, test_lengthMask[j]])
            else:
                loss = loss_fn(EKF.x[j, :, :], test_target[j, :, :])
        else:
            loc = torch.tensor([True, False, False])  # Adjust based on which states to consider (e.g., position only)
            if args.randomLength:
                loss = loss_fn(EKF.x[j, loc, test_lengthMask[j]], test_target[j, loc, test_lengthMask[j]])
            else:
                loss = loss_fn(EKF.x[j, loc, :], test_target[j, loc, :])

    # Print the current loss
        print(f"Loss at iteration {j + 1}: {loss.item()}")


    # Backpropagation and parameter update
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # Backpropagate the loss on the posterior
        optimizer.step()


        MSE_EKF_linear_arr[j] = loss.item()

    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    print("EKF with IK (Posterior Loss) - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")
    print("Inference Time:", t)

    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out]
"""
def EKF_Train(args ,SysModel, IKModel, cv_input, cv_target, cv_des, train_input, train_target, train_des, optimizer, path_results):
    
    N_steps = args.n_steps_EKF

    N_E = len(train_input)
    N_CV = len(cv_input)
    N_B = args.n_batch
    loss_fn = nn.MSELoss(reduction='mean')
    
    MSE_cv_linear_epoch = torch.zeros([N_steps])
    MSE_cv_dB_epoch = torch.zeros([ N_steps])

    MSE_train_linear_epoch = torch.zeros([N_steps])
    MSE_train_dB_epoch = torch.zeros([N_steps])
    

    ##############
    ### Epochs ###
    ##############

    MSE_cv_dB_opt = 1000
    MSE_cv_idx_opt = 0
    
    EKF_train = ExtendedKalmanFilterWithIK(SysModel, IKModel, args).to(args.device)
    EKF_cv = ExtendedKalmanFilterWithIK(SysModel, IKModel, args).to(args.device)
    
    for ti in range(0,N_steps):

        ###############################
        ### Training Sequence Batch ###
        ###############################
        optimizer.zero_grad()
        # Training Mode
        IKModel.train()   

        # Init Training Batch tensors
        y_training_batch = torch.zeros([N_B, SysModel.n, SysModel.T]).to(args.device)
        train_target_batch = torch.zeros([N_B, SysModel.m, SysModel.T]).to(args.device)
        x_out_training_batch = torch.zeros([N_B, SysModel.m, SysModel.T]).to(args.device)
        train_des_batch = torch.zeros([N_B, SysModel.n, SysModel.T]).to(args.device)

        # Randomly select N_B training sequences
        assert N_B <= N_E # N_B must be smaller than N_E
        n_e = random.sample(range(N_E), k=N_B)
        ii = 0
        for index in n_e:
            y_training_batch[ii,:,:] = train_input[index]
            train_target_batch[ii,:,:] = train_target[index]
            train_des_batch[ii,:,:] = train_des[index]
            ii += 1
        
        # EKF_train = ExtendedKalmanFilterWithIK(SysModel, IKModel, args).to(args.device)
        
        # Init Sequence
        EKF_train.Init_batched_sequence(
            #SysModel.m1x_0[:N_B, :].unsqueeze(-1),  # Correctly reshape and expand to [N_T, m, 1]
            SysModel.m1x_0[n_e, :].unsqueeze(-1),  # Correctly reshape and expand to [N_T, m, 1]
            SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_B, -1, -1) ) # Expand across batch size
    
        # Forward Computation
        EKF_train.GenerateBatch(y_training_batch,train_des_batch)
        KG_array_train = EKF_train.KG_array
        EKF_out_train = EKF_train.x
        EKF_out_m1y_train = EKF_train.y_hat
        bias_prior = EKF_train.bias_prior_traj
        j_t = EKF_train.j_t_traj
        
        # Compute Training Loss
        #MSE_trainbatch_linear_LOSS = 0
        if args.CompositionLoss2:
            # MSE_trainbatch_linear_LOSS = loss_fn(EKF_out_train[:,:3,:], train_target_batch[:,:3,:]) + args.alpha*loss_fn(EKF_out_m1y_train - bias_prior,y_training_batch)
            MSE_trainbatch_linear_LOSS = loss_fn(EKF_out_train[:,:3,:], train_target_batch[:,:3,:]) + args.alpha*loss_fn(j_t,y_training_batch)
        else:
                
            MSE_trainbatch_linear_LOSS = loss_fn(EKF_out_train[:,:3,:], train_target_batch[:,:3,:])

        # dB Loss
        MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
        MSE_train_dB_epoch[ti] = 10 * torch.log10(MSE_train_linear_epoch[ti])
        print("---------------------------------------------------------")
        print("EKF linear MSE positin:", loss_fn(EKF_out_train[:,:3,:], train_target_batch[:,:3,:]))
        print("EKF linear MSE bias with alpha:",args.alpha*loss_fn(j_t,y_training_batch) )
        
        ##################
        ### Optimizing ###
        ##################

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        MSE_trainbatch_linear_LOSS.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(IKModel.parameters(), max_norm=1.0)
        print('-----gradients:-----', IKModel.fc1.weight.grad.norm())
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

        #################################
        ### Validation Sequence Batch ###
        #################################

        # Cross Validation Mode
        IKModel.eval()
        
        with torch.no_grad():
            # EKF_cv = ExtendedKalmanFilterWithIK(SysModel, IKModel, args)
            
            SysModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences

            x_out_cv_batch = torch.empty([N_CV, SysModel.m, SysModel.T_test]).to(args.device)
            
            # Init Sequence
            EKF_train.Init_batched_sequence( # EKF_train
                # SysModel.m1x_0[:N_CV, :].unsqueeze(-1),  # Correctly reshape and expand to [N_T, m, 1]
                cv_target[:,:,0].unsqueeze(-1),  # Correctly reshape and expand to [N_T, m, 1]
                SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_CV, -1, -1) ) # Expand across batch size  # used to be EKF_cv
            #SysModel.m1x_0.view(1, SysModel.m, 1).expand(N_CV, -1, -1)
            # execute EKF with IK DNN
            EKF_train.GenerateBatch(cv_input,cv_des) # used to be EKF_cv # used to be EKF_train
            KG_array_cv = EKF_train.KG_array  # used to be EKF_cv
            EKF_out_cv = EKF_train.x  # used to be EKF_cv
            
            # Compute CV Loss
            #MSE_cvbatch_linear_LOSS = 0
            
            MSE_cvbatch_linear_LOSS = loss_fn(EKF_out_cv[:,:3,:], cv_target[:,:3,:])

            # dB Loss
            MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
            MSE_cv_dB_epoch[ti] = 10 * torch.log10(MSE_cv_linear_epoch[ti])
            
            if (MSE_cv_dB_epoch[ti] < MSE_cv_dB_opt):
                MSE_cv_dB_opt = MSE_cv_dB_epoch[ti]
                MSE_cv_idx_opt = ti
                
                torch.save(IKModel, path_results + 'best-model-IKModel.pt')

        ########################
        ### Training Summary ###
        ########################
        print(ti, "EKF MSE Training :", MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", MSE_cv_dB_epoch[ti],
              "[dB]")
                  
        if (ti > 1):
            d_train = MSE_train_dB_epoch[ti] - MSE_train_dB_epoch[ti - 1]
            d_cv = MSE_cv_dB_epoch[ti] - MSE_cv_dB_epoch[ti - 1]
            print("EKF diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

        print("EKF Optimal idx:", MSE_cv_idx_opt, "Optimal :", MSE_cv_dB_opt, "[dB]")
    plot_training_curves(MSE_train_dB_epoch, MSE_cv_dB_epoch, save_path="loss_curve_EKF.png")
    return [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch]
    
def EKFTest(args, SysModel, test_input, test_target,test_des, path_results):
    N_T = test_target.size()[0]
    loss_fn = nn.MSELoss(reduction='mean')
    MSE_EKF_linear_arr = torch.zeros(N_T)
    EKF_out = torch.zeros([N_T, SysModel.m, test_input.size()[2]])
    KG_array = torch.zeros([N_T, SysModel.m, SysModel.n, test_input.size()[2]])

    #start = time.time()

    # Instantiate the EKF with the IK neural network as the observation model
    IKModel = torch.load(path_results + 'best-model-IKModel.pt', map_location=args.device)
    EKF = ExtendedKalmanFilterWithIK(SysModel, IKModel, args)


            # # Check if the shape is [150, 6] (which is already batch size x state dimension)
            # if SysModel.m1x_0.shape == (N_T, SysModel.m):
            #     EKF.Init_batched_sequence(
            #         SysModel.m1x_0.unsqueeze(-1),  # Add a new dimension to get [N_T, m, 1]
            #         SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1)  # Expand m2x_0 across batch size
            #     )
    SysModel.m1x_0 = test_target[:,:,0] 
    start = time.time()     
    EKF.Init_batched_sequence(
                    SysModel.m1x_0.unsqueeze(-1),  # Correctly reshape and expand to [N_T, m, 1]
                    SysModel.m2x_0.view(1, SysModel.m, SysModel.m).expand(N_T, -1, -1))  # Expand across batch size)
    EKF.GenerateBatch(test_input, test_des)
    if args.device == "cuda":
        torch.cuda.synchronize()  # <-- force CPU to wait for GPU work
    
    end = time.time()
    t = end - start
    
    KG_array = EKF.KG_array
    EKF_out = EKF.x
        
    
    # Compute loss for the entire batch
        
    loss = loss_fn(EKF.x[:,:3,:].to(args.device),test_target[:,:3,:].to(args.device))
        
    print(f'EKF test linearloss: {loss}')
    
    
    MSE_EKF_linear_avg = loss.item()
    MSE_EKF_dB_avg = 10 * torch.log10(torch.tensor(MSE_EKF_linear_avg))


    return [MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, t]
