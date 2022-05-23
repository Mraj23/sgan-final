"""
This code is largely adopted from the sgan code developed by the Authors of SocialGAN
A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi.  Social GAN: Socially acceptable324trajectories with generative adversarial networks.  
InProceedings of the IEEE Conference on325Computer Vision and Pattern Recognition (CVPR), pages 2255-2264, 2018

Author: Agrim Gupta
Link: https://github.com/agrimgupta92/sgan
"""


import argparse
import os
import torch
import numpy as np

from attrdict import AttrDict

from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from sgan.data.loader import data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

class SGANInference(object):
    
    def __init__(self, model_path):
        # To initialize it, a path to a pretrained model is needed
        # models are stored in sgan/models
        # for example: model_path = "models/sgan-models/eth_8_model.pt"
        #
        # model checkpoint names are like this
        # (dataset name)_(observation length)_model.pt
        #
        # dataset_name is the dataset that is the test set 
        # (the dataset that was not seen during the training of this model)
        #
        # obervation_length is the input length of the trajectory to this model

        path = model_path
        # number of samples to draw to get the final predicted trajectory
        self.num_samples = 20

        self.cuda = torch.device('cuda:0')
        checkpoint = torch.load(path,map_location=torch.device('cpu'))
        self.generator = self.get_generator(checkpoint)
        self.args = AttrDict(checkpoint['args'])
        return
        
    def get_generator(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.to(self.cuda)
        generator.eval()
        return generator


    def evaluate(self, obs_traj):
        # inputs:
        # depending on the observation length of your chosen model
        # the output obs_traj should be a numpy array of either size nx8x2 or nx12x2
        # where n is the number of people in the scene
        # obs_traj is simply the observed trajectories (sequence of coordinates)
        #
        # outputs:
        # outputs nx8x2 predicted trajectories (sequence of coordinates)

        num_people = obs_traj.shape[0]
        traj_length = obs_traj.shape[1]
        obs_traj_rel = obs_traj[:, 1:traj_length] - obs_traj[:, 0:traj_length - 1]
        obs_traj_rel = np.append(np.array([[[0,0]]] * num_people), obs_traj_rel, axis=1)
        obs_traj = np.transpose(obs_traj, (1, 0, 2))
        obs_traj_rel = np.transpose(obs_traj_rel, (1, 0, 2))
        seq_start_end = np.array([[0, num_people]])

        with torch.no_grad():
            obs_traj = torch.from_numpy(obs_traj).type(torch.float)
            obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
            seq_start_end = torch.from_numpy(seq_start_end).type(torch.int)
            obs_traj = obs_traj.to(self.cuda)
            obs_traj_rel = obs_traj_rel.to(self.cuda)
            seq_start_end = seq_start_end.to(self.cuda)

            pred_traj_avg = []
            for _ in range(self.num_samples):
                pred_traj_fake_rel = self.generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]

                )
                tmp = pred_traj_fake.cpu().numpy()
                pred_traj_avg.append(np.transpose(tmp,(1,0,2)))
            pred_traj_avg = np.mean(np.asarray(pred_traj_avg), 0)

        return pred_traj_avg



def computeDistances(pred_traj,n):

    num_people = len(pred_traj)
    time_steps = len(pred_traj[0])
    xy_cors = [[100,100] for x in range(num_people) ]

    n_traj = pred_traj[n]
    
    for i in range(num_people):
        if i != n:
            p_traj = pred_traj[i]
            rel_pos = np.array(p_traj) - np.array(n_traj)
            dist = (rel_pos[:,0]**2) + (rel_pos[:,1]**2)
            index = dist.argmin()
            sx, sy = rel_pos[index]
            xy_cors[i][0], xy_cors[i][1] = sx,sy 
    return xy_cors


def get_model_matrix(matrices):
    model_matrix = np.zeros((len(matrices[0]),len(matrices[0]),2))
    time_len = np.shape(matrices)[0]
    num_ppl = np.shape(matrices)[1]

    for i in range(num_ppl):
        for j in range(num_ppl):
            min_dist = np.Inf
            for t in range(time_len):
                dist = matrices[t, i, j, 0]**2 + matrices[t, i, j, 1]**2
                if dist < min_dist:
                    model_matrix[i, j] = matrices[t, i, j, :]
                    min_dist = dist

    """
    for i in range(len(matrices[0])):
        row = matrices[:,i,:,:]
        #print(np.shape(row))
        #print('row)')
        #print(row)
        for j in range(len(matrices[0])):
            subrow = row[:,j,:]
            #print(np.shape(subrow))
            dist = (subrow[:,0]**2) + (subrow[:,1]**2)
            #print(dist)
    """
        
        
    return model_matrix
        




trajPlan = SGANInference("models/sgan-models/eth_8_model.pt")
generator = trajPlan.generator
_args = trajPlan.args
_args.batch_size = 1
sets = [0]
_, loader = data_loader(_args, sets)

for batch in loader:
    obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
    ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0]
    print(ground_truth)
    if np.sum(mask) == 0:
        print('not good batch')
        break
    obs_traj = obs_traj.numpy()
    traj_length = len(obs_traj[1])
    matrices = []
    predicted = obs_traj.transpose((1,0,2))
    for j in range(50):
        sub_matrices = np.zeros((len(predicted),len(predicted),2))
        history = predicted
        predicted = trajPlan.evaluate(history)
        for x in range(len(predicted)):
            cors = computeDistances(predicted,x) #adjancey matrix
            cors = np.array(cors)
            sub_matrices[x,:,:] = cors
        #print('sub matrices')
        #print(sub_matrices)
        matrices.append(sub_matrices)
    matrices = np.array(matrices)
    
    model_matrix = get_model_matrix(matrices)
    #print(model_matrix)
    mse_metric = np.sum(np.linalg.norm(model_matrix - ground_truth, axis = 2)*mask)
    #print(mse_metric)
    #IF MASK is
    break





    
#print(trajPlan)

#take 1 batch
#make x number of predictions, find adjancey matrix and make a list of them
#find the best matrix out of them all (minimum)