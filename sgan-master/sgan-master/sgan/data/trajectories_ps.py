import os
import numpy as np
import math
import copy
import cv2
import pickle
from scipy.stats import norm
from scipy.spatial import distance

import torch
from torch.utils.data import Dataset

from sgan.data.data_loader import DataLoader as dl
#from torch.utils.data import DataLoader as dl


def seq_collate(data):
    (obs_seq_list, 
     obs_seq_rel_list, 
     ground_truth_list, 
     mask_list, 
     render_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, seq_len, input_size
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(1, 0, 2)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(1, 0, 2)
    out = [
        obs_traj, 
        obs_traj_rel, 
        ground_truth_list,
        mask_list,
        render_list,
        seq_start_end
    ]

    return tuple(out)


def get_closest_interaction(traj1, traj2, is_ang=True):
    # Checks the closest interaction point between a pair of
    # trajectories. (closest = smallest L2 distance)
    # Inputs:
    # traj1, traj2: N x 2 numpy arrays
    #               Trajectories of length N
    #               Both trajectories must be synched and start 
    #               at the same time.
    # is_ang: True - returns information in radial coordinates (dist, angle)
    #         False - returns information in cartesian coordinates (dx, dy)
    # Outputs:
    # info: relative position at closest interaction point 
    #       (dist, angle) or (dx, dy)
    # min_frame: frame id when the clesest interaction happens
    #             relative to the start of the trajectories

    traj1_len = np.shape(traj1)[0]
    traj2_len = np.shape(traj2)[0]

    min_dist = np.inf
    min_dist_dir = 0
    min_rel_pos = [0, 0]
    min_frame = 0
    for f in range(min(traj1_len, traj2_len)):
        pos1 = traj1[f]
        pos2 = traj2[f]
        rel_pos = np.array(pos2) - np.array(pos1)
        dist = np.linalg.norm(rel_pos)
        if dist < min_dist:
            min_dist = dist
            min_dist_dir = np.arctan2(rel_pos[0], rel_pos[1])
            min_rel_pos = [rel_pos[0], rel_pos[1]]
            min_frame = f

    if is_ang:
        info = [min_dist, min_dist_dir]
    else:
        info = min_rel_pos
    return info, min_frame


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, datasets, obs_len=8, threshold=0.002, skip=10, 
        min_ped=2, is_ang=False, fps=25
    ):
        # Inputs:
        # datasets: list of indexes to generate the overall dataset
        #           0  - ETH
        #           1  - HOTEL
        #           2  - ZARA1
        #           3  - ZARA2
        #           4  - UNIV
        # obs_len: Number of time-steps in input trajectories
        # skip: Number of frames to skip while making the dataset
        # threshold: Minimum error to be considered for non linear traj
        #            when using a linear predictor (not used for now)
        # min_ped: Minimum number of pedestrians that should be in a seqeunce
        # is_ang: whether closest interaction in radial coordinates
        #         or cartesian coordinates
        # fps: fps to load the raw dataset. 
        #      Higher means more accurate closest interaction info
        #      but longer processing time.
        #      Also heavily correlated with skip
        #      e.g. fps=25, skip=10, the final dataset fps will be fps/skip = 2.5
        super(TrajectoryDataset, self).__init__()

        self.is_ang = is_ang
        a = dl('eth', 0, fps)
        b = dl('eth', 1, fps)
        c = dl('ucy', 0, fps)
        d = dl('ucy', 1, fps)
        e = dl('ucy', 2, fps)
        self.fps = fps
        self.datasets = datasets
        self.class_list = [a, b, c, d, e]
        
        # step 1: Gather pairwise pedetrian interaction info globally
        self.interaction_info = self._get_interaction_pairs(obs_len)
        print("Interaction info gathering complete")

        # step 2: Convert into and prepare for pyTorch dataloader
        self.inputs, self.ground_truths, self.masks, self.render_info = \
            self._init_environment(obs_len, skip, threshold, min_ped)
        print("Dataloader initialization complete")
        self.num_data = len(self.inputs)

        return

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        # Final processing may also happen here. Conver to various formats etc.
        obs_traj = self.inputs[index]
        obs_traj_rel = np.zeros_like(obs_traj)
        obs_traj_rel[:, 1:] = obs_traj[:, 1:] - obs_traj[:, :-1]
        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
        
        return [obs_traj,
                 obs_traj_rel, 
                 self.ground_truths[index], 
                 self.masks[index],
                 self.render_info[index]]


    def _get_interaction_pairs(self, obs_len):
        # the step to obtain pairwise global pedestrian closest interaction info
        interaction_info = {}
        for dt in self.datasets:
            #print('Processing dataset: {0:d}'.format(dt))
            cl = self.class_list[dt]
            start_frames = cl.people_start_frame
            end_frames = cl.people_end_frame

            num_people = len(start_frames)
            for i in range(num_people):
                for j in range(num_people):
                    if not (i == j):
                        has_interaction, start, end = self._check_interaction(
                                                        start_frames[i], start_frames[j],
                                                        end_frames[i], end_frames[j])
                        if has_interaction and ((end - start) >= obs_len):
                            traj1 = np.zeros((end - start, 2))
                            traj2 = np.zeros((end - start, 2))
                            for f in range(start, end):
                                state = self._gather_state_info(i, j, f, cl)
                                traj1[f - start, :] = state[0]['pos']
                                traj2[f - start, :] = state[1]['pos']

                            info, min_frame = get_closest_interaction(traj1, traj2, 
                                                                      self.is_ang)
                            interaction_info[(dt, i, j)] = \
                                (np.array(info), start + min_frame)

        return interaction_info

    def _check_interaction(self, start1, start2, end1, end2):
        # 'interaction' means the two pedestrians appear in the scene at the same time
        # for at least 1 frame.
        #
        # if they do, return from which frame to which frame this happens.

        start = max(start1, start2)
        end = min(end1, end2)
        if end <= start:
            return False, 0, 0
        else:
            return True, start, end + 1

    def _gather_state_info(self, per1, per2, frame, data_class):
        # This function returns information about pedestrians
        # Returns a tuple of two elements (pedestrian information), where
        # the first pedestrian is always the primary pedestrian.
        # Each pedestrian info is a dictionary containing the position and velocity
        # of the agent.

        start_frames = data_class.people_start_frame
        coordinates = data_class.people_coords_complete
        velocities = data_class.people_velocity_complete

        state1 = dict()
        state2 = dict()
        state1['pos'] = coordinates[per1][frame - start_frames[per1]]
        state2['pos'] = coordinates[per2][frame - start_frames[per2]]
        state1['vel'] = velocities[per1][frame - start_frames[per1]]
        state2['vel'] = velocities[per2][frame - start_frames[per2]]
        return (state1, state2)

    def _init_environment(self, seq_len, skip, threshold, min_ped):
        # initialize the dataset, decide what to store in input, ground truths
        # also initialize additional info such as masks and rendering info
        inputs = []
        ground_truths = []
        future_masks = []
        render_info = []
        for dt in self.datasets:
            print('Converting dataset: {0:d}'.format(dt))
            cl = self.class_list[dt]
            start_frames = cl.people_start_frame
            end_frames = cl.people_end_frame
            coordinates = cl.people_coords_complete
    
            for f in range(0, max(end_frames) - (seq_len * skip) - 1):
                curr_people = []
                num_people = len(start_frames)
                for p in range(num_people):
                    if (f >= start_frames[p]) and (f < (end_frames[p] - seq_len * skip)):
                        curr_people.append(p)
                num_curr_people = len(curr_people)

                if num_curr_people >= min_ped:
                    # get input
                    input_traj = np.zeros((num_curr_people, seq_len, 2))
                    for i in range(num_curr_people):
                        for j in range(seq_len):
                            p = curr_people[i]
                            input_traj[i, j] = \
                                np.array(coordinates[p][(f+j*skip) - start_frames[p]])

                    # get ground truth and mask
                    adj_matrix = np.zeros((num_curr_people, num_curr_people, 2))
                    future_mask = np.zeros((num_curr_people, num_curr_people))
                    for i in range(num_curr_people):
                        for j in range(num_curr_people):
                            if not(i == j):
                                p1 = curr_people[i]
                                p2 = curr_people[j]
                                key = (dt, p1, p2)
                                if key in self.interaction_info.keys():
                                    adj_matrix[i, j] = self.interaction_info[key][0]
                                    interaction_frame = self.interaction_info[key][1]
                                    if interaction_frame >= (f + seq_len):
                                        future_mask[i, j] = 1
                                else:
                                    print(key)
                                    raise Exception("Key doesn't exist: not possible!")
                    inputs.append(input_traj)
                    ground_truths.append(adj_matrix)
                    future_masks.append(future_mask)
                    render_info.append(f)

        return inputs, ground_truths, future_masks, render_info