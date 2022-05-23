import cv2
import numpy as np

def _realign_trajectories(trajectories, traj_starts, v_fps, traj_fps, num_traj):
    # bring the trajectories in traj_fps time-synched to v_fps

    if v_fps == traj_fps:
        return trajectories
    else:
        fps_ratio = traj_fps / v_fps
        for i in range(num_traj):
            tmp_traj = []
            traj = trajectories[i]
            traj_len = np.shape(traj)[0]
            if traj_len > 1:
                timer = 0
                while (timer < (traj_len - 1)):
                    prev_idx = int(np.round(np.floor(timer)))
                    next_idx = int(np.round(np.ceil(timer)))
                    pos = traj[prev_idx] + ((traj[next_idx] - traj[prev_idx]) * 
                                            (timer - prev_idx))
                    tmp_traj.append(pos)
                    timer += fps_ratio
            trajectories[i] = np.array(tmp_traj)
        return trajectories

def _estimate_ends(trajectories, traj_starts, num_traj):
    # Add traj lengths to traj_starts
    traj_ends = np.zeros_like(traj_starts)
    for i in range(num_traj):
        traj_ends[i] = traj_starts[i] + np.shape(trajectories[i])[0]
    return traj_ends

def _get_colors():
    return [(255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 0, 127),
            (0, 127, 0),
            (127, 0, 0),
            (0, 127, 127),
            (127, 0, 127),
            (127, 127, 0)]

def plot_trajectories(vid_name, 
                      trajectories, 
                      traj_starts, 
                      traj_fps, 
                      output_name="output.avi"):
    # vid_name: name of the video to be read
    #   can be None for empty canvas
    # trajectories: N x T x 2 
    #   N - number of people; T - traj len; 2- x,y coordinates in pixels
    #   List of numpy arrays
    # traj_starts: coordinate starts in frames or frame ids (fps same as video)
    #   must be in same index order as N in "trajectories"
    #   A numpy array
    # traj_fps: The fps of the trajectories
    #   used to calculate how many frames are in between trajectories
    #   An integer
    # output_name: The output file name for the generated video

    ws = 3  # adjust for traj point size

    num_traj = len(trajectories)
    min_start = np.min(traj_starts)

    # Note about fps:
    # If vid_name is none, then output video fps = traj_fps
    # otherwise, output video fps = input video fps
    if not(vid_name is None):
        v_read = cv2.VideoCapture(vid_name)
        v_fps = v_read.get(cv2.cv.CV_CAP_PROP_FPS)
        width  = int(v_read.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(v_read.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        v_read.set(cv2.CV_CAP_PROP_POS_FRAMES, min_start)
    else:
        v_fps = traj_fps
        width = 1280
        height = 1024
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    v_write = cv2.VideoWriter(output_name, fourcc, v_fps, (width, height))

    trajectories = _realign_trajectories(trajectories, traj_starts, 
                                         v_fps, traj_fps, num_traj)

    traj_ends = _estimate_ends(trajectories, traj_starts, num_traj)
    max_end = np.max(traj_ends)

    colors = _get_colors()
    num_colors = len(colors)

    ped_set = np.zeros(num_traj)  # used to control which peds are currently in frame
    mask_set = np.zeros((num_traj, height, width, 3), dtype=np.uint8)
    curr_pt_set = np.zeros((num_traj, 2), dtype=np.int32)
    vid_idx = min_start
    while (vid_idx < max_end):
        if not(vid_name is None):
            if (v_read.isOpened()):
                ret, frame = v_read.read()
                if ret == False:
                    raise Exception("grab frame error!")
            else:
                raise Exception("v_read not open!")
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(num_traj):
            if (vid_idx >= traj_starts[i]) and (vid_idx < traj_ends[i]):
                if ped_set[i] == 0:
                    ped_set[i] = 1
                pt = np.int32(np.round(trajectories[i][vid_idx - traj_starts[i]]))
                
                window_r = list(range(max(pt[1]-ws, 0), min(pt[1]+ws, height-1) + 1))
                window_c = list(range(max(pt[0]-ws, 0), min(pt[0]+ws, width-1) + 1))
                mask = np.zeros((len(window_r), len(window_c), 3), dtype=np.uint8) + 1
                mask_set[i, 
                         max(pt[1]-ws, 0):(min(pt[1]+ws, height-1) + 1),
                         max(pt[0]-ws, 0):(min(pt[0]+ws, width-1) + 1),
                         :] = 1

                curr_pt_set[i] = pt
            else:
                if ped_set[i] == 1:
                    ped_set[i] = 0

        for i in range(num_traj):
            if ped_set[i] == 1:
                color = colors[(i % num_colors) + 1]
                color_mask = mask_set[i]
                color_mask[:, :, 0] *= color[0]
                color_mask[:, :, 1] *= color[1]
                color_mask[:, :, 2] *= color[2]
                frame = frame * (1 - mask_set[i]) + color_mask
                frame = cv2.putText(
                  img = frame,
                  text = str(i),
                  org = (curr_pt_set[i, 0], curr_pt_set[i, 1]),
                  fontFace = cv2.FONT_HERSHEY_DUPLEX,
                  fontScale = 1.0,
                  color = (255, 255, 255),
                  thickness = 1
                )
        v_write.write(frame)
        vid_idx += 1;

    if not(vid_name is None):
        v_read.release()
    v_write.release()
    return