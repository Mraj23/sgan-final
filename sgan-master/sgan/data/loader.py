from torch.utils.data import DataLoader
from sgan.data.trajectories_ps import TrajectoryDataset, seq_collate #traj ps


def data_loader(args, sets):
    dset = TrajectoryDataset(
        sets, 
        obs_len=args.obs_len,
        skip=args.skip, 
        is_ang=False, 
        fps=25)

    print(args.batch_size)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
