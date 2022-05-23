from trajectories_ps import TrajectoryDataset, seq_collate

from torch.utils.data import DataLoader

a = TrajectoryDataset([0])
loader = DataLoader(a, batch_size=8, shuffle=True, collate_fn=seq_collate)

for batch in loader:
    print('batch\n')
    #print(batch)
    print('size\n')
    obs, gt, masks, start, end, o = batch
    print(obs.size())
    print(gt.size())
    print(len(masks[0]))
    print(len(batch))
    break