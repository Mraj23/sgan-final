import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=(0, ), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_path = [0]
    val_path = [1]
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    print(args.pooling_type)
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


    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)


    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
       
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
       
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses':[],
            
            'losses_ts': [],
            'metrics_val': [],
            'metrics_train': [],
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
           
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None
        }
    t0 = None

    while t < args.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            
        
            step_type = 'g'
            losses_g = generator_step(args, batch, generator,
                                        optimizer_g)
            checkpoint['norm_g'].append(
                get_total_norm(generator.parameters())
            )


            for p in generator.parameters():
                if p.requires_grad:
                    print(torch.sum(p.grad))
            print(losses_g.item())
            exit()

             
            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))
          
            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                
                logger.info('  [loss]: {:.3f}'.format(losses_g.item()))
                checkpoint['G_losses'].append(losses_g.item())
                checkpoint['losses_ts'].append(t)

            #''' Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator)
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator)

                
                logger.info('  [val] : {:.3f}'.format(metrics_val))
                checkpoint['metrics_val'].append(metrics_val)
               
                logger.info('  [train] : {:.3f}'.format(metrics_train))
                checkpoint['metrics_train'].append(metrics_train)

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)

                logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                break

        print(losses_g.item())


def generator_step(args, batch, generator, optimizer_g):
    ''' batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
    loss_mask, seq_start_end) = batch #MODIFY LIKE INFERENCE'''
    obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
    obs_traj = obs_traj.cuda()
    obs_traj_rel = obs_traj_rel.cuda()
    #ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0]
    outputs = generator(obs_traj, obs_traj_rel, seq_start_end)
    
    #loss = torch.zeros(1).to(obs_traj)
    loss = None
    for i in range(len(outputs)):
        output = outputs[i]
        ground_truth, mask = ground_truth_list[i], mask_list[i]
        ground_truth = torch.tensor(ground_truth).cuda()
        mask = torch.tensor(mask).cuda()
        tmp_loss = torch.sum(torch.norm(output - ground_truth, dim = 2))
        if loss is None:
            loss = tmp_loss
        else:
            loss += tmp_loss


    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return loss


def check_accuracy(args, loader, generator):
    
    generator.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
            obs_traj = obs_traj.cuda()
            obs_traj_rel = obs_traj_rel.cuda()
            #ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0]
            outputs = generator(obs_traj, obs_traj_rel, seq_start_end)
            
            loss_output = torch.zeros(1).to(obs_traj)
            for i in range(len(outputs)):
                output = outputs[i]
                ground_truth, mask = ground_truth_list[i], mask_list[i]
                ground_truth = torch.tensor(ground_truth).cuda()
                mask = torch.tensor(mask).cuda()
                loss_output += torch.sum(torch.norm(output - ground_truth, dim = 2))
            total_loss += loss_output.item()
    generator.train()
    return total_loss


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


#data loader, remove discriminator things, change loss function

#In model: modify pooling (n*n) x h to n x n x h
#          attach 1x1 convolution layers after pool_h in generator's forward function, output channel size is 2