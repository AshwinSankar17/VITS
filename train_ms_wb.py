import os
import json
import math
import torch
import wandb
import argparse
import itertools
import argparse
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioSpeakerEmotionLoader,
  TextAudioSpeakerEmotionCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

def get_rank():
  if not dist.is_available() or not dist.is_initialized():
    return 0
  return dist.get_rank()

class WandbLogger:
    def __init__(self, **kwargs):
      if self.is_primary:
        wandb.init(
          project=kwargs.get('wandb_project', "vits"),
          name=kwargs.get('wandb_run_name', "vits_rasa_preboosting"),
          entity=kwargs.get('wandb_entity', "indic-asr"),
          config=kwargs,  # Save all args as configuration
        )

    @property
    def is_primary(self):
      return get_rank() == 0

    def save_config(self, config):
      if self.is_primary:
        wandb.config.update(config)  # Save config to Wandb
          
    def log_dict(self, scalars, global_step):
      if self.is_primary:
        wandb.log(scalars, step=global_step)

    def add_scalar(self, tag, scalar_value, global_step=None):
      if self.is_primary:
        wandb.log({tag: scalar_value}, step=global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
      if self.is_primary:
        wandb.log({f"{main_tag}/{key}": value for key, value in tag_scalar_dict.items()}, step=global_step)

    def add_image(self, tag, img_tensor, global_step=None):
      if self.is_primary:
        wandb.log({tag: wandb.Image(img_tensor)}, step=global_step)

    def add_images(self, main_tag, img_tensors, global_step=None):
      if self.is_primary:
        wandb.log({f"{main_tag}/{key}": wandb.Image(img) for key, img in img_tensors.items()}, step=global_step)
    
    def add_audios(self, main_tag, audio_tensors, sampling_rate, global_step=None):
      if self.is_primary:
        wandb.log({f"{main_tag}/{key}": wandb.Audio(audio, sampling_rate) for key, audio in audio_tensors.items()}, step=global_step)

    def close(self):
      if self.is_primary:
        wandb.finish()

torch.backends.cudnn.benchmark = False
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  print ('num of gpus', n_gpus)
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8106'

  hps = utils.get_hparams()
  # run(0, 1, hps)
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    # writer = SummaryWriter(log_dir=hps.model_dir)
    # writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    wandb_logger = WandbLogger(**hps.__dict__)

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerEmotionLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerEmotionCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerEmotionLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      n_emotions=hps.data.n_emotions, #####
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
    
    # epoch_str = 1
    
    # global_step = 0
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, wandb_logger)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  wandb_logger = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emotions) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)
    emotions = emotions.cuda(rank, non_blocking=True) #####

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers, emotions)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {
          "learning_rate": lr, 
          "grad_norm_d": grad_norm_d, 
          "grad_norm_g": grad_norm_g,
        }
        wandb_logger.log_dict(scalar_dict, global_step=global_step)
        scalar_dict = {
          "g_loss_total": loss_gen_all, 
          "d_loss_total": loss_disc_all,
          "g_fm": loss_fm, "mel": loss_mel, "dur": loss_dur, "kl": loss_kl,
          **{"g_{}".format(i): v for i, v in enumerate(losses_gen)},
          **{"d_r_{}".format(i): v for i, v in enumerate(losses_disc_r)},
          **{"d_g_{}".format(i): v for i, v in enumerate(losses_disc_g)}
        }
      
        wandb_logger.add_scalars("train", scalar_dict, global_step=global_step)

        image_dict = { 
            "slice_mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice_mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all_mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all_attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }

        wandb_logger.add_images("train", image_dict)
        # utils.summarize(
        #   writer=writer,
        #   global_step=global_step, 
        #   images=image_dict,
        #   scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, wandb_logger)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, wandb_logger):
    # print ("Evaluating the model now!!")
    generator.eval()
    mel_loss_total = []
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():

      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, emotions) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        speakers = speakers.cuda(0)
        emotions = emotions.cuda(0)

        # remove else
        # x = x[:1]
        # x_lengths = x_lengths[:1]
        # spec = spec[:1]
        # spec_lengths = spec_lengths[:1]
        # y = y[:1]
        # y_lengths = y_lengths[:1]
        # speakers = speakers[:1]
        # emotions = emotions[:1]
        # break

      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, emotions, max_len=2000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )

      if (batch_idx == 0):
        image_dict = {
        "gen_mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
        }
        audio_dict = {
          "gen_audio": y_hat[0,:,:y_hat_lengths[0]]
        }
        if global_step == 0:
          wandb_logger.log_dict({"gt_mel": wandb.Image(utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()))})
          wandb_logger.log_dict({"gt_audio": wandb.Audio(y[0,:,:y_lengths[0]], hps.data.sampling_rate)})


      if mel.size(-1) > y_hat_mel.size(-1):
          # Case 1: `mel` is longer, so pad `y_hat_mel`
          length_diff = mel.size(-1) - y_hat_mel.size(-1)
          y_hat_mel = F.pad(y_hat_mel, (0, length_diff))
          # Mask the padded portion in `mel`
          mask = torch.arange(mel.size(-1)).expand(len(y_lengths), mel.size(-1)).cuda(0)
          mask = (mask < y_hat_lengths.unsqueeze(1)).float()
          mel_masked = mel * mask.unsqueeze(1)
          y_hat_mel_masked = y_hat_mel
          # y_hat_mel_masked = y_hat_mel * mask.unsqueeze(1)
          # mel_masked = mel

      elif y_hat_mel.size(-1) > mel.size(-1):
          # Case 2: `y_hat_mel` is longer, so pad `mel`
          length_diff = y_hat_mel.size(-1) - mel.size(-1)
          mel = F.pad(mel, (0, length_diff))

          # Mask the padded portion in `y_hat_mel`
          mask = torch.arange(y_hat_mel.size(-1)).expand(len(y_hat_lengths), y_hat_mel.size(-1)).cuda(0)
          mask = (mask < mel.size(-1)).float()
          # mel_masked = mel * mask.unsqueeze(1)
          # y_hat_mel_masked = y_hat_mel
          y_hat_mel_masked = y_hat_mel * mask.unsqueeze(1)
          mel_masked = mel

      else:
          mel_masked = mel
          y_hat_mel_masked = y_hat_mel

      # print ("calculated the masks and padding in evaluation")
        # Loss calculation
      with autocast(enabled=False):
        loss_mel = F.l1_loss(mel_masked, y_hat_mel_masked) * hps.train.c_mel
        # print (loss_mel)
        mel_loss_total.append(loss_mel.item())


    mel_loss_epoch = torch.tensor(mel_loss_total).float().mean()
    # print (mel_loss_epoch)
    scalar_dict = {"mel": mel_loss_epoch}

    wandb_logger.add_scalars("val", scalar_dict, global_step=global_step)
    wandb_logger.add_images("val", image_dict, global_step=global_step)
    wandb_logger.add_audios("val", audio_dict, hps.data.sampling_rate, global_step=global_step)

    # utils.summarize(
    #   writer=writer_eval,
    #   global_step=global_step, 
    #   images=image_dict,
    #   audios=audio_dict,
    #   audio_sampling_rate=hps.data.sampling_rate,
    #   scalars=scalar_dict
    # )
    generator.train()

        
if __name__ == "__main__":
  main()
