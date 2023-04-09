import os
# loss function related
from lib.utils.box_ops import ciou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mixformer_vit import build_mixformer_vit

# forward propagation related
from lib.train.actors import MixFormerActor
from lib.models.trans_discriminator import TransformerDiscriminator
# for import modules
import importlib


def prepare_input(res):
    res_t, res_s = res
    t = torch.FloatTensor(1, 3, res_t, res_t).cuda()
    s = torch.FloatTensor(1, 3, res_s, res_s).cuda()
    return dict(template=t, search=s)


def run(settings):
    settings.description = 'Training script for Mixformer'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val, loader_nat, data_processing_train = build_dataloaders(cfg, settings)

    # Create network
    if settings.script_name == "mixformer_vit":
        net = build_mixformer_vit(cfg)
        # print("load pretrain")
        print("Model loading from:", '/workspace/Mayank/original-udat/mixformer.pth')
        net.load_state_dict(torch.load('/workspace/Mayank/original-udat/mixformer.pth', map_location='cpu')['net'], strict=False)
        # print("load done")
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.save_every_epoch = True
    # Loss functions and Actors
    if settings.script_name in ["mixformer_vit"]:
        objective = {'ciou': ciou_loss, 'l1': l1_loss}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    accum_iter = getattr(cfg.TRAIN, "ACCUM_ITER", 1)

    model_Disc = TransformerDiscriminator(channels=256)
    optimizer_D = torch.optim.Adam(model_Disc.parameters(), lr=0.005, betas=(0.9, 0.99))
    model_Disc.cuda().train()

    if settings.local_rank != -1:
        model_Disc = DDP(model_Disc, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, model_Disc, optimizer_D, lr_scheduler, accum_iter=accum_iter, use_amp=use_amp, nat_loader=loader_nat, data_processing_train=data_processing_train)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
