class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/workspace/Mayank/mixf'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/workspace/Mayank/mixf/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/workspace/Mayank/mixf/pretrained_networks'
        self.lasot_dir = '/workspace/Mayank/mixf/data/lasot'
        self.got10k_dir = '/workspace/Mayank/mixf/data/got10k/train'
        self.lasot_lmdb_dir = '/workspace/Mayank/mixf/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/workspace/Mayank/mixf/data/got10k_lmdb'
        self.trackingnet_dir = '/workspace/Mayank/mixf/data/trackingnet'
        self.trackingnet_lmdb_dir = '/workspace/Mayank/mixf/data/trackingnet_lmdb'
        self.coco_dir = '/workspace/Mayank/mixf/data/coco'
        self.coco_lmdb_dir = '/workspace/Mayank/mixf/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/workspace/Mayank/mixf/data/vid'
        self.imagenet_lmdb_dir = '/workspace/Mayank/mixf/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
