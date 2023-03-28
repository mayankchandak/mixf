from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.nat2021_test_path = '/workspace/Mayank/dataset/test/NAT2021'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/workspace/Mayank/mixf/data/got10k_lmdb'
    settings.got10k_path = '/workspace/Mayank/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/workspace/Mayank/mixf/data/lasot_lmdb'
    settings.lasot_path = '/workspace/Mayank/mixf/data/lasot'
    settings.network_path = '/workspace/Mayank/mixf/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/workspace/Mayank/mixf/data/nfs'
    settings.otb_path = '/workspace/Mayank/mixf/data/OTB2015'
    settings.prj_dir = '/workspace/Mayank/mixf'
    settings.result_plot_path = '/workspace/Mayank/mixf/test/result_plots'
    settings.results_path = '/workspace/Mayank/mixf/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/workspace/Mayank/mixf'
    settings.segmentation_path = '/workspace/Mayank/mixf/test/segmentation_results'
    settings.tc128_path = '/workspace/Mayank/mixf/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/workspace/Mayank/mixf/data/trackingNet'
    settings.uav_path = '/workspace/Mayank/mixf/data/UAV123'
    settings.vot_path = '/workspace/Mayank/mixf/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

