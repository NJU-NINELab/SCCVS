
class Config():
    batch_size = 64
    training_epoch = 50
    channel = "AWGN" #"AWGN"
    SNR_MAX = 25
    SNR_MIN = 0
    snr = 0
    CR_low = 0.5
    CR_high = 0.9
    lr = 1e-5
    ch_lr = 1e-4
    weight_delay = 1e-5
    device = "cuda"
    checkpoints_dir = "checkpoints"
    logs_dir = f"logs"
    dataset_path = r"/jiangfeibo/pyb/datasets/Park"
    # dataset_path = r"D:\pythonProject\PythonProj\sd_server\datasets\Park"
    # dataset_path = r"/data/newbg/pyb"
