from pathlib import Path
import torch


class Config(object):
    """Main config file"""

    # torch param
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Is available device:  {device}')
    # Dataset parameters
    num_classes = 10
    class_names = range(num_classes)

    # path to train and validate csv file
    train_csv = None
    test_csv = None
    ckp_dir = Path('weights')

    # train params
    train_batch_size = 256
    test_batch_size = 256
    num_workers = 0
    lr = 0.0001
    best_weights_path = Path('./weights/best_ckp.pth')
    last_weights_path = Path('./weights/last_ckp.pth')
    epochs = 20
    ckp_interval = 1
    weight_decay = 1e-5
    predict_dir = Path('predict')
