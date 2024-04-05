import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from dataset.dataset import Mnist
from utils.trainer import Trainer, Losser
from utils.augment import TrainTransforms, TestingTransform
from model.model import Net
from cfg import Config


def parse_args():
    """ Parse command line params. """

    parser = argparse.ArgumentParser()
    # dataset params
    parser.add_argument(
        '--train_csv',
        type=Path,
        required=True,
        help='Path to train csv file.'
    )
    parser.add_argument(
        '--test_csv',
        type=Path,
        required=True,
        help='Path to test csv file.'
    )
    parser.add_argument(
        '--ckp_dir',
        type=Path,
        default='weights',
        help='Path to checkpoint folder.'
    )
    parser.add_argument(
        '--ckp_interval',
        type=int,
        default=1,
        help='Interval save weights.'
    )
    # train params
    parser.add_argument(
        '--train_batch',
        type=int,
        default=1000,
        help='Train batch size.'
    )
    parser.add_argument(
        '--test_batch',
        type=int,
        default=1000,
        help='Test batch size.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Num epochs in train.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay.'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # init config file
    cfg = Config()

    # edit config param
    cfg.train_csv = Path(args.train_csv)
    cfg.test_csv = Path(args.test_csv)
    cfg.ckp_dir = Path(args.ckp_dir)
    # create_dir
    if not cfg.ckp_dir.is_dir():
        cfg.ckp_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckp_interval = args.ckp_interval
    cfg.train_batch_size = args.train_batch
    cfg.test_batch_size = args.test_batch
    cfg.lr = args.lr
    cfg.epochs = args.epochs
    cfg.weight_decay = args.weight_decay

    # init augment for train and validate
    transform_train = TrainTransforms()
    transform_test = TestingTransform()

    # init dataset and dataloader
    train_data = Mnist(cfg.train_csv,
                       cfg,
                       transform_train)
    train_dataloader = DataLoader(train_data,
                                  cfg.train_batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers)
    test_data = Mnist(cfg.test_csv,
                      cfg,
                      transform_test)
    test_dataloader = DataLoader(test_data,
                                 cfg.test_batch_size,
                                 shuffle=False,
                                 num_workers=cfg.num_workers)

    # init model
    model = Net()
    # init losser
    losser = Losser(cfg)
    # init Trainer
    trainer = Trainer(cfg, model, losser,
                      train_loader=train_dataloader,
                      val_loader=test_dataloader)

    # hold on, machine learning moment (/*_*)/ ┻━━┻
    trainer.train()


if __name__ == '__main__':
    """ Main function. """
    main()

