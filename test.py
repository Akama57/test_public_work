import argparse

from cfg import Config
from pathlib import Path
from torch.utils.data import DataLoader
from dataset.dataset import Mnist
from utils.trainer import Trainer, Losser
from utils.augment import TestingTransform
from model.model import Net
from typing import Union


def parse_args():
    """ Parse command line params."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_csv',
        type=Path,
        required=True,
        help='Path to test csv file.'
    )
    parser.add_argument(
        '--test_batch',
        type=int,
        default=1000,
        help='Test batch size.'
    )

    # prediction csv file
    parser.add_argument(
        '--save_predict_path',
        type=Path,
        default='prediction',
        help='Path to save prediction csv file.'
    )
    parser.add_argument(
        '--best_weights',
        type=Union[bool, int],
        default=True,
        help='Choose weights for the test.\n'
             'You can choose the best, latest and with a specific number.'
    )
    return parser.parse_args()


def main():
    """ Main function. """
    args = parse_args()
    # init config file
    cfg = Config()
    # edit config param
    cfg.test_csv = Path(args.test_csv)
    cfg.predict_dir = Path(args.save_predict_path)

    if not cfg.predict_dir.is_dir():
        cfg.predict_dir.mkdir(parents=True, exist_ok=True)

    cfg.resume_from_best = args.best_weights

    # init transform
    transform_test = TestingTransform()
    # init dataset and dataloader
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
                      train_loader=None,
                      val_loader=test_dataloader)
    # init prediction file name
    predict_file = Path(cfg.predict_dir, 'prediction.csv')
    # load weights
    if type(cfg.resume_from_best) == bool:
        trainer.load_weights(epoch_number=None)
    else:
        trainer.load_weights(epoch_number=cfg.resume_from_best)
    # start test
    trainer.test(True, predict_file)


if __name__ == '__main__':
    main()
