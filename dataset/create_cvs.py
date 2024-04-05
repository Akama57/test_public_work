import csv
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List


def parse_args():
    """ Parse command line params."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_folder',
        type=Path,
        required=True,
        help='Path to train dataset.'
    )
    parser.add_argument(
        '--test_folder',
        type=Path,
        required=True,
        help='Path to test dataset.'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to save csv files.'
    )
    return parser.parse_args()


def to_csv_file(paths_data: List[Path], name: Path):
    """
    Save data to csv file.
    :param paths_data: path to any files;
    :param name: save path for csv file.

    """

    field_names = ['ID', 'PATH', 'CLASS_ID']

    with open(name, 'w') as file:
        writer_object = csv.DictWriter(file, fieldnames=field_names)
        for item, data in enumerate(tqdm(paths_data)):
            wdict = {'ID': item, 'PATH': str(data), 'CLASS_ID': int(data.parent.name)}
            writer_object.writerow(wdict)

    print(f'\nFile {name} is created!')


def main():
    """ Main function. """
    args = parse_args()
    # extension for files
    extensions = ['.jpg', '.png']
    # create output dir
    output_dir = args.output if args.output.is_dir() else args.output.mkdir(parents=True, exist_ok=True)

    paths_data = [args.train_folder,
                  args.test_folder]
    csv_file_names = [Path(output_dir, 'train.csv'),
                      Path(output_dir, 'test.csv')]

    for data, name in zip(paths_data, csv_file_names):
        p = data.glob('**/*')
        files = [x for x in p if x.suffix in extensions]
        to_csv_file(files, name)


if __name__ == '__main__':
    main()
