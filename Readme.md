# MNIST
## Test project for image classification.
### Requirements
- Python 3.10 supported.
- Pytorch 1.13.1 is recomnended.
- Please check the python package requirement from [`requirements.txt`](requirements.txt), and install using
```
pip install requirements.txt
```
### Dataset
The project used the [`MNIST`](https://github.com/myleott/mnist_png) dataset.
### 1. Create csv file for dataset.
```
python3 dataset/create_csv.py \
--train_folder [path-to-train-image-folder] \
--test_folder [path-to-test-image-folder] \
--output [path-to-save-folder]

```
### 2. Training.
```
python3 train.py \
--train_csv [path-to-train-image-csv] \
--test_csv [path-to-test-image-csv] \
Optional params
--ckp_dir [path-to-checkpoint-dir][default-weights] \
--ckp_interval [checkpoint-interval][default-1] \
--train_batch [train-batch-size][default-1000]
--test_batch [test-batch-size][default-1000]
--lr [learning-rate][default-0.0001]
--epochs [num-epochs][default-25]
--weight-decay [optimizer-weight-decay][default-1e-5]
```
### 3. Test.
Test and write results.
```
python3 test.py \
--test_csv [path-to-test-image-csv] \
Optional
--test_batch [test-batch-size][default-1000] \
--save_predict_path [path-to-save-predict-csv-file][default-prediction]\
--best_weights [What-weights-to-use-for-the-test][default-best-weights]
```
### 4. Evaluate.
Calculation accuracy and confusion matrix using  sikit-learn instruments.
```
--prediction_csv [path-to-prediction-csv-file]
```
### Vizualize trainig results.
```commandline
tensorboard --logdir runs
```