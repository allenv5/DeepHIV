# Folders

- `Datasets` contains all the four datasets used in the experiments.
- `Python` contains the python scripts of AT-HIV.
- `Sample` contains the sample data of training and testing.

# Usage

1. prepare the training and testing datasets by following the format in Sample folder.

2. run `python3 main.py` to execute AT-HIV. Several parameters have to be predetermined.

   `-e`: the size of Embedding layer, default value is 128;

   `-f1`: the filter size of first convolutional layer, default value is 512;

   `-f2`: the filter size of second convolutional layer, default value is 512;

   `-k1`: the kernel size of first convolutional layer, default value is 3;

   `-k2`: the kernel size of second convolutional layer, default value is 5;

   `-c1`: the value of C1;

   `-beta`: the value of beta;

   `-i`: input folder;

   `-cv`: optional parameters; AT-HIV will switch to cross validation mode if provided and the value of this parameter is the number of folds in cross validation. The cross-validation mode will predict the data of `train` in the `Sample` folder with `cv-fold cross-validation` experiments.

   Hence, a complete command to run the `Sample` data is `python3 main.py -c1 8 -beta 2 -i ../Sample -cv 10`.

3. check out the results.txt file in the input folder for the prediction results of `Sample` data.

Node: The codes should be compatible with Python 3.7 and Keras 2.3.1. If you get errors when running the scrips, please try the recommended versions.

   

   

   

   

   