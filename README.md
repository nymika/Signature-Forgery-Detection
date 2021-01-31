# Signature Forgery Detection
Signature samples are classified as forged or genuine using Convolutional Neural Networks.

## [](https://github.com/nymika/Signature-Forgery-Detection#models-used)Models Used

1.  **Basic CNN model**:  [conventional-sign-forgery-detect.ipynb](https://github.com/nymika/Signature-Forgery-Detection/blob/master/conventional-sign-forgery-detect.ipynb)  contains training for the cnn model. The model contains two Convolution Layers, two Pooling layers, two Dense layers and one dropout layer to reduce overfitting. The results obtained shows the validation loss does not increase and that there is no overfitting.
2. **VGG16 model**:  [transfer-learning-sign-forgery-detect.ipynb](https://github.com/nymika/Signature-Forgery-Detection/blob/master/transfer-learning-sign-forgery-detect.ipynb)  contains implemetation of VGG16 model pretrained on imagenet dataset. In this model all the layers were trained. It is implemented using Tensorflow Keras api. It is observed that there is early stopping in 16th epoch.
3. **Siamese Network for One shot learning**:  [siamese-model-one-shot-learning.ipynb](https://github.com/nymika/Signature-Forgery-Detection/blob/master/siamese-model-one-shot-learning.ipynb) contains implementation of siamese network on CNN. This model, given only one true signature of a person as reference determines whether the query signature belongs to the same person or is a forgery. We observe that the loss has decreased in every epoch. Hence, there is no overfitting. It's a well trained model. 

## [](https://github.com/nymika/Signature-Forgery-Detection#dataset)Dataset

The dataset used was taken from kaggle. This is the link:  [https://www.kaggle.com/robinreni/signature-verification-dataset](https://www.kaggle.com/robinreni/signature-verification-dataset)

## [](https://github.com/nymika/Signature-Forgery-Detection#requirenments)Requirements

-   Python 3
-   Matplotlib
-   Numpy
-   Tensorflow
-   Keras
-   Pandas

## [](https://github.com/nymika/Signature-Forgery-Detection#setup)Setup

 - Run the conventional-sign-forgery-detect.ipynb, transfer-learning-sign-forgery-detect.ipynb, siamese-model-one-shot-learning.ipynb files individually.

## [](https://github.com/nymika/Signature-Forgery-Detection#results)Results
The plotted accuracy and loss curves are in the [results folder.](https://github.com/nymika/Signature-Forgery-Detection/tree/master/Results)

| Model | Training Accuracy|Valiation Accuracy|
|--|--|--|
| CNN model  | 94.54% | 93.94% |
| VGG16 model  | 100% | 99.8%|
| Siamese model  | 100% | 100% |

## [](https://github.com/nymika/Signature-Forgery-Detection#conclusion) Conclusion
Conventional deep learning methods require large samples of data for a class in the classification process. One shot learning, being a method of meta learning, can perform classification tasks with one data point. Hence for the tasks like facial recognition, audio recognition, signature forgery verification this method of meta learning becomes more suitable at industrial level.




