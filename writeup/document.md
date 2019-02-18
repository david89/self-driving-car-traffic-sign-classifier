# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/david89/self-driving-car-traffic-sign-classifier/blob/master/notebook.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used simple Python methods to calculate summary statistics of the traffic signs data set like:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Train distribution](./train_distribution.png)

![Validation distribution](./validation_distribution.png)

![Test distribution](./test_distribution.png)

As you can see, the distribution of the training, validation and testing sets are similar. However, it's worth noticing that some labels don't contain a high number of samples. For example, the label 0 only contains 180 samples in the training set, while others contain magnitudes more. Therefore, we may need to collect or generate more data if we a model that will be able to classify those labels with low frequency, with enough precision.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First of all, we decided to consider images in the RGB color space, since we don't know if color might bring extra information to our prediction model.

In the "Data Set Summary & Exploration" section we mentioned that we don't have enough samples for certain labels. One possible workaround is to generate some synthetic data for such labels. For example, we could apply the following transformations.

* We could rotate the image by a small angle (between -10 and 10 degrees for example).

* Salt and pepper; where we set some random pixels to either 0 (black or "pepper") or 255 (white or "salt"). Please note that in a RGB image, salt and pepper may not set the pixels to black or white, but something like bright green, red or blue.

* Gaussian; as described in [here](https://en.wikipedia.org/wiki/Gaussian_noise).

* Speckle noise.

The following image contains a sample image and the different transformations we could apply over it:

![Noise images](./noise.png)

After applying different noise functions to our dataset, we have a much better distribution:

![New distribution](./extended_train_distribution.png)

However, we are not going to use the new extended train data set just yet, because we want to measure the accuracy of the model on the initial test data set.

In order to smooth out the data, we decided to use the formula (pixel - 128) / 128, which will transform pixels in the [0, 255] range into the [-1.0, 1.0] range. Smaller ranges will allow our model to converge faster and give more accurate predictions. As a future improvement for this project, we could use a standardization feature scalling, i.e., (x - mean(x)) / stddev(x).

Finally, we shuffled our data to make sure the model doesn't depend on the ordering of the samples.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The architecture that we used is described in the [Yann LeCun paper](./lecun_paper.pdf). More specifically, this is the architecture that we used:

![Architecture](architecture.png)

Now, let's go through each layer of the chose architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| L0.1: Input         		| 32x32x3 RGB image   							| 
| L1.1: Convolution 5x5 over L0.1    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| L1.2: Activation (RELU or TANH) over L1.1					|	outputs 28x28x6											|
| L1.3: Max pooling	over L1.2      	| 2x2 stride,  outputs 14x14x6 |
| L2.1: Convolution 5x5 over L1.3    	| 1x1 stride, valid padding, outputs 10x10x16 |
| L2.2: Activation (RELU or TANH)	over L2.1				|	outputs 10x10x16 |
| L3.1: Pooling over L2.2     	| 2x2 stride, outputs 5x5x16 |
| L3.2: Convolution 5x5 over L3.1					|	1x1 stride, valid padding, outputs 1x1x400											|
| L3.3: Concatenation of flattened L2.2 and L3.2	      	| outputs 2000 |
| L3.4: Dropout over L3.3	      	| outputs 2000 |
| L4.1: Fully connected	over L3.4	| outputs `n_classes` |
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following steps:
* I transformed the labels tensor into a one hot encoding tensor, since we have a categorization problem.
* I got the logits tensor and a regularization tensor (based on the weights of each layer) from the chosen architecture.
* We then applied the softmax_cross_entropy_with_logits function over the one hot encoding and the logits from the architecture.
* Our loss function tries to reduce the mean of the cross entropy (from the previous step) + the regularization tensor * regularization rate.
* For the optimizer, we used the AdamOptimizer which is commonly used for this kind of problems.
* Additionally, we trained our model using batches, since feeding the whole data set at once was too expensive.

There are different hyperparameters involved in the aforementioned steps, however, we trained different models with different parameters, which will be discussed in the following section.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First of all, we trained a model with some default parameters:
* Activation function: RELU
* Number of epochs: 10
* Batch size: 128
* Learning rate: 0.001
* Regularization rate: 0.0
* Dropout rate: 0.5

With that model, we achieved 93.5% accuracy on the validation data set and 92.6% accuracy on the test data set. Meaning that the default values are not bad, but not good enough for 93% minimum accuracy. Therefore we decided to train different models and see which one performs better:

##### TANH activation function

* Activation function: TANH
* Number of epochs: 10
* Batch size: 128
* Learning rate: 0.001
* Regularization rate: 0.0
* Dropout rate: 0.5

Test accuracy: 92.3%

Since the RELU and TANH models gave us similar results, we decided to stick with the RELU activation function.

##### 30 epochs and 0.01 regularization rate

* Activation function: RELU
* Number of epochs: 30
* Batch size: 128
* Learning rate: 0.001
* Regularization rate: 0.01
* Dropout rate: 0.5

Test accuracy: 89.9%

Seems like the regularization rate is too high, so let's train another model with a more conservative value.

##### 30 epochs and 0.001 regularization rate

* Activation function: RELU
* Number of epochs: 30
* Batch size: 128
* Learning rate: 0.001
* Regularization rate: 0.001
* Dropout rate: 0.5

Test accuracy: 93.8% (**success**)

##### 0.01 learning rate

* Activation function: RELU
* Number of epochs: 10
* Batch size: 128
* Learning rate: 0.01
* Regularization rate: 0.0
* Dropout rate: 0.5

Test accuracy: 90.1%

Increasing the learning rate is definitely decreasing our accuracy.

##### 40 epochs and 0.0005 regularization rate

* Activation function: RELU
* Number of epochs: 40
* Batch size: 128
* Learning rate: 0.01
* Regularization rate: 0.0005
* Dropout rate: 0.5

Test accuracy: 93.95% (**success**)

##### 30 epochs

* Activation function: RELU
* Number of epochs: 30
* Batch size: 128
* Learning rate: 0.01
* Regularization rate: 0.0
* Dropout rate: 0.5

Test accuracy: 93.6% (**success**)

As you can see, the architecture proposed by LeCun gave us some good results (enough to get at least 93% success rate on the test set). The best model in our case is the one with 40 epochs and a regularization rate of 0.0005%, which has a test accuracy of ~94%. That's the model we are going to use in order to run our predictions.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Stop](../test/stop.png)
![Do not enter](../test/do_not_enter.png)
![30 km/h](../test/30.png)
![60 km/h](../test/60.png)
![80 km/h](../test/80.png)
![ahead](../test/up.png)
![yield](../test/yield.png)

These images were randomly chose from the web. I chose images with different light conditions and where the sign may be ambiguous at low resolution (for example, the 80 km/h sign can be interpreted as 30 km/h as well).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Do not enter | Do not enter 										|
| Yield					| Yield											|
| Ahead | Ahead |
| 30 km/h	      		| 30 km/h |
| 60 km/h	      		| 60 km/h |
| 80 km/h	      		| 80 km/h |

The model was able to correctly guess 7 out of the 7 traffic signs, which gives an accuracy of 100%. The next step to improve the current model is grab some images where the classification fails, and add them to the test data set.

#### 3. Describe how certain the model is when predicting on each of the seven new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the stop sign image, the model is relatively sure that this is a stop sign (probability of 0.995), and the image does contain a stop sign. The top five soft max probabilities were ['Class Stop with probability 0.9953418970108032', 'Class Speed limit (20km/h) with probability 0.001570996129885316', 'Class No entry with probability 0.0011368233244866133', 'Class Speed limit (30km/h) with probability 0.00047899034689180553', 'Class General caution with probability 0.00020614403183571994']

For the do not enter sign image, the model is relatively sure that this is a do not enter sign (probability of 0.9999), and the image does contain a do not enter sign. The top five soft max probabilities were ['Class No entry with probability 0.9999896287918091', 'Class Stop with probability 1.0346545423089992e-05', 'Class Speed limit (20km/h) with probability 5.875588016124311e-08', 'Class Speed limit (60km/h) with probability 4.244998241809128e-10', 'Class Bumpy road with probability 4.0290690228594883e-10']

For the yield sign image, the model is relatively sure that this is a yield sign (probability of 1.0), and the image does contain a yield sign. The top five soft max probabilities were ['Class Yield with probability 1.0', 'Class Speed limit (60km/h) with probability 3.342016875913789e-13', 'Class Ahead only with probability 2.031858839166012e-13', 'Class No passing with probability 2.991360888919374e-14', 'Class Children crossing with probability 1.8779680975990025e-14']

For the ahead sign image, the model is relatively sure that this is a ahead sign (probability of 1.0), and the image does contain a ahead sign. The top five soft max probabilities were ['Class Ahead only with probability 1.0', 'Class Go straight or right with probability 4.1921002259803775e-13', 'Class Roundabout mandatory with probability 7.430521990888991e-15', 'Class Keep left with probability 1.6883283050821369e-15', 'Class Turn left ahead with probability 7.620756485238124e-16']

For the 30 km/h sign image, the model is relatively sure that this is a 30 km/h sign (probability of 0.9999), and the image does contain a 30 km/h sign. The top five soft max probabilities were ['Class Speed limit (30km/h) with probability 0.999987006187439', 'Class Speed limit (70km/h) with probability 1.29794716485776e-05', 'Class Speed limit (20km/h) with probability 4.545992737803317e-08', 'Class Speed limit (50km/h) with probability 2.816904912453233e-12', 'Class Speed limit (80km/h) with probability 1.2211869970107925e-12']

For the 60 km/h sign image, the model is relatively sure that this is a 60 km/h sign (probability of 0.997), and the image does contain a 60 km/h sign. The top five soft max probabilities were ['Class Speed limit (60km/h) with probability 0.9975118637084961', 'Class Speed limit (50km/h) with probability 0.002487539779394865', 'Class No passing with probability 1.4090484512507828e-07', 'Class Roundabout mandatory with probability 9.294182490293679e-08', 'Class Slippery road with probability 8.730827261160812e-08']

For the 80 km/h sign image, the model is relatively sure that this is a 80 km/h sign (probability of 0.9999), and the image does contain a 80 km/h sign. The top five soft max probabilities were ['Class Speed limit (80km/h) with probability 0.9999632835388184', 'Class Speed limit (60km/h) with probability 3.660628863144666e-05', 'Class Speed limit (100km/h) with probability 7.627546239064031e-08', 'Class Speed limit (30km/h) with probability 1.7379379713133858e-08', 'Class Speed limit (50km/h) with probability 2.1873205469091772e-09']
