# 1
## 1.1
## 1.2
- underfitting: Take M=0 and M=1 as examples. The red lines don't fit the points good enough, only learn the low-level feature. This means the model is too simple and not suitable for the training data.
- overfitting: Take M=3 and M=9 as examples. The red lines not only fit the points but also learn the feature of noise, which means the model is too complex and can not be suitable for the new data.
## 1.3
- regularization:
    - L1 regularization:
            $L(W)=f(x,W,\widetilde{y})+|W|$
    - L2 regularization: 
            $L(W)=f(x,W,\widetilde{y})+W^TW$
- early stop:
    If the performance of model on validation set becomes bad, stop the training.
- dropout:
    randomly set some elements of weight to 0 according to probability.
- ensemble:
    train multiple models and use the average prediction as the result
# 2
The two major intense operations are convolutional layer and fully connected layer.
- convolutional layer: 
    - describe: use convolution operation to extract feature from input tensor with serval trainable kernel.
    - complexity:
    - advantages:
        - shared weight: reduced the number of parameter and computational complexity
        - extract local feature
    - disadvantages:
        - cannot extract global features
    - when to use: when we need to catch the local features, e.g. to identify the location of face in a picture
- fully connected layer:
    - describe: use a trainable tensor to build a full connection between previous neuron and next neuron.
    - complexity:
    - advantages:
        - it can extract global features from input tensors
        - can learn the linear combination of features
    - disadvantages:
        - it have many parameters. This maybe a challenge for processors and memory.
        - it is very easy to overfit.
    - when to use: when we want to learn the linearly combined feature or classify the image. e.g. to identify which animal this picture belongs to
# 3
## 3.1
preventing overfitting so that the model learn less or noting from noise.
## 3.2
## 3.3
- early stop:
    If the performance of model on validation set becomes bad, stop the training.
- dropout:
    randomly set some elements of weight to 0 according to probability.
- ensemble:
    train multiple models and use the average prediction as the result
# 4
## 4.1
- steps:
    - Normalization: weights in range [-1,+1]
    - Quantization to $\widetilde{w}$ by thresholding {-t,0,t}
    - Trained Quantization:

- how does it affect the backpropogation of the network: it use quantized values in forward progress but use full-precision values in backward propagation, which will introduce some noise.
## 4.2
## 4.3
- affection in accuracy:
    - bad $t$ may result in over-quantization. This will make the accuracy get lower
    - good $t$ can work as a regularization term because it can prune weight with too small values and prevent overfitting
- affection in execution speed:
    -  it can increase sprasity in model and reduce the number of multiplication and addition
