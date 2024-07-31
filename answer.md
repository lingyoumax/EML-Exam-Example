# 1
## 1.1
## 1.2
- underfitting: Take M=0 and M=1 as examples. The red lines don't fit the points good enough, only learn the low-level feature. This means the model is too simple and not suitable for the training data.
- overfitting: Take M=9 as an example. The red line not only fit the points but also learn the feature of noise, which means the model is too complex and can not be suitable for the new data.
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
        - use multiple kernels to extrat different features
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
        - it is very easy to overfit due to the large number of parameters
    - when to use: when we want to learn the linearly combined feature or classify the image. e.g. to identify which animal this picture belongs to
# 3
## 3.1
lower the complexity of model and prevent it from overfitting so that the model learn less or noting from noise.
## 3.2
- differences: 
    - L1 Regularization (Lasso): Adds a penalty equal to the absolute value of the magnitude of coefficients. This can lead to sparse solutions where some feature weights can become exactly zero, thus performing feature selection.
    - L2 Regularization (Ridge): Adds a penalty equal to the square of the magnitude of coefficients. This tends to distribute the error among all the terms, but it doesn’t necessarily eliminate any weights, as it only minimizes their effect.
- trade-off:
    - Sparsity: L1 regularization can produce sparse models that are simpler and interpretable but might miss some important predictors that have small but significant effects. L2 regularization typically results in models where all coefficients are shrunk towards zero but rarely exactly zero.
    - Stability: L2 is generally less sensitive to outliers than L1 because the penalty exponential growth in large coefficients is more controlled.
    - Computation: Solving L1 regularized problems can be more challenging than L2, as the objective function in L1 is not differentiable, requiring more sophisticated optimization algorithms.
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

- how does it affect the backpropogation of the network: 
    - it use quantized values in forward progress but use full-precision values in backward propagation, which will introduce some noise.
    - The primary effect of TTQ on backpropagation is through the "gradient masking" where gradients for weights quantized to zero are not updated. This can lead to sparsity in the gradient updates, which might help in regularizing the model and focusing updates on more significant weights.
## 4.2
The quantization typically occurs within the network's layers that handle the majority of computational operations, such as the fully connected layers and convolutional layers. The primary reason for quantizing within these layers is that they generally consume the most memory and computational resources due to their large number of parameters and the complexity of their operations.
## 4.3
- Higher t values may result in more weights being quantized to zero, which increases sparsity and potentially enhances execution speed due to fewer operations. However, too high a sparsity might lead to loss of information and decreased accuracy.
- Lower t values decrease the level of sparsity, maintaining more information in the network but with reduced computational savings.
# 5
## 5.1
## 5.2
- Model Efficiency: By allowing multiple filter sizes to operate at the same level, Inception blocks can efficiently learn appropriate features at multiple scales, making them more adaptable to various input types without significantly increasing computational requirements.
- Improved Performance: Networks utilizing Inception blocks can outperform those with a traditional stacking of convolutions due to their ability to capture a wider range of features from the input images.
- Reduced Overfitting: The parallel structure of Inception blocks helps in regularizing the model. The presence of pooling and convolutions side by side helps in capturing essential features without focusing too much on high-level details that might lead to overfitting.
- Scalability and Depth: The modular nature of the Inception blocks allows networks to deepen without a substantial increase in computational complexity, thanks to the dimensionality reduction done by 1x1 convolutions.
## 5.3
- Functionality:
    - Dimensionality Reduction: Before performing 3x3 or 5x5 convolutions, which are computationally expensive, 1x1 convolutions are used to reduce the depth of the input volume, thereby reducing the number of operations required in subsequent layers.
    - Channel Pooling: 1x1 convolutions can be seen as a form of channel pooling, where they combine features from the depth dimension, allowing the network to create a compressed feature space representation.
    - Increasing Non-linearity: By incorporating non-linear activation functions, 1x1 convolutions introduce additional non-linear properties to the decision function without affecting the receptive fields of the convolutions.
- Significance:
    - Computational Efficiency: They significantly reduce the computational cost by limiting the increase of dimensionality throughout the network layers.
    - Feature Reinforcement and Combination: 1x1 convolutions help in mixing features across the channel dimension before they are fed into larger convolutions. This pre-processing can help in enhancing the relevancy of features extracted by larger convolutional kernels.
# 6
- Explain: A grouped convolution is convoluted separately with g gropus of M'=M/g filter.
    - its output is the concatenation of all groups output along the channel axis
    - its input is spilt to g groups along the channel axis
- advantages:
    - easy to parallelize
    - save parameter
    - reduce computational complexity
- disadvantages:
    - may prune the correlated relationship between features
- cost metrics of grouped convolution
    - MACs: $g*((H-R+1)*(W-S+1)*(C/g)*R*S*(M/g))=(H-R+1)*(W-S+1)*R*S*C/g$
    - Parameters: 
        - Weight: $g*(R*S*C/g*M/g)=R*S*C*M/g$
        - Biases: $g*M/g=M$
    - Activations: $g*(H-R+1)*(W-S+1)*M/g=(H-R+1)*(W-S+1)*M$
- cost metrics of non-grouped convolution
    - MACs: $(H-R+1)*(W-S+1)*R*S*C$
    - Parameters: 
        - Weight: $R*S*C*M$
        - Biases: $M$
    - Activations: $(H-R+1)*(W-S+1)*M$
# 7
## 7.1
- data collection:
    - collect as much flower images as possible, so that the model can easily learn the features
    - make the number of each classes are almost equal, so that the model have no perference on one class
- preprocessing:
    - remove the images contained different classes of flower
    - resize the image to 64x64 pixels, so that the size is more suitable for cache and memory
    - normalize the values of images to make them ranging from 0 to 1
- augmentation
    - flip
    - crop
    - rotate
    - noise
## 7.2
# 8
