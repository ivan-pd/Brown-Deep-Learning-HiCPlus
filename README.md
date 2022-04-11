# HiCPlus 
Ijeoma Meremikwu (imeremik)
Hannah Julius (hjulius)
Elizabeth Wu (ewu32)
Ivan Pineda-Dominguez (ipinedad)

## Introduction: 	
Hi-C Maps are among one of the most popular methods for studying the 3D organization of the genome. Unfortunately, this technology is plagued by high sequencing costs forcing researchers to use low sampled sequencing ultimately resulting in low-resolution data. To solve this problem, Zhang and colleagues explore the use of convolutional neural networks to enhance the resolution of Hi-C data in their paper: Enhancing Hi-C data resolution with deep convolutional neural network HiCPlus. In this paper, Zhang and colleagues were able to devise a CNN model (HiCPlus) that outperformed other currently used data enhancement methods including 2-D Gaussian smoothing. For our project, we intend to implement the HiCPlus model outlined in the paper and replicate the results demonstrated in the paper using a different dataset. We chose to implement this paper due to the intersection of its content with the interests of members in the group. In addition, the simplicity of the HiCPlus model architecture, which consists of only 3 convolution layers, seemed like a worthwhile and feasible deep learning architecture to  implement using the knowledge we have acquired in CS1470. This problem is a supervised learning regression problem, specifically because we are predicting hi-C contact values from noisy inputs.		

# Data:
We will be using Hi-C Data from GM12878 cells. In particular, we will be using a high-resolution (10kb) Hi-C matrix of GM12878 as well as a 10kb matrix of the same cell with sequencing reads down-sampled by 1/16. If time permits, as part of our stretch goal, we intend to use Hi-C data of K562 and IMR90 cells to replicate results from the paper illustrating that the HiCPlus model can enhance Hi-C data of different cell types when using multiple cell datasets for training. 
 
We found Hi-C data collection outside of what is provided in the paper to be quite difficult. For this reason we are extremely thankful to Ghulam Murtaza for sharing the data and preprocessing scripts he has collected and developed as part of his own research with Hi-C data.
 
We expect preprocessing and model training to take a long time due to the nature of the data we are using. As mentioned in the HiCNN paper, training the HiCPlus model takes approximately 28 hours using a NVIDIA V100 GPU. With this in mind we intend to use Brown’s high performance computing cluster OSCAR which one of our team members has priority access to use. 

# Methodology: 
The HiCPlus model architecture consists of a 3 layered ConvNet.
Conv Layer 1: Pattern Extraction and Representation
Input: An NxN low-resolution sample 
16 filters each with size 5x5
ReLU non-linear activation function
Conv Layer 2: Non-linear mapping between the patterns on high-and low-resolution maps
Input: Output from Conv Layer 1
16 filters each with size 1x1
ReLU non-linear activation function
Conv Layer 2: Combining patterns to predict high-resolution maps
Input: Output from Conv Layer 2
1 filter 
Note: Mean square error (MSE) is used as the loss function in the training process along with backpropagation and gradient descent.
 
We will be training our model using Brown’s high performance computing cluster Oscar. We believe that the hardest part of implementing the model will be with regards to image pre-processing our data. In the HiCPlus paper the authors “Divide a Hi-C matrix into multiple square-like sub-regions with fixed size, and each sub-region is treated as one sample”. To replicate this result we expect to spend a significant amount of time figuring out how to divide our Hi-C matrix to provide good inputs to our CNN model. 
 
# Metrics: 
In the HiCPlus paper, Zhang and colleagues implemented multiple different tests to show the utility of a CNN based Hi-C data enhancer. The authors demonstrated that chromatin interactions are predictable from neighboring cells using a CNN, that it is possible to enhance a chromatin interaction matrix with low-sequence depths, that Hi-C interaction matrices can be enhanced using HiCPlus across different cell types, and that its possible to identify chromatin interaction from HiCPlus-enhanced matrices.  For the purpose of this project we will focus on predicting chromatin interaction from neighboring cells and enhancing chromatin interaction matrices with low-sequence depths as our base and target goals respectively. As a stretch goal we hope to use our HiCPlus implementation for Hi-C data enhancement across different cells. However, we predict that our stretch goal will require more time with regards to data collection, preprocessing, and training  that  might not be feasible to complete within our given timeframe. 
 
In order to measure the accuracy of our HiCPlus implementation we plan to implement the same measures of accuracy as outlined in the paper. Primarily, we will be using Pearson and Spearman correlation coefficients between the predicted HiCPlus values and the real values of our high-resolution HiC-Data. As an alternative, we can also use Mean Square Error as a measure of accuracy as shown in the HiCNN paper by Liu and Wang .
