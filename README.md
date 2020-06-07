# feature-selection-extraction
Feature selection and feature extraction code implementation
##### Paper: 
Feature Selection and Feature Extraction in Pattern Analysis: A Literature Review 
##### Link: 
https://arxiv.org/abs/1905.02845

Aim of the paper is to compare commonly used feature selection and feature extraction methods on a single task - digit recogtion.

The methods compared in the paper are:

## Feature Selection
1. Filter Methods
  1. Correlation Criteria
  2. Mutual Information
  3. Chi Square Statistics
  4. Markov Blanket
  5. Consistency Based Filter
  6. Fast Correlation-Based Filter
  7. INTERACT
  8. Minimal Redundancy Maximal Relevance
  
2. Wrapper Methods
  1. Sequential Forward Selection 
  2. Sequential Backward Selection
  3. Particle Swarm Optimization
  4. Genetic Algorithms
  5. Geometric PSO
  
## Feature Extraction
1. Unsupervised Methods
  1. PCA
  2. Dual PCA
  3. Kernel PCA
  4. Multidimensional Scaling
  5. Isomap
  6. Locally Linear Embedding
  7. Laplacian Eigenmap
  8. Maximum Variance Unfolding
  9. Autoencoders and neural networks
  10. tSNE
  
2. Supervised Methods
  1. Fisher LDA
  2. Kernel FLDA
  3. Supervised PCA
  4. Metric Learning

Many of the above mentioned methods are implemented in various Python packages:
### sklearn.manifold
LocallyLinearEmbedding, Isomap, MDS,  SpectralEmbedding(LaplacianEigenmap), TSNE
### sklearn.decomposition 
PCA, KernelPCA
### sklearn.discriminant_analysis
LDA

WEKA was also used. It is an open source software with implementations of feature selection methods like INTERACT.

Few of the methods were adapted from publicly uploaded code on GitHub or other sources.

Rest are implemented by authors of the paper.

Plotting code outputs images as the example below (Autoencoder feature extraction method):

![Embedded Space Plot](/AE_plot.png)
Format: ![Alt Text](url)
