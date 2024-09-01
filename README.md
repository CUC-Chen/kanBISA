# Kolmogorov-Arnold Theory Inspired Networks for Realistic Blind Image Sharpness Assessment

This repository contains the implementation of Kolmogorov-Arnold Networks (KANs) for realistic blind image sharpness assessment (BISA). The work explores the application of KANs, inspired by the Kolmogorov-Arnold theorem, for score regression tasks in image quality assessment, particularly focusing on image sharpness without reference images.

## Overview

### Abstract

Score prediction is pivotal in blind image sharpness assessment after quantitative features are collected. Inspired by the Kolmogorov-Arnold theorem, the Kolmogorov-Arnold Network (KAN) is developed and has shown significant success in function fitting. This work presents a Taylor series-based KAN model (TaylorKAN) and explores various KAN models on four realistic image databases: BID2011, CID2013, CLIVE, and KonIQ-10k. Using 15 mid-level and 2048 high-level features, and setting support vector regression as a baseline, the results demonstrate that TaylorKAN and other KAN models outperform or are competitive with existing methods, although high-level features yield inferior performance on the CLIVE dataset. This study is the first to explore KAN models for blind image quality assessment, providing insights into the selection and improvement of KAN models for related numerical regression tasks.

### Keywords

- Kolmogorov-Arnold Theory
- Blind Image Sharpness Assessment
- Machine Learning
- Image Quality Assessment

## Model Architecture

### Kolmogorov-Arnold Network (KAN)

The KAN architecture is rooted in the Kolmogorov-Arnold theorem, which posits that any continuous multivariate function can be expressed as a finite sum of continuous univariate functions. This property allows KANs to decompose high-dimensional problems into multiple one-dimensional problems, offering a powerful approach for function approximation and physical problem-solving.

### KAN Variants

- **TaylorKAN**: Based on the Taylor series expansion, this model enhances the network's capacity to model non-linear relationships by learning expansion coefficients during the training process.
- **BSRBF-KAN**: Combines B-spline (BS) and radial basis function (RBF) for smoothness, continuity, and powerful interpolation.
- **ChebyKAN**: Utilizes Chebyshev polynomials, known for minimizing maximum error in polynomial approximation, making it suitable for high-accuracy tasks.
- **HermiteKAN**: Employs Hermite polynomials, ideal for approximating Gaussian-like functions.
- **JacobiKAN**: Leverages Jacobi polynomials, which are orthogonal and flexible in handling diverse boundary conditions.
- **WavKAN**: Uses wavelets, particularly Mexican Hat wavelets, for capturing localized features and variations in functions.

## Datasets

The models were tested on four well-known image quality assessment databases:

1. **BID2011**: Contains 586 images, score range [0, 5]
2. **CID2013**: Contains 474 images, score range [0, 100]
3. **CLIVE**: Contains 1,169 images, score range [0, 100]
4. **KonIQ-10k**: Contains 10,073 images, score range [0, 5]

## Feature Extraction

Two sets of features were prepared:

- **Mid-Level Features**: 15 features per image derived from various BISA indicators.
- **High-Level Features**: 2048 features per image extracted from the last fully connected layer of ResNet50, pre-trained on ImageNet.

## Performance Metrics

Performance is evaluated using:

- **Pearson Linear Correlation Coefficient (PLCC)**
- **Spearman Rank Order Correlation Coefficient (SRCC)**

Higher values indicate better performance.

## Results

- **Mid-Level Features**: KAN models generally outperform SVR on most datasets, with TaylorKAN leading in most cases.
- **High-Level Features**: KAN models surpass SVR on BID2011 and KonIQ-10k but show inferior results on CID2013 and CLIVE.

## Conclusion

The study demonstrates that KAN models, particularly TaylorKAN, can be highly effective for blind image sharpness assessment. While these models show promise, further research is needed to address the challenges presented by datasets like CLIVE and to explore additional feature selection methods and learning strategies.
