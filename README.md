# 🌟 Kolmogorov-Arnold Theory Inspired Networks for Realistic Blind Image Sharpness Assessment

This repository contains the implementation of Kolmogorov-Arnold Networks (KANs) for realistic blind image sharpness assessment (BISA). In this work, we explore the application of KANs, inspired by the Kolmogorov-Arnold theorem, for score regression tasks in image quality assessment, particularly focusing on image sharpness without reference images.

## 📄 Overview

### 🔍 Abstract

Score prediction is crucial in blind image sharpness assessment after quantitative features are collected. Inspired by the Kolmogorov-Arnold theorem, we developed the Kolmogorov-Arnold Network (KAN), which has shown significant success in function fitting. In this study, we present a Taylor series-based KAN model (TaylorKAN) and explore various KAN models on four realistic image databases: BID2011, CID2013, CLIVE, and KonIQ-10k. Using 15 mid-level and 2048 high-level features, and setting support vector regression as a baseline, our results demonstrate that TaylorKAN and other KAN models outperform or are competitive with existing methods, although high-level features yield inferior performance on the CLIVE dataset. This is the first study to explore KAN models for blind image quality assessment, providing insights into the selection and improvement of KAN models for related numerical regression tasks.

### 🔑 Keywords

- Kolmogorov-Arnold Theory
- Blind Image Sharpness Assessment
- Machine Learning
- Image Quality Assessment

## 🧠 Model Architecture

### Kolmogorov-Arnold Network (KAN)

The KAN architecture is rooted in the Kolmogorov-Arnold theorem, which posits that any continuous multivariate function can be expressed as a finite sum of continuous univariate functions. This property allows KANs to decompose high-dimensional problems into multiple one-dimensional problems, offering a powerful approach for function approximation and physical problem-solving.

### KAN Variants

- **TaylorKAN**: Based on the Taylor series expansion, this model enhances the network's capacity to model non-linear relationships by learning expansion coefficients during the training process.
- **BSRBF-KAN**: Combines B-spline (BS) and radial basis function (RBF) for smoothness, continuity, and powerful interpolation.
- **ChebyKAN**: Utilizes Chebyshev polynomials, known for minimizing maximum error in polynomial approximation, making it suitable for high-accuracy tasks.
- **HermiteKAN**: Employs Hermite polynomials, ideal for approximating Gaussian-like functions.
- **JacobiKAN**: Leverages Jacobi polynomials, which are orthogonal and flexible in handling diverse boundary conditions.
- **WavKAN**: Uses wavelets, particularly Mexican Hat wavelets, for capturing localized features and variations in functions.

## 📊 Performance Summary

### Mid-Level Features

| Dataset   | Model       | PLCC | SRCC | Dataset   | Model       | PLCC | SRCC |
|-----------|-------------|------|------|-----------|-------------|------|------|
| BID2011   | MLP         | 0.589| 0.588| CID2013   | MLP         | 0.839| 0.835|
|           | SVR         | 0.564| 0.561|           | SVR         | 0.831| 0.822|
|           | BSRBF KAN   | 0.697| 0.705|           | BSRBF KAN   | 0.833| 0.802|
|           | ChebyKAN    | 0.711| 0.769|           | ChebyKAN    | 0.753| 0.792|
|           | HermiteKAN  | 0.721| **0.779**|           | HermiteKAN  | 0.815| 0.822|
|           | JacobiKAN   | 0.719| 0.778|       | JacobiKAN   | 0.792| 0.830|
|           | WavKAN      | 0.655| 0.685|           | WavKAN      | 0.850| 0.824|
|           | TaylorKAN   | **0.757**| 0.750|             | TaylorKAN   | **0.859**| **0.843**|
|-----------|-------------|------|------|-----------|-------------|------|------|
| CLIVE     | MLP         | 0.572| 0.542| KonIQ-10k | MLP         | 0.752| 0.721|
|           | SVR         | 0.589| **0.561**|       | SVR         | **0.764**| **0.732**|
|           | BSRBF KAN   | 0.601| 0.514|           | BSRBF KAN   | 0.742| 0.667|
|           | ChebyKAN    | 0.522| 0.454|           | ChebyKAN    | 0.752| 0.675|
|           | HermiteKAN  | 0.581| 0.522|           | HermiteKAN  | 0.747| 0.678|
|           | JacobiKAN   | 0.572| 0.475|           | JacobiKAN   | 0.749| 0.676|
|           | WavKAN      | 0.606| 0.498|           | WavKAN      | 0.762| 0.690|
|           | TaylorKAN   | **0.608**| 0.488|             | TaylorKAN   | 0.719| 0.640|

### Deeply Learned Features

| Dataset   | Model       | PLCC | SRCC | Dataset   | Model       | PLCC | SRCC |
|-----------|-------------|------|------|-----------|-------------|------|------|
| BID2011   | MLP         | 0.746| 0.725| CID2013   | MLP         | **0.925**| **0.909**|
|           | SVR         | 0.774| 0.752|           | SVR         | 0.924| 0.906|
|           | BSRBF KAN   | 0.818| 0.813|           | BSRBF KAN   | 0.858| 0.861|
|           | ChebyKAN    | 0.809| 0.816|           | ChebyKAN    | 0.525| 0.527|
|           | HermiteKAN  | 0.825| 0.831|           | HermiteKAN  | 0.636| 0.635|
|           | JacobiKAN   | **0.826**| **0.834**|           | JacobiKAN   | 0.683| 0.685|
|           | WavKAN      | 0.797| 0.811|           | WavKAN      | 0.839| 0.850|
|           | TaylorKAN   | 0.792| 0.788|           | TaylorKAN   | 0.750| 0.756|
|-----------|-------------|------|------|-----------|-------------|------|------|
| CLIVE     | MLP         | 0.465| 0.440| KonIQ-10k | MLP         | 0.607| 0.576|
|           | SVR         | 0.733| **0.724**|           | SVR         | 0.803| 0.783|
|           | BSRBF KAN   | 0.737| 0.663|           | BSRBF KAN   | 0.840| 0.807|
|           | ChebyKAN    | 0.683| 0.545|           | ChebyKAN    | 0.838| 0.804|
|           | HermiteKAN  | 0.741| 0.704|           | HermiteKAN  | 0.844| 0.810|
|           | JacobiKAN   | 0.747| 0.694|           | JacobiKAN   | 0.842| 0.809|
|           | WavKAN      | **0.761**| 0.687|           | WavKAN      | **0.848**| **0.817**|
|           | TaylorKAN   | 0.667| 0.563|           | TaylorKAN   | 0.831| 0.801|

## 🚀 Conclusion

Our study illustrates that KAN models, particularly TaylorKAN, demonstrate strong potential in blind image sharpness assessment tasks. While these models have shown competitive performance in several cases, especially with mid-level features, the results also reveal areas where further refinement is necessary. For instance, high-level features did not perform as well on certain datasets such as CLIVE, indicating that there is still room for improvement in feature selection and model adaptation. Future work should focus on addressing these challenges to enhance the robustness and applicability of KAN models across a broader range of datasets.

