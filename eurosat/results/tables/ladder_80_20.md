| Method (80/20 split, seed 0) | test acc | val acc | macro-F1 | paper | minutes |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ResNet-50, ImageNet, fine-tuned (224 px) | 98.76 | 98.47 | 98.72 | 98.57 | 24.3 |
| GoogLeNet, ImageNet, fine-tuned (224 px) | 98.24 | 98.19 | 98.17 | 98.18 | 11.3 |
| ResNet-50, from scratch (128 px) | 98.09 | 97.92 | 98.04 | 96.43 | 13.3 |
| GoogLeNet, from scratch (128 px) | 98.00 | 97.92 | 97.95 | 96.02 | 14.9 |
| CNN, two conv layers (64 px) | 94.70 | 94.49 | 94.50 | 87.96 | 1.0 |
| ResNet-50, ImageNet, last layer only (224 px) | 94.33 | – | – | – | – |
| BoVW (SIFT, k=500) + SVM | 71.69 | 72.69 | 70.53 | 70.05 | 11.4 |
| BoVW (SIFT, k=100) + SVM | 68.69 | 68.47 | 67.43 | 67.22 | 2.3 |
| BoVW (SIFT, k=10) + SVM | 55.13 | 56.57 | 53.22 | 58.55 | 4.2 |
