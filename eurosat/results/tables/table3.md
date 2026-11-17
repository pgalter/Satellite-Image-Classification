| Network (ImageNet-pretrained, 80/20) | paper | ours (seed 0) | 95% CI (seed 0) | ours, mean ± std over seeds | last layer only | val acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet-50 | 98.57 | 98.76 | [98.43, 99.02] | 98.86 ± 0.16 (n=3) | 94.33 | 98.47 |
| GoogLeNet | 98.18 | 98.24 | [97.85, 98.56] | 98.41 ± 0.17 (n=3) | 93.67 | 98.19 |

McNemar test, ResNet-50 vs GoogLeNet on the same 5,400 test images (seed 0): 60 images only ResNet-50 gets right, 32 only GoogLeNet; exact p = 0.005.
