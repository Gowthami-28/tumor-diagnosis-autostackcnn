# tumor-diagnosis-autostackcnn
# Brain Tumor Diagnosis using Auto Stack CNN

## Results
| Model | Accuracy |
|-------|----------|
| Custom CNN | 96.76% |
| MobileNet | 96.11% |
| ResNet50 | 75.04% |
| VGG16 | 74.23% |
| **Auto Stack CNN** | **98.45%**  |

## Dataset
- Br35H Brain Tumor Detection Dataset
- 2470 training images
- 617 validation images
- 2 classes: Tumor / No Tumor

## Models Used
- Custom CNN (built from scratch)
- MobileNet (transfer learning)
- ResNet50 (transfer learning)
- VGG16 (transfer learning)
- Auto Stack Ensemble (weighted combination)

## Key Finding
Auto Stack CNN (98.45%) outperformed all individual models
by using weighted ensemble with shuffle=False for correct
label alignment.

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Google Colab

