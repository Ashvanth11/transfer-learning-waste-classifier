# Efficient Transfer Learning for Multi-Class Image Classification

This project implements transfer learning techniques using multiple pre-trained CNN architectures to classify images of nine types of waste. The goal is to evaluate and compare the performance of these models on a real-world image dataset using Keras.

---

## Dataset

[RealWaste Dataset - UCI Repository](https://archive.ics.uci.edu/dataset/908/realwaste)

The dataset consists of images grouped into **nine waste categories** (e.g., cardboard, glass, metal, paper, plastic, trash, etc.). Images are stored in separate folders per class.

### Key Notes:
- Around 400–500 images per class (varies slightly).
- Images have varying sizes and were resized/padded for uniformity.
- Dataset is not uploaded to this repo due to size limits.
- **To use the dataset:**
  - Manually download or request access (if applicable).
  - Place folder structure as:
    ```
    dataset/
      ├── cardboard/
      ├── glass/
      ├── metal/
      └── ...
    ```

---

## Models Used

Transfer learning was applied using the following pre-trained CNNs:
- **ResNet50**
- **ResNet101**
- **EfficientNetB0**
- **VGG16**

Only the final dense layer was trained. All other layers were frozen to act as a feature extractor. Regularization techniques such as L2 penalty, dropout (20%), and batch normalization were used.

---

## Key Features

- Image preprocessing: resizing and zero-padding
- Data augmentation: zoom, crop, rotate, contrast, flip
- Early stopping and validation monitoring
- Performance metrics: Accuracy, Precision, Recall, F1 Score, AUC
- Final model evaluation across test set

---

## Results

| Model         | Accuracy | Precision | Recall | F1 Score | AUC  |
|---------------|----------|-----------|--------|----------|------|
| ResNet50      | .84      | .85       | .84    | .84      | .985 |
| ResNet101     | .84      | .84       | .84    | .84      | .983 |
| EfficientNetB0| .83      | .84       | .83    | .83      | .981 |
| VGG16         | .77      | .77       | .77    | .77      | .972 |


---

## Conclusion

In this project, we evaluated the performance of four pre-trained deep learning models—**ResNet-50**, **ResNet-101**, **EfficientNetB0**, and **VGG16**—as frozen feature extractors for a multi-class waste classification task. Each model was paired with a compact fully connected head (**ReLU + Softmax**, **L2 regularization**, **Batch Normalization**, and **Dropout**) and trained using only the outputs of the frozen base.

**ResNet-50** outperformed the others, achieving the highest test **accuracy (0.84)**, **macro F1 score (0.85)**, and **macro AUC (0.985)**, while maintaining balanced performance across all nine waste classes. While **ResNet-101** and **EfficientNetB0** produced competitive results with **0.83 test accuracy**, they did not surpass ResNet-50. **VGG16** showed the weakest generalization on this task with a test accuracy of **0.77**.

## Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/efficient-transfer-learning.git
cd efficient-transfer-learning

# Set up virtual environment
python -m venv env
source env/bin/activate  # or env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Files in This Repo

| File                          | Description |
|-------------------------------|-------------|
| `waste_image_classification.ipynb` | Main implementation with training, validation, evaluation |
| `requirements.txt`           | Dependencies list |
| `.gitignore`                 | Git ignore rules |
| `README.md`                  | This file |

---
