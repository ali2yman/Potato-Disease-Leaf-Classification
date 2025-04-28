# Potato-Disease-Leaf-Classification
This project focuses on detecting potato leaf diseases (Early Blight, Late Blight) and distinguishing them from Healthy leaves using Deep Learning and Transfer Learning techniques.  We use the Potato Leaf Disease (PLD) dataset, apply proper preprocessing, and a powerful model based on EfficientNetB0 architecture.

.

## 🔥 Project Workflow

1. **Data Preprocessing**
   - Resize all images to **224x224**.
   - Apply **image scaling** (normalization between 0-1).
   - Handle **class imbalance** using class weights or augmentation.
   - Perform **data augmentation** during training only.

2. **Data Augmentation Techniques**
   - Random rotation
   - Zoom
   - Horizontal flip
   - Brightness adjustments

3. **Model Building**
   - **Backbone**: `EfficientNetB0` pretrained on ImageNet.
   - **Custom head**: 
     - Global Average Pooling
     - Batch Normalization
     - Dense(512) + Dropout(0.5)
     - Dense(128) + Dropout(0.3)
     - Final Dense(3) with Softmax for classification

4. **Training Strategy**
   - First, train only the new layers (with the base frozen).
   - Then, unfreeze selected layers for **fine-tuning**.
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy
   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

5. **Evaluation**
   - Generate classification reports and confusion matrices.
   - Visualize predictions with Matplotlib.
   - Test model with individual images.

---

## 🛠️ Key Technologies

- Python
- TensorFlow / Keras
- scikit-learn
- NumPy
- Matplotlib

---

## 📊 Results

After training, the model achieves strong performance on the validation and test sets, effectively distinguishing between Early Blight, Late Blight, and Healthy leaves.


## 🚀 How to Run

1. Clone this repository.
2. Install requirements:
   ```bash
   pip install tensorflow scikit-learn matplotlib numpy