# lung-cancer
This project uses a deep learning-based image classification model to detect and classify lung tissue images into three categories:  Lung Adenocarcinoma (lung_aca)  Lung Benign Tissue (lung_n)  Lung Squamous Cell Carcinoma (lung_scc)  The model is built on VGG16, a pre-trained CNN architecture fine-tuned on a custom lung cancer dataset.
⚙️ Tech Stack

Python, TensorFlow / Keras

NumPy, Matplotlib, OpenCV

Flask (for web deployment)

Google Colab (for training & testing)

🧠 Model Details

Base Model: VGG16 (pre-trained on ImageNet)

Image Size: 224 × 224

Training Accuracy: ~95%

Optimizer: Adam

Loss: Categorical Crossentropy

🚀 Features

✅ Classifies lung tissue into 3 distinct types
✅ High accuracy on test images
✅ Flask web app for real-time image upload and prediction
✅ Easily extendable to other histopathology datasets




