# 🌾 Crop Health Classifier with PyTorch & Streamlit

This project detects whether a crop image is **damaged** or **non-damaged** using a CNN model built in **PyTorch**. It also includes a simple **Streamlit** web app for real-time inference.

## 🚀 Features

- Custom-built SimpleCNN model
- Streamlit web interface for image upload
- Confusion matrix + TensorBoard logging
- Easy to retrain or test with your own images

## 🧠 Model

The model is trained on two classes:
- `damaged`
- `non_damaged`

🧩 Pretrained model is located at:  
`trained_models/best_cnn.pt`

## 📁 Project Structure

```
├── data/                       ← Training and test data (not uploaded)
│   └── crop/
│       ├── train/
│       └── test/
├── trained_models/
│   └── best_cnn.pt            ← Pretrained model
├── dataset.py
├── model.py
├── train_cnn.py
├── test.py
├── app.py                     ← Streamlit Web App
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/crop-health-classifier.git
cd crop-health-classifier
pip install -r requirements.txt
```

## 🖼️ Run Streamlit App

```bash
streamlit run app.py
```

## 🏋️‍♂️ Retrain the Model (Optional)

```bash
python train_cnn.py   --root ./data   --epochs 10   --batch_size 32   --images_size 112   --logging board   --trained_models trained_models
```

## 🧪 Test a Single Image

```bash
python test.py --images_path path_to_your_image.jpg
```

## 📥 Download Pretrained Model

If `trained_models/best_cnn.pt` is not present, download it here:

🔗 [Download from Google Drive](https://your_google_drive_link)

Put it into the `trained_models/` directory.

## 🛑 Disclaimer

- `data/` folder is excluded from this repository.
- Please use your own dataset with structure:

```bash
data/
└── crop/
    ├── train/
    │   ├── damaged/
    │   └── non_damaged/
    └── test/
        ├── damaged/
        └── non_damaged/
```

## 📜 License

MIT — free for personal & commercial use.

## 🙌 Author

Developed by [Pea](https://github.com/PeaIsPea)
