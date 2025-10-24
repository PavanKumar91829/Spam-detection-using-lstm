# Spam Detection using LSTM (RNN)

## Overview  
This project implements a **Spam Detection System** using a **Recurrent Neural Network (RNN)** with **Long Short-Term Memory (LSTM)** layers.  
It classifies text messages as **Spam** or **Ham (Not Spam)** based on their content.  
The model is trained on labeled text data and deployed using **Streamlit** for an interactive web interface.

---
## 📂 Project Structure
```
├── LSTM_Text_Clf.ipynb # Model training & evaluation notebook
├── Dataset/
│ └── spam.csv # Dataset used for training
├── Streamlit_App/
│ ├── app.py # Streamlit interface for predictions
│ ├── requirements.txt # Dependencies for Streamlit app
│ ├── spam_classifier.pth # Trained LSTM model (PyTorch)
│ └── vocab.pkl # Saved tokenizer/vocabulary
└── README.md # Project documentation file

```
## ✨ Features
- Text preprocessing: tokenization, padding, and vocabulary creation  
- LSTM-based binary text classification model  
- Training and evaluation with accuracy and loss visualization  
- Streamlit web app for real-time spam detection  
- Model persistence with `.pth` and `vocab.pkl`

---

## 🛠 Technologies Used
- **Python 3.8+**
- **PyTorch**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Matplotlib / Seaborn**
- **Streamlit** for deployment

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/PavanKumar91829/Spam-detection-using-lstm.git
cd Spam-detection-using-lstm
```

### 2️⃣ Install dependencies  
You can use the requirements file included in the Streamlit app:
```bash
pip install -r Streamlit_App/requirements.txt
```

### 3️⃣ Run Jupyter Notebook (for training)
```bash
jupyter notebook LSTM_Text_Clf.ipynb
```

### 4️⃣ Run the Streamlit App
```bash
cd Streamlit_App
streamlit run app.py
```

---

## 🚀 Usage
- The notebook handles data loading, preprocessing, training, and evaluation.  
- The Streamlit app allows you to input text messages and see predictions in real-time:
  - **Spam** → flagged as unwanted or malicious  
  - **Ham** → legitimate/non-spam messages

---

## 📊 Model Performance
- Achieved **high accuracy (~97%)** on validation data  
- Well-balanced precision and recall scores for both spam and ham classes  
- Visualized training and validation loss curves in notebook

---

## 🔮 Future Improvements
- Explore using **Attention mechanisms**, **GRU**, or **Transformer-based models** to achieve better results
- Integrate **pre-trained embeddings** (GloVe, FastText)
- Deploy as a **REST API** using Flask or FastAPI
- Enhance dataset diversity for multilingual spam detection

---

## 🧾 License
This project is licensed under the **MIT License** — feel free to use and modify for educational purposes.

---

## 👨‍💻 Author
**Pavan Kumar A**  
🔗 GitHub: [PavanKumar91829](https://github.com/PavanKumar91829)  

---

> ⭐ If you find this project useful, please consider giving it a star on GitHub!
