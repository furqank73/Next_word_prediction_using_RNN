# Next Word Prediction using RNN

This project demonstrates a **Next Word Prediction Model** built using a **Simple Recurrent Neural Network (RNN)**.  
The model learns sequential patterns from text and predicts the most likely next word based on a given input sentence.  

---

## 📌 Features
- Text preprocessing (tokenization & padding)
- RNN model built with **TensorFlow/Keras**
- Train/test split with evaluation
- Generate next word predictions
- Interactive **Streamlit App** for real-time predictions

---

## 🛠️ Tech Stack
- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas  
- Scikit-learn  
- Streamlit  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/furqank73/next_word_prediction_using_RNN.git
cd next_word_prediction_using_RNN
````

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Training Script

```bash
python train_model.py
```

### 5. Launch Streamlit App

```bash
streamlit run app.py
```

---

## 🎯 Usage

* Enter the beginning of a sentence.
* The model will predict the most likely **next word(s)**.
* Example:

  * Input: `"Machine learning"`
  * Output: `"Machine learning algorithms"`

---

## 📊 Project Structure

```
📂 next_word_prediction_using_RNN
│── app.py                # Streamlit app for predictions
│── train_model.py        # Script to train the RNN model
│── model.h5              # Saved trained model (after training)
│── tokenizer.pkl         # Saved tokenizer
│── requirements.txt      # Project dependencies
│── README.md             # Project documentation
```

---

## 📌 Example (Streamlit App)

```
📝 Next Word Prediction App  
This app uses a SimpleRNN model to complete your sentence automatically.  

Enter the beginning of your sentence:  
👉 "Deep learning"  

✅ Completed Sentence:  
"Deep learning models are powerful"
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.

---

## 👨‍💻 Author

**M Furqan Khan**

* 🌐 [GitHub](https://github.com/furqank73)
* 🔗 [LinkedIn](https://www.linkedin.com/in/furqan-khan-256798268/)
* 📊 [Kaggle](https://www.kaggle.com/fkgaming)

```
