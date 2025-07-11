import gradio as gr
import numpy as np
import re
import pickle
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load model, tokenizer, label encoder
model = load_model("model/model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
max_len = 35

def preprocess_text(text):
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()  # simpler alternative to nltk.word_tokenize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_emotion(text):
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post")
    
    prediction = model.predict(padded)[0]  # model output shape: (6,)
    predicted_index = np.argmax(prediction)
    confidence = round(prediction[predicted_index] * 100, 2)
    
    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "angry",
        4: "fear",
        5: "surprise"
    }

    emotion = emotion_map.get(predicted_index, "Unknown")
    
    return f"{emotion} ({confidence}%)", [[emotion, confidence]]


# Build UI with Blocks
with gr.Blocks(theme=gr.themes.Base(), title="Emotion Detection") as demo:
    gr.Markdown("## 💬 Emotion Detection from Text")
    gr.Markdown("Enter a sentence, and the model will predict associated emotions.")

    with gr.Row():
        input_box = gr.Textbox(label="Your Sentence", lines=3, placeholder="e.g., I am really happy and excited today!")

    with gr.Row():
        predict_btn = gr.Button("🔍 Predict")
        clear_btn = gr.Button("❌ Clear")

    with gr.Row():
        output_text = gr.Textbox(label="Predicted Emotions", interactive=False)
        label_list = gr.Dataframe(headers=["Emotion", "Confidence %"], interactive=False)

    # Logic connections
    predict_btn.click(predict_emotion, inputs=input_box, outputs=[output_text, label_list])
    clear_btn.click(fn=lambda: ("", []), outputs=[output_text, label_list])

demo.launch()
