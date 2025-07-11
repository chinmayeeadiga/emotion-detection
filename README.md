# Emotion Detection from Text 

This project uses a CNN model to detect emotions from text inputs. It classifies input sentences into one of six emotions: `sadness`, `joy`, `love`, `angry`, `fear`, or `surprise`.

##  Features
- Preprocessing includes lowercasing, punctuation removal, tokenization, stopword removal, emoji handling, lemmatization
- Trained CNN model using Keras
- Simple and beautiful UI built using Gradio
- Can run locally in one click

##  Emotion Mapping
_____________________
| Index | Emotion   |
|-------|-----------|
| 0     | sadness   |
| 1     | joy       |
| 2     | love      |
| 3     | angry     |
| 4     | fear      |
| 5     | surprise  |
---------------------
##  How to Run

```bash
# Clone the repo
git clone https://github.com/chinmayeeadiga/emotion-detection.git
cd emotion-detection

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
