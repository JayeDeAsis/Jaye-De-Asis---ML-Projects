import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import pickle
from PIL import Image
from img2vec_pytorch import Img2Vec
from io import BytesIO

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained image classification model
model_path = 'de-asis-model_dog_breeds.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize Img2Vec
img2vec = Img2Vec()

# Custom CSS for better UI
st.markdown("""
    <style>
    /* General styles */
    body {
        font-family: Arial, sans-serif;
        color: #333;
    }
    .stApp {
        background-color: #ffffff;
        padding: 2rem;
    }
    /* Sidebar styles */
    .css-1v0mbdj.eknhn3m10 {
        display: none;
    }
    .css-1d391kg {
        background-color: #f0f2f6;
        color: #333;
        border-right: 2px solid #ccc;
    }
    .css-1e5imcs, .css-h5rgaw {
        color: #007BFF !important;
    }
    .css-1e5imcs:hover, .css-h5rgaw:hover {
        background-color: #e9f5ff;
        color: #0056b3 !important;
    }
    /* Header styles */
    .css-h4g6ky, .css-1v3fvcr, h2 {
        color: #333;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    /* Button styles */
    .stButton>button {
        background-color: #007BFF;
        color: #fff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: #fff;
    }
    /* Input field styles */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border: 1px solid #007BFF;
        border-radius: 5px;
        padding: 10px;
    }
    /* Navigation list items */
    .sidebar-content .block-container li {
        font-size: 18px;
    }
    /* Sidebar header styles */
    .css-10trblm.e16nr0p33 {
        font-size: 1.5rem;
        font-weight: bold;
        color: #007BFF;
        padding: 0.5rem 1rem;
    }
    /* Section header styles */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📋 ITEQMT Machine Learning Application Portfolio")
section = st.sidebar.radio("Navigate", ["Home", "Prediction", "Sentiment Analysis", "Image Classification", "Source Codes"])

if section == "Home":
    st.title("ITEQMT Machine Learning Application Portfolio")
    st.write("## Multiple Maching Learning by Jaye")
    st.image("https://getwallpapers.com/wallpaper/full/b/7/0/765203-technology-background-images-1920x1080-hd.jpg", use_column_width=True)  # Replace with your image URL
    st.write("""
        Welcome to my Machine Learning Application Portfolio. Here, you will find various machine learning projects that I have worked on, along with the source code and descriptions.
    """)
    st.header("About Me")
    st.write("""
        I am a passionate machine learning enthusiast dedicated to developing innovative applications using cutting-edge technologies.
    """)
    st.header("Description of My Apps")
    st.write("""
        - **Prediction**: Make predictions based on input features using a pre-trained machine learning model.
        - **Sentiment Analysis**: Analyze the sentiment of a given text using a pre-trained sentiment analysis model.
        - **Image Classification**: Classify images into different categories using a pre-trained image classification model.
    """)
    st.header("What I Have Learned in Class")
    st.write("""
        In class, I have learned various machine learning techniques and tools, including data preprocessing, model training, evaluation, and deployment using Streamlit. I have also gained hands-on experience in developing end-to-end machine learning applications to solve real-world problems.
    """)

elif section == "Prediction":
    st.sidebar.subheader("Prediction")
    st.subheader("Prediction")
    st.write("This application allows you to make predictions based on input features using a pre-trained machine learning model.")

    @st.cache_data
    def load_data():
        datasetCSV = pd.read_csv("/content/drive/MyDrive/Significant Earthquake Dataset 1900-2023.csv")
        features = ['Mag', 'Depth', 'Latitude', 'Longitude']
        X = datasetCSV[features]
        y = (datasetCSV['Mag'] >= 6.0).astype(int)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, imputer

    @st.cache_resource
    def train_models(X_train, y_train):
        # Random Forest
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_classifier = RandomForestClassifier()
        grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=5)
        grid_search_rf.fit(X_train, y_train)

        # SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        svm = SVC(kernel='linear')
        svm.fit(X_train_scaled, y_train)

        return grid_search_rf, scaler, svm

    X_train, X_test, y_train, y_test, imputer = load_data()
    grid_search_rf, scaler, svm = train_models(X_train, y_train)

    # Input features
    feature1 = st.number_input("Magnitude", min_value=0.0, max_value=10.0)
    feature2 = st.number_input("Depth", min_value=0.0)
    feature3 = st.number_input("Latitude")
    feature4 = st.number_input("Longitude")

    # Define a new earthquake instance
    new_earthquake = pd.DataFrame({
        'Mag': [feature1],
        'Depth': [feature2],
        'Latitude': [feature3],
        'Longitude': [feature4]
    })

    # Impute and scale the new instance
    new_earthquake_imputed = imputer.transform(new_earthquake)
    new_earthquake_scaled = scaler.transform(new_earthquake_imputed)

    if st.button("Make Prediction"):
        # Make a prediction using the trained Random Forest model
        prediction_rf = grid_search_rf.best_estimator_.predict(new_earthquake_scaled)

        # Make a prediction using the trained SVM model
        prediction_svm = svm.predict(new_earthquake_scaled)

        # Print the predictions
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Prediction Results")
        if prediction_rf[0] == 1:
            st.success("Random Forest predicts the earthquake to be significant (magnitude >= 6.0).")
        else:
            st.warning("Random Forest predicts the earthquake to be not significant (magnitude < 6.0).")

        if prediction_svm[0] == 1:
            st.success("SVM predicts the earthquake to be significant (magnitude >= 6.0).")
        else:
            st.warning("SVM predicts the earthquake to be not significant (magnitude < 6.0).")

elif section == "Sentiment Analysis":
    st.title("Sentiment Analysis with Emojis")
    st.subheader("Analyze the sentiment of a given text using a pre-trained sentiment analysis model.")

    # Define word features function
    def word_features(words):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        return dict([(word, True) for word in words if word.lower() not in stop_words])

    # Define emotion word lists
    emotion_happy = ['happy', 'joyful', 'delighted', 'cheerful', 'content']
    emotion_excited = ['excited', 'thrilled', 'ecstatic', 'enthusiastic', 'energetic']
    emotion_sad = ['sad', 'unhappy', 'melancholy', 'depressed', 'tearful']
    emotion_nervous = ['nervous', 'anxious', 'worried', 'apprehensive', 'restless']
    emotion_scared = ['scared', 'fearful', 'terrified', 'panicked', 'trembling']

    # Prepare training data
    happy_features = [(word_features(word_tokenize(word)), 'Happy 😊') for word in emotion_happy]
    excited_features = [(word_features(word_tokenize(word)), 'Excited 😃') for word in emotion_excited]
    sad_features = [(word_features(word_tokenize(word)), 'Sad 😔') for word in emotion_sad]
    nervous_features = [(word_features(word_tokenize(word)), 'Nervous 😬') for word in emotion_nervous]
    scared_features = [(word_features(word_tokenize(word)), 'Scared 😱') for word in emotion_scared]

    train_set = happy_features + excited_features + sad_features + nervous_features + scared_features

    # Train Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(train_set)

    # Function to analyze sentiment
    def analyze_sentiment(sentence):
        tokens = word_tokenize(sentence)
        features = word_features(tokens)
        sentiment = classifier.classify(features)
        return sentiment

    # Main app logic
    sentence = st.text_input("Enter a sentence", "")
    if st.button("Analyze Sentiment"):
        if sentence.strip():
            sentiment = analyze_sentiment(sentence)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("Sentiment Analysis Result")
            st.success(f"Sentiment: {sentiment}")
        else:
            st.warning("No input sentence provided.")

elif section == "Image Classification":
    st.sidebar.subheader("Image Classification")
    st.subheader("Image Classification")
    st.write("This application allows you to classify images into different categories using a pre-trained image classification model.")

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

    # Define class names
    class_names = ["beagle", "bulldog", "dalmatian", "german-shepherd"]

    def classify_image(upload):
        image = Image.open(upload).convert('RGB')
        col1.write("### Image to be classified")
        col1.image(image)

        col2.write("### Classification Result")
        features = img2vec.get_vec(image)
        pred = model.predict([features])[0]
        prediction_probabilities = model.predict_proba([features])[0]

        # Display prediction probabilities with class names
        for class_name, prob in zip(class_names, prediction_probabilities):
            col2.write(f'{class_name.capitalize()}: {prob*100:.2f}%')

    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error("The uploaded file is too large. Please upload a file smaller than 5MB.")
        else:
            classify_image(my_upload)
    else:
        st.write("Please upload an image file.")

elif section == "Source Codes":
    st.title("Source Codes")
    source_codes = st.selectbox("Choose source code to view:", ["Prediction", "Sentiment Analysis", "Image Classification"])
    if source_codes == "Prediction":
        st.code("""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Read the dataset
datasetCSV = pd.read_csv("/content/Significant Earthquake Dataset 1900-2023.csv")

# Select relevant features
from google.colab import drive
drive.mount('/content/drive')
features = ['Mag', 'Depth', 'Latitude', 'Longitude']
X = datasetCSV[features]

# Define a label based on magnitude (e.g., magnitude >= 6.0 is a significant earthquake)
y = (datasetCSV['Mag'] >= 6.0).astype(int)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest classifier
rf_classifier = RandomForestClassifier()

# Initialize GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=5)

# Fit the Random Forest model
grid_search_rf.fit(X_train, y_train)

# Get the best parameters for Random Forest
best_params_rf = grid_search_rf.best_params_
print("Best Parameters for Random Forest:", best_params_rf)

# Evaluate the Random Forest model
accuracy_rf = grid_search_rf.score(X_test, y_test)
print("Accuracy for Random Forest:", accuracy_rf)

# Scale the features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# Evaluate the SVM model on the test set
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'Accuracy for SVM: {accuracy_svm}')

# Define a new earthquake instance
new_earthquake = pd.DataFrame({
    'Mag': [6.5],
    'Depth': [10.0],
    'Latitude': [35.7],
    'Longitude': [-118.2]
})

# Impute and scale the new instance
new_earthquake_imputed = imputer.transform(new_earthquake)
new_earthquake_scaled = scaler.transform(new_earthquake_imputed)

# Make a prediction using the trained Random Forest model
prediction_rf = grid_search_rf.best_estimator_.predict(new_earthquake_scaled)

# Make a prediction using the trained SVM model
prediction_svm = svm.predict(new_earthquake_scaled)

# Print the predictions
if prediction_rf[0] == 1:
    print("Random Forest predicts the earthquake to be significant (magnitude >= 6.0).")
else:
    print("Random Forest predicts the earthquake to be not significant (magnitude < 6.0).")

if prediction_svm[0] == 1:
    print("SVM predicts the earthquake to be significant (magnitude >= 6.0).")
else:
    print("SVM predicts the earthquake to be not significant (magnitude < 6.0).")
        """)
    elif source_codes == "Sentiment Analysis":
        st.code("""
!pip install streamlit
import streamlit as st
import nltk
from nltk.classify import NaiveBayesClassifier
nltk.download('punkt')

def word_features(words):
    return dict([(word, True) for word in words])

emotion_happy = ['happy', 'joyful', 'delighted', 'cheerful', 'content']
emotion_excited = ['excited', 'thrilled', 'ecstatic', 'enthusiastic', 'energetic']
emotion_sad = ['sad', 'unhappy', 'melancholy', 'depressed', 'tearful']
emotion_nervous = ['nervous', 'anxious', 'worried', 'apprehensive', 'restless']
emotion_scared = ['scared', 'fearful', 'terrified', 'panicked', 'trembling']

happy_features = [(word_features(word.split()), 'Happy 😊') for word in emotion_happy]
excited_features = [(word_features(word.split()), 'Excited 😃') for word in emotion_excited]
sad_features = [(word_features(word.split()), 'Sad 😔') for word in emotion_sad]
nervous_features = [(word_features(word.split()), 'Nervous 😬') for word in emotion_nervous]
scared_features = [(word_features(word.split()), 'Scared 😱') for word in emotion_scared]

train_set = happy_features + excited_features + sad_features + nervous_features + scared_features

classifier = NaiveBayesClassifier.train(train_set)

with open("DE ASIS_SentimentAnalyzer_Model_StreamlitApp.py", "w") as file:
    file.write
def analyze_sentiment(sentence):
    classifier = get_classifier()
    sentiment = classifier.classify(word_features(word_tokenize(sentence)))
    return sentiment

def main():
    st.title("Sentiment Analysis with Emojis")
    sentence = st.text_input("Enter a sentence", "")
    if st.button("Analyze Sentiment"):
        if sentence.strip():
            sentiment = analyze_sentiment(sentence)
            st.write("Sentiment:", sentiment)
        else:
            st.write("No input sentence provided.")

if __name__ == "__main__":
    main()
        """)
    elif source_codes == "Image Classification":
        st.code("""
!pip install img2vec_pytorch
!pip install torch torchvision
!pip install scikit-learn==1.4.2

from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import sklearn
print(sklearn.__version__)

# Define paths
data_dir = '/content/drive/MyDrive/dog-breeds/dog-breeds'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Initialize Img2Vec
img2vec = Img2Vec()

# Create train and val directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Function to check if a file is an image
def is_image_file(filename):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    return filename.lower().endswith(valid_extensions)

# Split the data into training and validation sets
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)
    if os.path.isdir(category_path) and category not in ['train', 'val']:
        images = [f for f in os.listdir(category_path) if is_image_file(f)]
        if len(images) == 0:
            print(f"Skipping empty directory: {category_path}")
            continue
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

        # Move training images
        train_category_dir = os.path.join(train_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        for img in train_images:
            shutil.move(os.path.join(category_path, img), os.path.join(train_category_dir, img))

        # Move validation images
        val_category_dir = os.path.join(val_dir, category)
        os.makedirs(val_category_dir, exist_ok=True)
        for img in val_images:
            shutil.move(os.path.join(category_path, img), os.path.join(val_category_dir, img))

# Verify directories
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError(f"Check the directory paths. Train: {train_dir}, Val: {val_dir}")

# Initialize Img2Vec
img2vec = Img2Vec()

# Prepare data
data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        category_path = os.path.join(dir_, category)
        if os.path.isdir(category_path):
            for img_path in os.listdir(category_path):
                img_path_ = os.path.join(category_path, img_path)
                if is_image_file(img_path_):
                    img = Image.open(img_path_).convert('RGB')
                    img_features = img2vec.get_vec(img)
                    features.append(img_features)
                    labels.append(category)
    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

# Plot sample images
class_names = sorted(os.listdir(train_dir))
nrows = len(class_names)
ncols = 10
plt.figure(figsize=(ncols*1.5, nrows*1.5))
for row in range(nrows):
    class_name = class_names[row]
    img_paths = [os.path.join(train_dir, class_name, filename)
        for filename in os.listdir(os.path.join(train_dir, class_name)) if is_image_file(filename)]
    for col in range(min(ncols, len(img_paths))):
        plt.subplot(nrows, ncols, row*ncols + col + 1)
        img = plt.imread(img_paths[col])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(class_name, fontsize=8)
plt.tight_layout()
plt.show()

# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly']
}

# Create a GridSearchCV object
model = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, scoring='accuracy')

# Fit the model
model.fit(data['training_data'], data['training_labels'])

# Print best parameters and score
print("Best Parameters:", model.best_params_)
print("Best Score:", model.best_score_)

# Save the model
model_path = '/content/drive/MyDrive/dog-breeds/de-asis-model_dog_breeds.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Load and test the model (Example)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()
class_labels = sorted(os.listdir(train_dir))

image_path = '/content/drive/MyDrive/dog-breeds/example/test_images/test.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Test image not found: {image_path}")

img = Image.open(image_path).convert('RGB')
features = img2vec.get_vec(img)
features_2d = features.reshape(1, -1)

# Get prediction probabilities
prediction_probabilities = model.predict_proba(features_2d)[0]
for ind, prob in enumerate(prediction_probabilities):
    print(f'Class {class_labels[ind]}: {prob*100:.2f}%')

# Get prediction
pred = model.predict(features_2d)
print(f'Predicted Class: {pred[0]}')

!pip install streamlit
!pip install img2vec_pytorch

#note we need to install a specific version to avoid having issues with scikit-learn library
!pip install scikit-learn==1.4.2


#Since using StreamLit library requires a python file (.py), this codes writes a python file in Google Colab
%%writefile app.py
import pickle
from PIL import Image
from io import BytesIO
from img2vec_pytorch import Img2Vec
import streamlit as st

#NOTE don't forget to upload the picke (model) file to your Google Colab First
#to run this code
#you can use any model that is capable of classifiying images that uses img2vec_pytorch
with open('de-asis-model_dog_breeds.pkl', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

## Streamlit Web App Interface
st.set_page_config(layout="wide", page_title="Image Classification for Dog Breeds")

st.write("## Image Classification Model in Python!")
st.write(
    ":grin: Upload an image of a dog and we'll try to classify its breed based on the trained features :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
@st.cache_data
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="jpg")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    image = Image.open(upload)
    col1.write("Image to be predicted :camera:")
    col1.image(image)

    col2.write("Category :wrench:")
    img = Image.open(my_upload)
    features = img2vec.get_vec(img)
    pred = model.predict([features])

    # print(pred)
    col2.header(pred)
    # st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    st.write("by Jaye De Asis...")
    # fix_image("./dog.jpg")

! wget -q -O - ipv4.icanhazip.com

! streamlit run app.py & npx localtunnel --port 8501
        """)
