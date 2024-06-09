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
    /* Sidebar header */
    .css-1v0mbdj.eknhn3m10 {
        display: none;
    }
    /* Sidebar content */
    .css-1d391kg {
        background-color: #f0f2f6;
        color: #333;
        border-right: 2px solid #ccc;
    }
    /* Sidebar items */
    .css-1e5imcs, .css-h5rgaw {
        color: #007BFF !important;
    }
    .css-1e5imcs:hover, .css-h5rgaw:hover {
        background-color: #e9f5ff;
        color: #0056b3 !important;
    }
    /* Main app background */
    .stApp {
        background-color: #ffffff;
        padding: 2rem;
    }
    /* Section headers */
    .css-h4g6ky {
        color: #333;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    /* Buttons */
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
    /* Text input fields */
    .stTextInput>div>div>input {
        border: 1px solid #007BFF;
        border-radius: 5px;
        padding: 10px;
    }
    .stNumberInput>div>div>input {
        border: 1px solid #007BFF;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📋 ITEQMT Machine Learning Application Portfolio")
section = st.sidebar.radio("Navigate", ["Home", "Prediction", "Sentiment Analysis", "Image Classification"])

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

    # Load the dataset
    datasetCSV = pd.read_csv("/content/drive/MyDrive/Significant Earthquake Dataset 1900-2023.csv")

    # Select relevant features
    features = ['Mag', 'Depth', 'Latitude', 'Longitude']
    X = datasetCSV[features]
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

    # Scale the features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the SVM model
    svm = SVC(kernel='linear')
    svm.fit(X_train_scaled, y_train)

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

