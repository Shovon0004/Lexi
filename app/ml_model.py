import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_user_data():
    data = pd.read_csv('user_feedback.csv')
    return data

def train_model():
    data = load_user_data()
   
    # Separate features and labels
    X = data[['font_name', 'font_size', 'line_spacing', 'letter_spacing', 'text_color']]
    y = data['feedback']
   
    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['font_size', 'line_spacing', 'letter_spacing']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['font_name', 'text_color'])
        ])
   
    # Create a pipeline with preprocessing and RandomForestClassifier
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
   
    model.fit(X, y)
    return model

def predict_formatting(user_preferences, model):
    prediction = model.predict(user_preferences)
    # If the model predicts 0 (needs improvement), suggest optimal dyslexia-friendly settings.
    # We omit 'text_color' from the suggestions to preserve the user's manual color choices.
    if prediction[0] == 0:
        return {
            'font_name': 'OpenDyslexic',
            'font_size': 14,
            'line_spacing': 16,
            'letter_spacing': 0.2
        }
    else:
        return {}  # Return empty dict to use user's original preferences
