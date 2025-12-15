import numpy as np
import pandas as pd
import os
import sys
from src.logger import logging  # Ensure your logger is properly set up
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded successfully from %s', file_path)
        return df
    except Exception as e:
        logging.error('Error loading data from %s: %s', file_path, e)
        raise


class PodcastPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.genre_mapping = {
            'Music': 0, 'True Crime': 1, 'Health': 2, 'Education': 3,
            'Technology': 4, 'Business': 5, 'Lifestyle': 6,
            'Sports': 7, 'Comedy': 8, 'News': 9
        }
        self.day_mapping = {
            'Tuesday': 0, 'Monday': 1, 'Wednesday': 2,
            'Saturday': 3, 'Friday': 4, 'Thursday': 5, 'Sunday': 6
        }
        self.time_mapping = {
            'Night': 0, 'Afternoon': 1, 'Morning': 2, 'Evening': 3
        }
        self.label_encoders = {}
        self.num_medians = {}
        self.feature_names_ = None

    def _data_process(self, df):
        df = df.copy()

        # Your feature engineering
        df['Episode_Title_num'] = (
            df['Episode_Title'].astype(str).str.replace('Episode ', '', regex=False).astype(int)
        )
        # numeric medians applied later
        df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(
            {'Neutral': 0, 'Positive': 1, 'Negative': -1}
        )

        df['Ad_Density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-3)
        df['Popularity_Diff'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
        df['Popularity_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
        df['Host_Popularity_squared'] = df['Host_Popularity_percentage'] ** 2
        df['Popularity_Average'] = (
            df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
        ) / 2
        
        df['Genre_Num'] = df['Genre'].map(self.genre_mapping)
        df['Publication_Day_Num'] = df['Publication_Day'].map(self.day_mapping)
        df['Publication_Time_Num'] = df['Publication_Time'].map(self.time_mapping)

        return df

    def fit(self, X, y=None):
        X = self._data_process(X)

        # 1) Fit TF-IDF on Podcast_Name
        tfidf_train = self.vectorizer.fit_transform(X['Podcast_Name'])
        tfidf_df = pd.DataFrame(
            tfidf_train.toarray(),
            columns=self.vectorizer.get_feature_names_out(),
            index=X.index
        )

        # 2) Fill numeric medians and store them
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        for col in num_cols:
            median_val = X[col].median()
            self.num_medians[col] = median_val
            X[col] = X[col].fillna(median_val)

        # 3) Label encode categorical columns
        cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        X[cat_cols] = X[cat_cols].fillna("Missing")
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # 4) Combine tabular + TF-IDF and remember feature order
        X_full = pd.concat([X, tfidf_df], axis=1)
        self.feature_names_ = X_full.columns.tolist()


        return self

    def transform(self, X):
        X = self._data_process(X)

        # 1) TF-IDF using existing vocab
        tfidf_test = self.vectorizer.transform(X['Podcast_Name'])
        tfidf_df = pd.DataFrame(
            tfidf_test.toarray(),
            columns=self.vectorizer.get_feature_names_out(),
            index=X.index
        )

        # 2) Fill numeric using training medians
        for col, median_val in self.num_medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)

        # 3) Apply label encoders (handle unknowns as "Missing" if needed)
        cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
        X[cat_cols] = X[cat_cols].fillna("Missing")
        for col, le in self.label_encoders.items():
            # Map unknown labels to a fallback if required
            X[col] = X[col].map(lambda v: v if v in le.classes_ else "Missing")
            # Ensure encoder knows "Missing"
            if "Missing" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Missing")
            X[col] = le.transform(X[col])

        X_full = pd.concat([X, tfidf_df], axis=1)

        # Reindex to match training feature order
        X_full = X_full.reindex(columns=self.feature_names_, fill_value=0)

        return X_full.values


def apply_feature_transformer(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply Feature Transformer to vectorize both text categorical data separately for X_train and X_test."""
    try:
        logging.info("Applying Feature Transformer to numerical features...")

        # Separate features (X) and target (y)
        X_train = train_data.drop(columns=['Listening_Time_minutes'])
        y_train = train_data['Listening_Time_minutes']
        X_test = test_data.drop(columns=['Listening_Time_minutes'])
        y_test = test_data['Listening_Time_minutes']

        preprocessor = PodcastPreprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_train = pd.DataFrame(X_train,columns=preprocessor.feature_names_)


        X_test = preprocessor.transform(X_test)
        X_test = pd.DataFrame(X_test,columns=preprocessor.feature_names_)
        # joblib.dump(preprocessor, "podcast_transform.pkl")

        # Reconstruct DataFrames
        train_transformed = pd.concat([X_train, y_train], axis=1)
        test_transformed = pd.concat([X_test, y_test], axis=1)

        logging.info("Feature Transformation applied successfully.")

        return train_transformed, test_transformed, preprocessor
    except Exception as e:
        logging.error("Error during Feature Transformation: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error("Error saving data to %s: %s", file_path, e)
        raise

def save_feature_transformer(transformer, file_path: str) -> None:
    """Save the trained FeatureTransformer to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(transformer, file)
        logging.info('FeatureTransformer saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the FeatureTransformer: %s', e)
        raise

def main():
    try:
        # Load processed train & test data from interim
        train_data = load_data('./data/interim/train_preprocessed.csv')
        test_data = load_data('./data/interim/test_preprocessed.csv')

        # Apply feature transformer
        X_train_processed, X_test_processed, preprocessor = apply_feature_transformer(train_data, test_data)

        # Save the transformed data
        save_data(X_train_processed, './data/processed/train_final.csv')
        save_data(X_test_processed, './data/processed/test_final.csv')

        # Save the preprocessor
        save_feature_transformer(preprocessor, './models/feature_transformer.pkl')
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error("Feature engineering failed: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()