"""
Issue classification for supply chain anomalies.

This module classifies detected anomalies into specific issue types.
"""

import pandas as pd
import numpy as np
import logging
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Try to import SMOTE with error handling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    # Define a dummy SMOTE class for graceful degradation
    class SMOTE:
        def __init__(self, *args, **kwargs):
            pass
        
        def fit_resample(self, X, y):
            print("SMOTE not available. Using original data.")
            return X, y

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IssueClassifier:
    """
    Classifies supply chain anomalies into specific issue types.
    
    This class provides both rule-based and machine learning approaches to categorize
    detected anomalies into specific issue types, such as inventory imbalances, 
    sales performance gaps, pricing issues, etc.
    
    Attributes:
        classifier (sklearn.pipeline.Pipeline): ML classifier pipeline
        issue_types (list): Available issue type categories
        numerical_preprocessor (sklearn.preprocessing.StandardScaler): Scaler for numerical features
        categorical_preprocessor (sklearn.preprocessing.OneHotEncoder): Encoder for categorical features
    """
    
    def __init__(self):
        """Initialize the issue classifier."""
        self.classifier = None
        self.preprocessor = None
        self.issue_types = [
            'Inventory_Imbalance',
            'Sales_Performance_Gap',
            'Pricing_Issue',
            'Supply_Chain_Disruption',
            'Sell_Through_Bottleneck',
            'Aged Inventory'
            'Unknown'
        ]
        logger.info("Issue classifier initialized")

    def classify_with_rules(self, data):
        """
        Apply rule-based classification to anomalies.
        
        This method uses predefined business rules to classify anomalies without
        requiring ML model training, making it suitable for all dataset sizes.
        
        Args:
            data (pd.DataFrame): DataFrame containing data with anomaly flags
            
        Returns:
            pd.DataFrame: Subset of data containing only anomalies with issue types
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Anomaly detection must be performed first. 'is_anomaly' column not found.")
        
        # Make a copy to avoid modifying the original
        result_df = data.copy()
        
        # Define the conditions for each issue type
        conditions = [
            (result_df['WeeksOfStockT1'] > 8) | (result_df['WeeksOfStockT2'] < 2),
            (result_df['TargetQty'] < 0.7),
            (result_df['PricePositioning'] > 110),
            (result_df['Backlog'] < result_df['Backlog'].quantile(0.25)),
            (result_df['SellThruToRatio'] < 0.7),
            (result_df['AgedInventoryPct'] > 20)
        ]

        # Create an issue type column with default value
        result_df['issue_type'] = 'Unknown'
        
        # Apply rule-based classification only for anomalies
        for condition, issue in zip(conditions, self.issue_types[:-1]):  # Exclude 'No_Issue'
            mask = condition & (result_df['is_anomaly'] == 1)
            result_df.loc[mask, 'issue_type'] = issue
        
        # Count the issues
        issue_counts = result_df[result_df['is_anomaly'] == 1]['issue_type'].value_counts()
        logger.info("Rule-based issue classification completed")
        logger.info(f"Issue Type Distribution:\n{issue_counts}")
        
        # Create a subset of anomalies for return
        anomalies = result_df[result_df['is_anomaly'] == 1].copy()
        
        return anomalies

    def train_classifier(self, data, numerical_features, categorical_features):
        """
        Train a machine learning classifier for issue type classification.
        
        This method builds and trains a RandomForest classifier to categorize anomalies,
        handling class imbalance and small dataset challenges.
        
        Args:
            data (pd.DataFrame): DataFrame containing data with anomaly flags and rule-based issue types
            numerical_features (list): List of numerical feature column names
            categorical_features (list): List of categorical feature column names
            
        Returns:
            tuple: (trained classifier, preprocessor)
        """
        if 'is_anomaly' not in data.columns or 'issue_type' not in data.columns:
            raise ValueError("Data must contain 'is_anomaly' and 'issue_type' columns")
        
        # Apply rule-based classification first to get initial labels
        data_with_rules = self.classify_with_rules(data)
        
        # Extract features and target
        anomalies = data[data['is_anomaly'] == 1].copy()
        
        # Transfer rule-based classifications to the anomalies dataframe
        anomalies['issue_type'] = data_with_rules['issue_type']
        
        X = anomalies[numerical_features + categorical_features]
        y = anomalies['issue_type']
        
        # Check if we have multiple classes and enough samples
        if len(np.unique(y)) <= 1:
            logger.warning("Not enough distinct issue types for classification. Using rule-based only.")
            return None, None
        
        # Get sample counts for each class
        class_counts = dict(Counter(y))
        logger.info(f"Class distribution before balancing: {class_counts}")
        
        # Feature preprocessing setup
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'  # Include any columns not specified
        )
        
        # Split data for training and testing
        try:
            # Try to use stratified split if possible
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, 
                stratify=y if len(y) > len(np.unique(y))*3 else None
            )
        except ValueError:
            # Fall back to non-stratified split if stratified fails
            logger.warning("Stratified split failed. Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Preprocess the features
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        
        # Handle class imbalance based on dataset size
        n_samples = len(X_train)
        min_samples_needed = 6  # For SMOTE with default k_neighbors=5
        
        # Check if SMOTE can be applied
        sampling_successful = False
        all_classes_have_enough = all(count >= min_samples_needed for count in class_counts.values())
        
        if SMOTE_AVAILABLE and all_classes_have_enough and n_samples >= 20:
            logger.info("Using SMOTE for class balancing")
            try:
                # Try with standard SMOTE
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)
                sampling_successful = True
            except ValueError as e:
                logger.warning(f"SMOTE failed: {str(e)}. Using original imbalanced data.")
                X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
        else:
            logger.info("Dataset too small for SMOTE or SMOTE not available. Using class weights instead.")
            X_train_balanced, y_train_balanced = X_train_preprocessed, y_train
        
        # Create a classifier with appropriate settings for dataset size
        if n_samples < 20:
            # For very small datasets, simpler model with strong regularization
            logger.info("Using simpler model for small dataset")
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced' if not sampling_successful else None,
                random_state=42
            )
        else:
            # For larger datasets, more complex model
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced' if not sampling_successful else None,
                random_state=42
            )
        
        # Train the classifier
        clf.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate if possible
        if len(X_test) > 0:
            try:
                X_test_preprocessed = preprocessor.transform(X_test)
                y_pred = clf.predict(X_test_preprocessed)
                logger.info("\nIssue Classification Report:")
                logger.info(classification_report(y_test, y_pred))
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
        
        # Create a pipeline with the preprocessor
        self.classifier = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        return self.classifier, preprocessor

    def classify_issues(self, data, numerical_features, categorical_features):
        """
        Classify anomalies into issue types using both rules and ML.
        
        This method combines rule-based and ML-based classification approaches,
        preferring ML predictions but falling back to rules when needed.
        
        Args:
            data (pd.DataFrame): DataFrame containing data with anomaly flags
            numerical_features (list): List of numerical feature column names
            categorical_features (list): List of categorical feature column names
            
        Returns:
            pd.DataFrame: Data with added issue_type classification
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Anomaly detection must be performed first. 'is_anomaly' column not found.")
        
        # Make a copy to avoid modifying the original
        result_df = data.copy()
        
        # First apply rule-based classification
        rule_based = self.classify_with_rules(result_df)
        
        # Transfer issue types to main dataframe
        result_df.loc[result_df['is_anomaly'] == 1, 'issue_type'] = rule_based['issue_type']
        
        # If we have enough data, try ML classification
        anomalies = result_df[result_df['is_anomaly'] == 1]
        if len(anomalies) >= 10 and len(np.unique(anomalies['issue_type'])) > 1:
            try:
                # Train the classifier if not already trained
                if self.classifier is None:
                    self.train_classifier(result_df, numerical_features, categorical_features)
                
                if self.classifier is not None:
                    # Apply ML classification
                    X = anomalies[numerical_features + categorical_features]
                    ml_predictions = self.classifier.predict(X)
                    
                    # Update issue types with ML predictions
                    anomalies_index = anomalies.index
                    for i, idx in enumerate(anomalies_index):
                        result_df.loc[idx, 'predicted_issue'] = ml_predictions[i]
                        
                    # For non-"No_Issue" predictions, use the ML prediction
                    no_issue_mask = (result_df['is_anomaly'] == 1) & (result_df['predicted_issue'] != 'No_Issue')
                    result_df.loc[no_issue_mask, 'issue_type'] = result_df.loc[no_issue_mask, 'predicted_issue']
                    
                    logger.info("Machine learning issue classification applied successfully")
            except Exception as e:
                logger.error(f"Error during ML classification: {str(e)}")
                logger.info("Falling back to rule-based classification only")
        else:
            logger.info("Not enough data for ML classification. Using rule-based only.")
        
        # Return only the anomalies
        return result_df[result_df['is_anomaly'] == 1].copy()

    def save_model(self, filepath):
        """
        Save the trained issue classifier to disk.
        
        Args:
            filepath (str): Path prefix for saving model files
        """
        if self.classifier is not None:
            joblib.dump(self.classifier, f"{filepath}_issue_classifier.pkl")
            logger.info(f"Issue classifier saved to {filepath}_issue_classifier.pkl")
        else:
            logger.warning("No trained classifier to save")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained issue classifier from disk.
        
        Args:
            filepath (str): Path prefix where model files are stored
            
        Returns:
            IssueClassifier: Loaded model instance
        """
        # Create a new instance
        classifier = cls()
        
        try:
            # Load the saved classifier
            classifier.classifier = joblib.load(f"{filepath}_issue_classifier.pkl")
            logger.info(f"Issue classifier loaded from {filepath}_issue_classifier.pkl")
        except (FileNotFoundError, IOError) as e:
            logger.error(f"Error loading issue classifier: {str(e)}")
            logger.info("Will use rule-based classification only")
        
        return classifier