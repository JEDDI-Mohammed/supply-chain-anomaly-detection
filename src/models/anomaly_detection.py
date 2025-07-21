import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Anomaly detection for supply chain data using ensemble methods.
    
    This class implements an ensemble approach that combines multiple anomaly detection
    algorithms (Isolation Forest, Local Outlier Factor, and One-Class SVM) for more
    robust anomaly detection in supply chain data.
    
    Attributes:
        scaler (StandardScaler): Scaler for normalizing input features
        contamination (float): Expected proportion of anomalies in the dataset
        isolation_forest (IsolationForest): Isolation Forest model instance
        lof (LocalOutlierFactor): Local Outlier Factor model instance
        ocsvm (OneClassSVM): One-Class SVM model instance
    """
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the anomaly detector with configurable parameters.
        
        Args:
            contamination (float): Expected proportion of anomalies (default: 0.05)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.scaler = StandardScaler()
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=self.contamination,
            random_state=self.random_state
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=False
        )
        
        self.ocsvm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=self.contamination
        )
        
        logger.info(f"Anomaly detector initialized with contamination {contamination}")

    def fit_transform(self, data, numerical_features):
        """
        Fit the anomaly detection models to the data and identify anomalies.
        
        This method trains the ensemble of anomaly detection models on the provided data
        and adds anomaly flags and scores to the input DataFrame.
        
        Args:
            data (pd.DataFrame): Input data containing supply chain metrics
            numerical_features (list): List of numerical feature column names to use
            
        Returns:
            pd.DataFrame: Original data with added anomaly columns
        """
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be None or empty")
            
        # Make a copy to avoid modifying the original
        result_df = data.copy()
        
        # Extract numerical features
        X = result_df[numerical_features]
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Fitting anomaly detection models on {len(X)} samples with {len(numerical_features)} features")
        
        # Train models
        self.isolation_forest.fit(X_scaled)
        # LOF fit_predict directly
        lof_pred = self.lof.fit_predict(X_scaled)
        self.ocsvm.fit(X_scaled)
        
        # Get predictions from each model (-1 for anomalies, 1 for normal)
        if_pred = self.isolation_forest.predict(X_scaled)
        ocsvm_pred = self.ocsvm.predict(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        if_anomalies = [1 if x == -1 else 0 for x in if_pred]
        lof_anomalies = [1 if x == -1 else 0 for x in lof_pred]
        ocsvm_anomalies = [1 if x == -1 else 0 for x in ocsvm_pred]
        
        # Create a vote count for each observation
        votes = np.array([if_anomalies, lof_anomalies, ocsvm_anomalies]).sum(axis=0)
        
        # Mark as anomaly if at least 2 models agree (majority voting)
        result_df['is_anomaly'] = [1 if v >= 2 else 0 for v in votes]
        
        # Compute ensemble anomaly score (average of normalized scores)
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        lof_scores = -self.lof.negative_outlier_factor_
        ocsvm_scores = -self.ocsvm.decision_function(X_scaled)
        
        # Normalize scores to [0,1] range for fair comparison
        if_scores_norm = self._normalize_scores(if_scores)
        lof_scores_norm = self._normalize_scores(lof_scores)
        ocsvm_scores_norm = self._normalize_scores(ocsvm_scores)
        
        # Average the normalized scores
        result_df['anomaly_score'] = np.mean([if_scores_norm, lof_scores_norm, ocsvm_scores_norm], axis=0)
        
        # Store individual model results for analysis
        result_df['if_anomaly'] = if_anomalies
        result_df['lof_anomaly'] = lof_anomalies
        result_df['ocsvm_anomaly'] = ocsvm_anomalies
        
        anomaly_count = result_df['is_anomaly'].sum()
        logger.info(f"Ensemble anomaly detection completed. Found {anomaly_count} anomalies out of {len(result_df)} records.")
        
        return result_df
    
    def predict(self, data, numerical_features):
        """
        Predict anomalies on new data using pre-trained models.
        
        Args:
            data (pd.DataFrame): New data to evaluate
            numerical_features (list): List of numerical feature column names to use
            
        Returns:
            pd.DataFrame: Input data with added anomaly columns
        """
        if not hasattr(self.isolation_forest, 'offset_') or not hasattr(self.ocsvm, 'offset_'):
            raise ValueError("Models have not been trained. Call fit_transform first.")
            
        # Make a copy to avoid modifying the original
        result_df = data.copy()
        
        # Extract numerical features
        X = result_df[numerical_features]
        
        # Scale the data using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model (-1 for anomalies, 1 for normal)
        if_pred = self.isolation_forest.predict(X_scaled)
        ocsvm_pred = self.ocsvm.predict(X_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        if_anomalies = [1 if x == -1 else 0 for x in if_pred]
        ocsvm_anomalies = [1 if x == -1 else 0 for x in ocsvm_pred]
        
        # For LOF, we need to use predict as it's been trained in novelty mode
        # For LOF in fit_predict mode, this would need to be refactored
        
        # Create a vote count (without LOF)
        votes = np.array([if_anomalies, ocsvm_anomalies]).sum(axis=0)
        
        # Mark as anomaly if all models agree (requires unanimity without LOF)
        result_df['is_anomaly'] = [1 if v >= 2 else 0 for v in votes]
        
        # Compute ensemble anomaly score (average of normalized scores)
        if_scores = -self.isolation_forest.decision_function(X_scaled)
        ocsvm_scores = -self.ocsvm.decision_function(X_scaled)
        
        # Normalize scores to [0,1] range for fair comparison
        if_scores_norm = self._normalize_scores(if_scores)
        ocsvm_scores_norm = self._normalize_scores(ocsvm_scores)
        
        # Average the normalized scores (without LOF)
        result_df['anomaly_score'] = np.mean([if_scores_norm, ocsvm_scores_norm], axis=0)
        
        # Store individual model results for analysis
        result_df['if_anomaly'] = if_anomalies
        result_df['ocsvm_anomaly'] = ocsvm_anomalies
        
        anomaly_count = result_df['is_anomaly'].sum()
        logger.info(f"Ensemble anomaly detection on new data completed. Found {anomaly_count} anomalies out of {len(result_df)} records.")
        
        return result_df

    def _normalize_scores(self, scores):
        """
        Normalize anomaly scores to [0,1] range.
        
        Args:
            scores (np.array): Raw anomaly scores
            
        Returns:
            list: Normalized scores in [0,1] range
        """
        min_val, max_val = min(scores), max(scores)
        if max_val > min_val:
            return [(s - min_val) / (max_val - min_val) for s in scores]
        else:
            return [0.5 for _ in scores]  # Default if all scores are equal

    def save_model(self, filepath):
        """
        Save the trained anomaly detection models to disk.
        
        Args:
            filepath (str): Path prefix for saving model files
        """
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'ocsvm': self.ocsvm,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        joblib.dump(model_data, f"{filepath}_anomaly_model.pkl")
        logger.info(f"Anomaly detection models saved to {filepath}_anomaly_model.pkl")

    @classmethod
    def load_model(cls, filepath):
        """
        Load trained anomaly detection models from disk.
        
        Args:
            filepath (str): Path prefix where model files are stored
            
        Returns:
            AnomalyDetector: Loaded model instance
        """
        model_data = joblib.load(f"{filepath}_anomaly_model.pkl")
        
        # Create a new instance
        detector = cls(
            contamination=model_data['contamination'],
            random_state=model_data['random_state']
        )
        
        # Restore model states
        detector.scaler = model_data['scaler']
        detector.isolation_forest = model_data['isolation_forest']
        detector.ocsvm = model_data['ocsvm']
        
        logger.info(f"Anomaly detection models loaded from {filepath}_anomaly_model.pkl")
        return detector