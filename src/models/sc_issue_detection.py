"""
Supply Chain Issue Detection system.

This module integrates all components of the supply chain issue detection system.
"""

"""
Supply Chain Issue Detection system.

This module integrates all components of the supply chain issue detection system,
providing a facade for the entire process from data loading to recommendation generation.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

from src.data.preprocess import DataPreprocessor
from src.models.anomaly_detection import AnomalyDetector
from src.models.issue_classification import IssueClassifier
from src.models.recommendation import RecommendationGenerator
from src.visualization.visualize import AnomalyVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupplyChainIssueDetection:
    """
    Integrated supply chain issue detection system.
    
    This class serves as a facade for the entire supply chain issue detection
    process, integrating data preprocessing, anomaly detection, issue classification,
    recommendation generation, and visualization components.
    
    Attributes:
        preprocessor (DataPreprocessor): Component for data loading and preprocessing
        anomaly_detector (AnomalyDetector): Component for anomaly detection
        issue_classifier (IssueClassifier): Component for issue classification
        recommendation_generator (RecommendationGenerator): Component for recommendation generation
        visualizer (AnomalyVisualizer): Component for visualization
        data (pd.DataFrame): The loaded and processed data
    """
    
    def __init__(self, use_llm=None, api_key=None, contamination=0.05, random_state=42):
        """
        Initialize the supply chain issue detection system.
        
        Args:
            use_llm (bool, optional): Whether to use LLM for recommendations
            api_key (str, optional): API key for LLM service
            contamination (float, optional): Expected proportion of anomalies
            random_state (int, optional): Random seed for reproducibility
        """
        self.preprocessor = DataPreprocessor()
        self.anomaly_detector = AnomalyDetector(
            contamination=contamination,
            random_state=random_state
        )
        self.issue_classifier = IssueClassifier()
        self.recommendation_generator = RecommendationGenerator(
            use_llm=use_llm,
            api_key=api_key
        )
        self.visualizer = AnomalyVisualizer()
        self.data = None
        
        logger.info("Supply Chain Issue Detection system initialized")
    
    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: The loaded data
        """
        self.data = self.preprocessor.load_data(filepath)
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the data and engineer features.
        
        Returns:
            pd.DataFrame: The preprocessed data with engineered features
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
            
        self.data = self.preprocessor.preprocess_data(self.data)
        self.data = self.preprocessor.engineer_features(self.data)
        
        return self.data
    
    def train_anomaly_detector(self):
        """
        Train the anomaly detection models and identify anomalies.
        
        Returns:
            pd.DataFrame: The data with added anomaly flags and scores
        """
        if self.data is None:
            raise ValueError("Data not preprocessed. Please call preprocess_data() first.")
            
        self.data = self.anomaly_detector.fit_transform(
            self.data, 
            self.preprocessor.numerical_features
        )
        
        return self.data
    
    def classify_issues(self, use_ml=True):
        """
        Classify anomalies into specific issue types.
        
        Args:
            use_ml (bool, optional): Whether to use ML-based classification
                in addition to rule-based classification
                
        Returns:
            pd.DataFrame: DataFrame containing only the anomalies with issue types
        """
        if 'is_anomaly' not in self.data.columns:
            raise ValueError("Anomaly detection not performed. Please call train_anomaly_detector() first.")
            
        if use_ml:
            anomalies = self.issue_classifier.classify_issues(
                self.data,
                self.preprocessor.numerical_features,
                self.preprocessor.categorical_features
            )
        else:
            anomalies = self.issue_classifier.classify_with_rules(self.data)
            
        return anomalies
    
    def generate_recommendations(self, anomalies):
        """
        Generate recommendations for identified anomalies.
        
        Args:
            anomalies (pd.DataFrame): DataFrame containing anomalies with issue types
            
        Returns:
            pd.DataFrame: Anomalies with added recommendation columns
        """
        return self.recommendation_generator.generate_recommendations(anomalies)
    
    def visualize_anomalies(self, column1, column2, figsize=(10, 6)):
        """
        Create a scatter plot visualization of anomalies.
        
        Args:
            column1 (str): Name of the column for the x-axis
            column2 (str): Name of the column for the y-axis
            figsize (tuple, optional): Figure size in inches
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if 'is_anomaly' not in self.data.columns:
            raise ValueError("Anomaly detection not performed. Please call train_anomaly_detector() first.")
            
        return self.visualizer.visualize_anomalies(self.data, column1, column2, figsize)
    
    def visualize_with_pca(self, figsize=(18, 8)):
        """
        Create a PCA visualization of anomalies and their issue types.
        
        Args:
            figsize (tuple, optional): Figure size in inches
            
        Returns:
            tuple: (PCA model, DataFrame with PCA components added)
        """
        if 'is_anomaly' not in self.data.columns or 'issue_type' not in self.data.columns:
            raise ValueError("Anomaly detection and issue classification must be performed first.")
            
        anomalies = self.data[self.data['is_anomaly'] == 1].copy()
        
        return self.visualizer.visualize_with_pca(
            anomalies,
            self.preprocessor.numerical_features,
            figsize
        )
    
    def analyze_pca_results(self, pca, pca_anomalies):
        """
        Analyze PCA results to validate issue categorization quality.
        
        Args:
            pca: Fitted PCA model
            pca_anomalies (pd.DataFrame): DataFrame with PCA components added
            
        Returns:
            dict: Analysis results
        """
        return self.visualizer.analyze_pca_results(pca, pca_anomalies)
    
    def save_models(self, filepath_prefix):
        """
        Save all trained models to disk.
        
        Args:
            filepath_prefix (str): Path prefix for saving model files
        """
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        
        # Save preprocessor state
        self.preprocessor.save_state(f"{filepath_prefix}_preprocessor.pkl")
        
        # Save anomaly detector
        self.anomaly_detector.save_model(filepath_prefix)
        
        # Save issue classifier
        self.issue_classifier.save_model(filepath_prefix)
        
        # Save the pipeline configuration
        config = {
            'numerical_features': self.preprocessor.numerical_features,
            'categorical_features': self.preprocessor.categorical_features,
            'engineered_features': self.preprocessor.engineered_features,
            'contamination': self.anomaly_detector.contamination,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(config, f"{filepath_prefix}_config.pkl")
        
        logger.info(f"Models and configuration saved with prefix: {filepath_prefix}")
    
    @classmethod
    def load_models(cls, filepath_prefix, use_llm=None, api_key=None):
        """
        Load trained models from disk.
        
        Args:
            filepath_prefix (str): Path prefix where model files are stored
            use_llm (bool, optional): Whether to use LLM for recommendations
            api_key (str, optional): API key for LLM service
            
        Returns:
            SupplyChainIssueDetection: Initialized system with loaded models
        """
        try:
            # Load configuration
            config = joblib.load(f"{filepath_prefix}_config.pkl")
            
            # Create a new instance with the same contamination
            instance = cls(
                use_llm=use_llm,
                api_key=api_key,
                contamination=config.get('contamination', 0.05)
            )
            
            # Load preprocessor state
            instance.preprocessor.load_state(f"{filepath_prefix}_preprocessor.pkl")
            
            # Load anomaly detector
            instance.anomaly_detector = AnomalyDetector.load_model(filepath_prefix)
            
            # Load issue classifier
            instance.issue_classifier = IssueClassifier.load_model(filepath_prefix)
            
            logger.info(f"Models loaded from prefix: {filepath_prefix}")
            return instance
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def process_new_data(self, filepath):
        """
        Process new data using pre-trained models.
        
        This method loads new data, preprocesses it, detects anomalies,
        classifies issues, and generates recommendations in a single call.
        
        Args:
            filepath (str): Path to the new data CSV file
            
        Returns:
            tuple: (Full DataFrame with anomalies, DataFrame with only anomalies and recommendations)
        """
        # Load and preprocess data
        self.load_data(filepath)
        self.preprocess_data()
        
        # Detect anomalies using pre-trained models
        self.data = self.anomaly_detector.predict(
            self.data,
            self.preprocessor.numerical_features
        )
        
        # Classify issues
        anomalies = self.issue_classifier.classify_with_rules(self.data)
        
        # Generate recommendations
        recommendations = self.recommendation_generator.generate_recommendations(anomalies)
        
        logger.info(f"Processed new data from {filepath}: {len(recommendations)} anomalies detected")
        
        return self.data, recommendations
    
    def run_full_pipeline(self, filepath, save_models_path=None, use_ml=True):
        """
        Run the complete analysis pipeline on a dataset.
        
        This method executes the full workflow from data loading to
        recommendation generation in a single call.
        
        Args:
            filepath (str): Path to the data CSV file
            save_models_path (str, optional): Path prefix to save models
            use_ml (bool, optional): Whether to use ML-based classification
            
        Returns:
            tuple: (Full DataFrame, DataFrame with recommendations)
        """
        # Load and preprocess data
        self.load_data(filepath)
        self.preprocess_data()
        
        # Detect anomalies
        self.train_anomaly_detector()
        
        # Classify issues
        anomalies = self.classify_issues(use_ml=use_ml)
        
        # Generate recommendations
        recommendations = self.recommendation_generator.generate_recommendations(anomalies)
        
        # Save models if path provided
        if save_models_path:
            self.save_models(save_models_path)
        
        logger.info(f"Full pipeline completed for {filepath}: {len(recommendations)} anomalies detected")
        
        return self.data, recommendations