"""
Data preprocessing for supply chain anomaly detection.

This module handles loading, cleaning, and feature engineering for supply chain data.
"""

"""
Data preprocessing for supply chain anomaly detection.

This module handles loading, cleaning, and feature engineering for supply chain data.
"""

import pandas as pd
import numpy as np
import logging
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing for supply chain metrics.
    
    This class handles loading data from CSV files, cleaning and preprocessing 
    the data, and engineering additional features to improve anomaly detection.
    
    Attributes:
        data (pd.DataFrame): The loaded data
        numerical_features (list): List of numerical feature column names
        categorical_features (list): List of categorical feature column names
        engineered_features (list): List of engineered feature column names
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.data = None
        
        # Define feature lists
        self.numerical_features = [
            'sell_thru', 'dollars', 'shipments', 'inventory', 'aged_inventory',
            't1_wos', 'sell_to_t2', 't2_wos', 't2_inventory_t2',
             'Eligible_Sellthru', 'Target_Sellthru',
            'Achiev_Sellthru', 'Eligible_Shipments', 'Target_Shipments',
            'Achiev_Shipments'
        ]
        
        self.categorical_features = ['Global_Distributor_Group_Name','reporter_name', 'reporter_hq_id', 'PRODUCT_GROUP_BMT','fiscal_year_quarter', 'product_number', 'reporter_country_code']
        
        self.engineered_features = [
            'SellThruToRatio', 
            'InventoryTurnoverRate', 
            'TargetAchievement',
            'TargetAchievement_ship', 
            'SupplyChainEfficiency', 
            'AgedInventoryPct'
        ]

    def load_data(self, filepath):
        """
        Load data from a CSV file or a Hive table.
        
        Args:
            filepath (str): Path to the CSV file or name of the Hive table
            
        Returns:
            pd.DataFrame: The loaded data
        """
        try:
            # Check if the filepath is for a CSV file or a Hive table
            if filepath.endswith('.csv') or '/' in filepath:
                # Load data from CSV file
                self.data = pd.read_csv(filepath)
                logger.info(f"Loaded CSV data from {filepath} with shape: {self.data.shape}")
            else:
                # Assume it's a Hive table path
                from pyspark.sql import SparkSession
                
                # Get or create a Spark session
                spark = SparkSession.builder.getOrCreate()
                
                # Read from Hive table
                spark_df = spark.read.table(filepath)
                
                # Convert to pandas DataFrame
                self.data = spark_df.toPandas()
                logger.info(f"Loaded Hive table data from {filepath} with shape: {self.data.shape}")
            
            # Basic validation of required columns
            if hasattr(self, 'numerical_features') and hasattr(self, 'categorical_features'):
                missing_cols = [col for col in self.numerical_features + self.categorical_features 
                            if col not in self.data.columns]
                
                if missing_cols:
                    logger.warning(f"Missing columns in data: {', '.join(missing_cols)}")
            else:
                logger.warning("numerical_features or categorical_features not initialized. Skipping column validation.")
                
            return self.data
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise

    def preprocess_data(self, data=None):
        """
        Clean and preprocess the data.
        
        Args:
            data (pd.DataFrame, optional): Data to preprocess. If None, uses self.data
            
        Returns:
            pd.DataFrame: The preprocessed data
        """
        if data is not None:
            self.data = data
            
        if self.data is None:
            raise ValueError("No data to preprocess. Load data first.")
        
        # Make a copy to avoid modifying the original
        preprocessed_data = self.data.copy()
        
        # Handle missing values in numerical features
        for col in self.numerical_features:
            if col in preprocessed_data.columns:
                # Replace missing values with median
                median_value = preprocessed_data[col].median()
                preprocessed_data[col] = preprocessed_data[col].fillna(median_value)
                
                # Log percentage of missing values replaced
                missing_pct = (self.data[col].isna().sum() / len(self.data)) * 100
                if missing_pct > 0:
                    logger.info(f"Replaced {missing_pct:.2f}% missing values in '{col}' with median ({median_value})")
        
        # Handle missing values in categorical features
        for col in self.categorical_features:
            if col in preprocessed_data.columns:
                # Replace missing values with mode
                mode_value = preprocessed_data[col].mode().iloc[0]
                preprocessed_data[col] = preprocessed_data[col].fillna(mode_value)
                
                # Log percentage of missing values replaced
                missing_pct = (self.data[col].isna().sum() / len(self.data)) * 100
                if missing_pct > 0:
                    logger.info(f"Replaced {missing_pct:.2f}% missing values in '{col}' with mode ({mode_value})")
        
        # Convert categorical columns to appropriate types
        for col in self.categorical_features:
            if col in preprocessed_data.columns:
                preprocessed_data[col] = preprocessed_data[col].astype(str)
        
        # Check for outliers in numerical features
        for col in self.numerical_features:
            if col in preprocessed_data.columns:
                # Identify outliers using IQR method
                q1 = preprocessed_data[col].quantile(0.25)
                q3 = preprocessed_data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                
                # Count outliers
                outliers = ((preprocessed_data[col] < lower_bound) | 
                           (preprocessed_data[col] > upper_bound)).sum()
                
                # Log percentage of outliers
                outlier_pct = (outliers / len(preprocessed_data)) * 100
                if outlier_pct > 5:  # Only log if significant
                    logger.info(f"Feature '{col}' has {outlier_pct:.2f}% outliers")
                    
                # Note: We don't cap outliers here as they might be actual anomalies
        
        logger.info("Data preprocessing completed")
        return preprocessed_data
    
    def engineer_features(self, data=None):
        """
        Create engineered features to improve anomaly detection.
        
        Args:
            data (pd.DataFrame, optional): Data to process. If None, uses self.data
            
        Returns:
            pd.DataFrame: Data with additional engineered features
        """
        if data is not None:
            self.data = data
            
        if self.data is None:
            raise ValueError("No data to process. Preprocess data first.")
        
        # Make a copy to avoid modifying the original
        result_df = self.data.copy()
        
        try:
            # 1. Sell-Through to Sell-To ratio (efficiency of partner sales)
            # Interpretation: Measures how effectively distributors are selling to end customers
            # High values (>1) indicate distributors are selling more than they're buying (reducing inventory)
            # Low values (<0.7) may indicate channel stuffing or sell-through issues
            result_df['SellThruToRatio'] = result_df['sell_thru'] / result_df['sell_to_t2'].replace(0, 1)
            
            # 2. Inventory turnover rates
            # Interpretation: How quickly inventory is sold and replaced
            # Higher values indicate faster inventory movement (generally positive)
            result_df['InventoryTurnoverRate'] = result_df['sell_to_t2'] / result_df['inventory'].replace(0, 1)
            
            # 3. Target achievement percentages
            # Interpretation: Sales performance against targets
            # Values close to 100% indicate accurate forecasting and good execution
            result_df['TargetAchievement'] = (result_df['Achiev_Sellthru'] / result_df['Target_Sellthru'].replace(0, 1)) * 100
            result_df['TargetAchievement_ship'] = (result_df['Achiev_Shipments'] / result_df['Target_Shipments'].replace(0, 1)) * 100


            # 4. Supply efficiency: Shipment vs Backlog
            # Interpretation: Ability to fulfill orders against existing backlog
            # Higher values indicate better supply chain performance
            result_df['SupplyChainEfficiency'] = result_df['shipments'] / (result_df['Eligible_Shipments'] + 1)#result_df['Shipments'] / (result_df['Backlog'] + 1)
            
            # 5. Aged Inventory Percentage
            # Interpretation: Percentage of inventory at risk of obsolescence
            # Higher values indicate potential inventory health issues
            result_df['AgedInventoryPct'] = (result_df['aged_inventory'] / result_df['inventory'].replace(0, 1)) * 100
            
            # Log summary statistics for the new features
            for feature in self.engineered_features:
                logger.info(f"Engineered feature '{feature}': mean={result_df[feature].mean():.2f}, " +
                           f"min={result_df[feature].min():.2f}, max={result_df[feature].max():.2f}")
            
            logger.info(f"Feature engineering completed: Added {len(self.engineered_features)} new features")
            
            # Update numerical features list to include engineered features
            self.numerical_features.extend(self.engineered_features)
            
            return result_df
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            # Return the original data if engineering fails
            return result_df
    
    def save_state(self, filepath):
        """
        Save the preprocessor state to disk.
        
        Args:
            filepath (str): Path to save the state
        """
        state = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'engineered_features': self.engineered_features
        }
        
        joblib.dump(state, filepath)
        logger.info(f"Preprocessor state saved to {filepath}")
    
    def load_state(self, filepath):
        """
        Load the preprocessor state from disk.
        
        Args:
            filepath (str): Path to the saved state
        """
        try:
            state = joblib.load(filepath)
            
            self.numerical_features = state.get('numerical_features', self.numerical_features)
            self.categorical_features = state.get('categorical_features', self.categorical_features)
            self.engineered_features = state.get('engineered_features', self.engineered_features)
            
            logger.info(f"Preprocessor state loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading preprocessor state: {str(e)}")
            logger.info("Using default feature lists")