#!/usr/bin/env python
"""
Training script for supply chain anomaly detection models with MLflow tracking.

This script trains the complete supply chain anomaly detection pipeline
and logs results to MLflow. It can be run from the command line with
various parameters to control the training process.
"""

import os
import sys
import argparse
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.sc_issue_detection import SupplyChainIssueDetection
from mlflow import mlflow_utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train supply chain anomaly detection model with MLflow tracking'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/model_params.yml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to input data CSV'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='models',
        help='Output directory for models'
    )
    
    parser.add_argument(
        '--use-llm', 
        action='store_true',
        help='Use LLM for generating recommendations'
    )
    
    parser.add_argument(
        '--experiment-name', 
        type=str, 
        default='supply_chain_anomaly_detection',
        help='MLflow experiment name'
    )
    
    parser.add_argument(
        '--register-model', 
        action='store_true',
        help='Register the model in MLflow Model Registry'
    )
    
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='supply_chain_anomaly_detector',
        help='Name for the registered model in MLflow Model Registry'
    )
    
    parser.add_argument(
        '--databricks', 
        action='store_true',
        help='Running in Databricks environment'
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        # Use default configuration if file is not available
        config = {
            'anomaly_detection': {
                'contamination': 0.05,
                'random_state': 42
            },
            'issue_classification': {
                'use_ml_classification': True
            },
            'recommendation': {
                'use_llm': args.use_llm,
                'high_priority_threshold': 0.7
            }
        }
        logger.info("Using default configuration")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Set up model and training parameters
    contamination = config['anomaly_detection'].get('contamination', 0.05)
    random_state = config['anomaly_detection'].get('random_state', 42)
    use_llm = config['recommendation'].get('use_llm', args.use_llm)
    use_ml_classification = config['issue_classification'].get('use_ml_classification', True)
    
    # Get API key from environment if using LLM
    api_key = os.environ.get('OPENAI_API_KEY') if use_llm else None
    
    # Start MLflow run
    with mlflow_utils.start_run(experiment_name=args.experiment_name) as run:
        run_id = run.info.run_id
        
        # Log parameters
        params = {
            'contamination': contamination,
            'random_state': random_state,
            'use_llm': use_llm,
            'use_ml_classification': use_ml_classification,
            'data_path': args.data,
            'input_filename': os.path.basename(args.data)
        }
        
        mlflow_utils.log_parameters(params)
        logger.info(f"Logged parameters to MLflow run: {run_id}")
        
        # Initialize model
        detector = SupplyChainIssueDetection(
            use_llm=use_llm,
            api_key=api_key,
            contamination=contamination,
            random_state=random_state
        )
        
        try:
            # Load and preprocess data
            data = detector.load_data(args.data)
            mlflow_utils.log_metrics({'data_rows': len(data), 'data_columns': len(data.columns)})
            
            detector.preprocess_data()
            logger.info("Data preprocessing completed")
            
            # Train anomaly detector
            detector.train_anomaly_detector()
            logger.info("Anomaly detection completed")
            
            # Log anomaly metrics
            anomaly_count = detector.data['is_anomaly'].sum()
            anomaly_pct = (anomaly_count / len(detector.data)) * 100
            
            anomaly_metrics = {
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_pct,
                'if_anomaly_count': detector.data['if_anomaly'].sum(),
                'lof_anomaly_count': detector.data['lof_anomaly'].sum() if 'lof_anomaly' in detector.data.columns else 0,
                'ocsvm_anomaly_count': detector.data['ocsvm_anomaly'].sum()
            }
            
            mlflow_utils.log_metrics(anomaly_metrics)
            
            # Create and log anomaly visualization
            fig1 = detector.visualize_anomalies('WeeksOfStockT1', 'TargetAchievement')
            mlflow_utils.log_figure(fig1, 'visualizations')
            
            fig2 = detector.visualize_anomalies('SellThruToRatio', 'InventoryTurnoverRate')
            mlflow_utils.log_figure(fig2, 'visualizations')
            
            # Classify issues
            anomalies = detector.classify_issues(use_ml=use_ml_classification)
            logger.info("Issue classification completed")
            
            # Log issue classification metrics
            issue_counts = anomalies['issue_type'].value_counts().to_dict()
            issue_metrics = {f"issue_count_{issue}": count for issue, count in issue_counts.items()}
            mlflow_utils.log_metrics(issue_metrics)
            
            # Generate recommendations
            recommendations = detector.generate_recommendations(anomalies)
            logger.info("Recommendation generation completed")
            
            # Log recommendations as artifact
            mlflow_utils.log_dataframe(recommendations, 'recommendations', 'csv')
            
            # Create and log PCA visualization if enough anomalies
            if len(anomalies) >= 3:
                try:
                    pca, pca_anomalies = detector.visualize_with_pca()
                    
                    # Analyze PCA results
                    pca_analysis = detector.analyze_pca_results(pca, pca_anomalies)
                    
                    # Log PCA metrics
                    pca_metrics = {
                        'pca_explained_variance': pca_analysis['total_explained_variance']
                    }
                    
                    if pca_analysis['silhouette_score'] is not None:
                        pca_metrics['silhouette_score'] = pca_analysis['silhouette_score']
                    
                    mlflow_utils.log_metrics(pca_metrics)
                    
                    # Log feature importance as artifact
                    feature_importance = pd.DataFrame(pca_analysis['feature_importance'])
                    mlflow_utils.log_dataframe(feature_importance, 'feature_importance', 'csv')
                    
                except Exception as e:
                    logger.error(f"Error during PCA visualization: {str(e)}")
            
            # Save the models locally
            model_path = os.path.join(args.output, f"supply_chain_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            detector.save_models(model_path)
            logger.info(f"Models saved to {model_path}")
            
            # Log model to MLflow
            model_info = mlflow_utils.log_model(
                detector, 
                "supply_chain_detector",
                custom_metrics=anomaly_metrics
            )
            
            # Register the model if requested
            if args.register_model and model_info:
                try:
                    model_version = mlflow_utils.register_model(
                        model_uri=model_info,
                        name=args.model_name,
                        description=f"Supply Chain Anomaly Detection model trained on {os.path.basename(args.data)}"
                    )
                    
                    logger.info(f"Model registered as {args.model_name} version {model_version.version}")
                    
                    # Add tags to the registered model
                    if model_version:
                        tags = {
                            'contamination': str(contamination),
                            'anomaly_count': str(anomaly_count),
                            'anomaly_percentage': f"{anomaly_pct:.2f}%",
                            'data_source': os.path.basename(args.data),
                            'training_date': datetime.now().strftime('%Y-%m-%d')
                        }
                        
                        # Add tags to the model version
                        mlflow_utils.add_model_version_tags(args.model_name, model_version.version, tags)
                except Exception as e:
                    logger.error(f"Error registering model: {str(e)}")
            
            # If running in Databricks, return metrics for the job
            if args.databricks:
                metrics = {
                    'anomaly_count': anomaly_count,
                    'anomaly_percentage': anomaly_pct,
                    'run_id': run_id
                }
                
                mlflow_utils.databricks_notebook_exit(metrics)
            
            logger.info(f"Training completed successfully. MLflow run ID: {run_id}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

if __name__ == '__main__':
    main()'] = pca_analysis['silhouette_score