"""
MLflow utilities for the Supply Chain Issue Detection project.

This module provides wrapper functions for MLflow integration.
"""

"""
MLflow utilities for the Supply Chain Issue Detection project.

This module provides wrapper functions and utilities to integrate the
project with MLflow for experiment tracking, model management, and deployment.
"""

import os
import logging
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import MLflow with conditional handling for environments without it
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Create mock objects for graceful degradation
    class MockMlflow:
        """Mock MLflow for environments where MLflow is not installed."""
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
            
    mlflow = MockMlflow()
    MlflowClient = lambda: None
    
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check MLflow availability
if not MLFLOW_AVAILABLE:
    logger.warning("MLflow not available. Install with 'pip install mlflow' for experiment tracking.")


def is_running_in_databricks():
    """
    Check if the code is running in a Databricks environment.
    
    Returns:
        bool: True if running in Databricks, False otherwise
    """
    return 'DATABRICKS_RUNTIME_VERSION' in os.environ


def start_run(experiment_name=None, run_name=None):
    """
    Start an MLflow run with an optional experiment name and run name.
    
    Args:
        experiment_name (str, optional): Name of the experiment to use
        run_name (str, optional): Name for the run
            
    Returns:
        mlflow.ActiveRun: MLflow run object or a context manager if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        # Return a dummy context manager if MLflow is not available
        class DummyContextManager:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            @property
            def info(self): return type('obj', (object,), {'run_id': 'dummy_run_id'})
        
        logger.warning("MLflow not available. Running without experiment tracking.")
        return DummyContextManager()
    
    try:
        # Set the experiment if provided
        if experiment_name:
            try:
                # Try to get the experiment
                experiment = mlflow.get_experiment_by_name(experiment_name)
                
                # Create the experiment if it doesn't exist
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                    
                mlflow.set_experiment(experiment_name)
                logger.info(f"Using MLflow experiment: {experiment_name}")
            except Exception as e:
                logger.error(f"Error setting MLflow experiment: {str(e)}")
                mlflow.set_experiment("supply_chain_anomaly_detection")
        
        # Generate a default run name if none provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"supply_chain_run_{timestamp}"
            
        # Start the run
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id} with name: {run_name}")
        return run
        
    except Exception as e:
        logger.error(f"Error starting MLflow run: {str(e)}")
        # Return a dummy context manager on error
        class ErrorContextManager:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            @property
            def info(self): return type('obj', (object,), {'run_id': 'error_run_id'})
        
        return ErrorContextManager()


def log_parameters(params_dict):
    """
    Log multiple parameters to MLflow.
    
    Args:
        params_dict (dict): Dictionary of parameter names and values to log
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Parameters not logged.")
        return
        
    try:
        # Log each parameter
        for key, value in params_dict.items():
            # Handle complex types that MLflow can't natively log
            if isinstance(value, (list, dict, set)):
                mlflow.log_param(key, str(value))
            else:
                mlflow.log_param(key, value)
        
        logger.info(f"Logged {len(params_dict)} parameters to MLflow")
    except Exception as e:
        logger.error(f"Error logging parameters to MLflow: {str(e)}")


def log_metrics(metrics_dict, step=None):
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values to log
        step (int, optional): Step value to associate with metrics
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Metrics not logged.")
        return
        
    try:
        # Log each metric
        for key, value in metrics_dict.items():
            # Make sure the value is numeric
            try:
                numeric_value = float(value)
                mlflow.log_metric(key, numeric_value, step=step)
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-numeric metric: {key}={value}")
        
        logger.info(f"Logged {len(metrics_dict)} metrics to MLflow")
    except Exception as e:
        logger.error(f"Error logging metrics to MLflow: {str(e)}")


def log_model(model, model_name, conda_env=None, custom_metrics=None, code_paths=None):
    """
    Log a model to MLflow.
    
    Args:
        model: Model object to log
        model_name (str): Name for the logged model
        conda_env (dict, optional): Conda environment specification
        custom_metrics (dict, optional): Custom metrics to log with the model
        code_paths (list, optional): List of local filesystem paths to Python file dependencies
    
    Returns:
        str: URI of the logged model or None if logging failed
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Model not logged.")
        return None
        
    try:
        # Log custom metrics if provided
        if custom_metrics:
            log_metrics(custom_metrics)
        
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            conda_env=conda_env,
            code_paths=code_paths
        )
        
        logger.info(f"Logged model to MLflow: {model_info.model_uri}")
        return model_info.model_uri
    except Exception as e:
        logger.error(f"Error logging model to MLflow: {str(e)}")
        return None


def log_figure(fig, artifact_path):
    """
    Log a matplotlib figure to MLflow.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to log
        artifact_path (str): Path to store the figure within the run
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Figure not logged.")
        return
        
    try:
        # Create a temporary file to save the figure
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            fig.savefig(temp_file.name, bbox_inches='tight', dpi=300)
            
        # Log the file as an artifact
        mlflow.log_artifact(temp_file.name, artifact_path)
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        logger.info(f"Logged figure to MLflow: {artifact_path}")
    except Exception as e:
        logger.error(f"Error logging figure to MLflow: {str(e)}")


def log_dataframe(df, artifact_path, file_format='csv'):
    """
    Log a pandas DataFrame to MLflow.
    
    Args:
        df (pd.DataFrame): DataFrame to log
        artifact_path (str): Path to store the DataFrame within the run
        file_format (str, optional): Format to save the DataFrame (csv or parquet)
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. DataFrame not logged.")
        return
        
    try:
        # Create a temporary file to save the DataFrame
        with tempfile.NamedTemporaryFile(
            suffix=f'.{file_format}', delete=False
        ) as temp_file:
            if file_format.lower() == 'csv':
                df.to_csv(temp_file.name, index=False)
            elif file_format.lower() == 'parquet':
                df.to_parquet(temp_file.name, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
        # Log the file as an artifact
        mlflow.log_artifact(temp_file.name, artifact_path)
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        logger.info(f"Logged DataFrame with {len(df)} rows to MLflow: {artifact_path}")
    except Exception as e:
        logger.error(f"Error logging DataFrame to MLflow: {str(e)}")


def log_dict_as_json(data_dict, artifact_path):
    """
    Log a dictionary as a JSON file to MLflow.
    
    Args:
        data_dict (dict): Dictionary to log
        artifact_path (str): Path to store the JSON file within the run
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Dictionary not logged.")
        return
        
    try:
        # Create a temporary file to save the dictionary as JSON
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            with open(temp_file.name, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
        # Log the file as an artifact
        mlflow.log_artifact(temp_file.name, artifact_path)
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        logger.info(f"Logged dictionary as JSON to MLflow: {artifact_path}")
    except Exception as e:
        logger.error(f"Error logging dictionary to MLflow: {str(e)}")


def register_model(model_uri, name, description=None, tags=None):
    """
    Register a model in the MLflow Model Registry.
    
    Args:
        model_uri (str): URI of the model to register
        name (str): Name to register the model with
        description (str, optional): Description for the model version
        tags (dict, optional): Tags to associate with the model version
    
    Returns:
        ModelVersion: Registered model version object or None if registration failed
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Model not registered.")
        return None
        
    try:
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=name
        )
        
        client = MlflowClient()
        
        # Add description if provided
        if description:
            client.update_model_version(
                name=name,
                version=model_version.version,
                description=description
            )
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        logger.info(f"Registered model in MLflow Model Registry: {name} (version: {model_version.version})")
        return model_version
    except Exception as e:
        logger.error(f"Error registering model in MLflow Model Registry: {str(e)}")
        return None


def transition_model_stage(name, version, stage):
    """
    Transition a registered model to a new stage.
    
    Args:
        name (str): Name of the registered model
        version (int): Version of the model to transition
        stage (str): New stage for the model (Staging, Production, or Archived)
    
    Returns:
        ModelVersion: Updated model version object or None if transition failed
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Model stage not transitioned.")
        return None
        
    try:
        client = MlflowClient()
        model_version = client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Transitioned model {name} version {version} to stage: {stage}")
        return model_version
    except Exception as e:
        logger.error(f"Error transitioning model stage: {str(e)}")
        return None


def load_model(model_uri):
    """
    Load a model from MLflow.
    
    Args:
        model_uri (str): URI of the model to load
    
    Returns:
        object: Loaded model or None if loading failed
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Model not loaded.")
        return None
        
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from MLflow: {model_uri}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {str(e)}")
        return None


def get_latest_model_version(model_name, stage=None):
    """
    Get the latest version of a registered model.
    
    Args:
        model_name (str): Name of the registered model
        stage (str, optional): Filter by stage (Staging, Production, or Archived)
    
    Returns:
        ModelVersion: Latest model version object or None if not found
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Cannot get latest model version.")
        return None
        
    try:
        client = MlflowClient()
        
        # Get all versions of the model
        versions = client.get_latest_versions(model_name, stages=[stage] if stage else None)
        
        if not versions:
            logger.warning(f"No versions found for model: {model_name}")
            return None
        
        # Sort by version number and get the latest
        latest_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        
        logger.info(f"Found latest version of model {model_name}: {latest_version.version}")
        return latest_version
    except Exception as e:
        logger.error(f"Error getting latest model version: {str(e)}")
        return None


def databricks_notebook_exit(metrics):
    """
    Exit a Databricks notebook with metrics for workflow integration.
    
    This function is specifically for use in Databricks notebooks that are run
    as a job step, allowing metrics to be passed to the Databricks workflow.
    
    Args:
        metrics (dict): Dictionary of metrics to return from the notebook
    """
    if not is_running_in_databricks():
        logger.warning("Not running in Databricks. Notebook exit with metrics skipped.")
        return
        
    try:
        import IPython
        
        # Convert all values to strings (Databricks requirement)
        string_metrics = {k: str(v) for k, v in metrics.items()}
        
        # Create the exit JSON
        exit_json = json.dumps(string_metrics)
        
        # Use the IPython dbutils to exit with the metrics
        IPython.get_ipython().run_line_magic(
            "scala", 
            f"dbutils.notebook.exit(\"{exit_json}\")"
        )
        
        logger.info("Exited Databricks notebook with metrics.")
    except Exception as e:
        logger.error(f"Error exiting Databricks notebook with metrics: {str(e)}")