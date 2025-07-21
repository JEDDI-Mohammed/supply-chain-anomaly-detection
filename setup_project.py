#!/usr/bin/env python
"""
Setup script to create the project directory structure and initialize empty Python modules.

This script creates the necessary directory structure for the supply chain anomaly detection
project and initializes empty Python modules with the appropriate imports and docstrings.
"""

import os
import shutil

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content=""):
    """Create file with the given content."""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def create_gitkeep(path):
    """Create .gitkeep file in the given directory."""
    gitkeep_path = os.path.join(path, '.gitkeep')
    create_file(gitkeep_path)

def create_init_file(path):
    """Create __init__.py file in the given directory."""
    init_path = os.path.join(path, '__init__.py')
    create_file(init_path)

def main():
    """Create project directory structure and initialize empty Python modules."""
    # Create main directories
    directories = [
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/visualization',
        'src/utils',
        'data/raw',
        'data/processed',
        'data/sample',
        'notebooks',
        'mlflow',
        'scripts',
        'tests',
        'docs',
        'config',
        'models'
    ]

    for directory in directories:
        create_directory(directory)
        if directory.startswith('src/'):
            create_init_file(directory)

    # Create .gitkeep files
    gitkeep_dirs = [
        'data/raw',
        'data/processed',
        'models'
    ]

    for directory in gitkeep_dirs:
        create_gitkeep(directory)

    # Create empty module files
    module_files = {
        'src/data/preprocess.py': 
        '"""\nData preprocessing for supply chain anomaly detection.\n\nThis module handles loading, cleaning, and feature engineering for supply chain data.\n"""\n\n',
        
        'src/models/anomaly_detection.py': 
        '"""\nAnomaly detection for supply chain data.\n\nThis module implements ensemble-based anomaly detection for supply chain metrics.\n"""\n\n',
        
        'src/models/issue_classification.py': 
        '"""\nIssue classification for supply chain anomalies.\n\nThis module classifies detected anomalies into specific issue types.\n"""\n\n',
        
        'src/models/recommendation.py': 
        '"""\nRecommendation generation for supply chain issues.\n\nThis module generates actionable recommendations for detected issues.\n"""\n\n',
        
        'src/visualization/visualize.py': 
        '"""\nVisualization tools for supply chain anomaly detection.\n\nThis module provides visualization functions for anomalies and issue types.\n"""\n\n',
        
        'src/models/sc_issue_detection.py': 
        '"""\nSupply Chain Issue Detection system.\n\nThis module integrates all components of the supply chain issue detection system.\n"""\n\n',
        
        'mlflow/mlflow_utils.py': 
        '"""\nMLflow utilities for the Supply Chain Issue Detection project.\n\nThis module provides wrapper functions for MLflow integration.\n"""\n\n',
        
        'scripts/train_model.py': 
        '#!/usr/bin/env python\n"""\nTraining script for supply chain anomaly detection models.\n\nThis script trains the complete supply chain anomaly detection pipeline.\n"""\n\n',
        
        'config/model_params.yml': 
        '# Supply Chain Anomaly Detection Model Configuration\n\n# Anomaly detection parameters\nanomaly_detection:\n  contamination: 0.05\n\n'
    }

    for path, content in module_files.items():
        create_file(path, content)

    # Make scripts executable
    os.chmod('scripts/train_model.py', 0o755)
    
    print("\nProject structure has been created successfully!")
    print("Run the following commands to initialize Git and install the package:")
    print("\ngit init")
    print("pip install -e .")

if __name__ == '__main__':
    main()