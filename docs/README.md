# Supply Chain Issue Detection

![Supply Chain](https://img.shields.io/badge/Supply%20Chain-Anomaly%20Detection-red)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-Integrated-orange)
![License](https://img.shields.io/badge/License-HP-blue)

A machine learning system for detecting and classifying supply chain anomalies and generating targeted recommendations for supply chain managers and executives.

## 🔍 Project Overview

This project implements an ensemble-based anomaly detection system for supply chain data with the following capabilities:

- **Anomaly Detection**: Identify unusual patterns in supply chain metrics using ensemble methods
- **Issue Classification**: Categorize detected anomalies into specific issue types
- **Root Cause Analysis**: Determine likely causes for identified issues
- **Recommendation Generation**: Provide business-focused, actionable recommendations
- **Visualization**: Create insightful visualizations of anomalies and their patterns
- **MLflow Integration**: Track experiments and manage models with MLflow

## 📊 Key Supply Chain Issues Detected

The system identifies five primary types of supply chain issues:

1. **Inventory Imbalance**: Excessive or insufficient inventory levels
2. **Sales Performance Gap**: Significant underperformance against sales targets
3. **Pricing Issue**: Suboptimal pricing positioning relative to competition
4. **Supply Chain Disruption**: Backlog and fulfillment challenges
5. **Sell-Through Bottleneck**: Distribution partners not effectively selling to end customers

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supply-chain-anomaly-detection.git
   cd supply-chain-anomaly-detection
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

```python
from src.models.anomaly_detection import ScIssueDetection

# Initialize the detector
detector = ScIssueDetection()

# Load and preprocess data
detector.load_data('data/raw/supply_chain_data.csv')
detector.preprocess_data()

# Detect anomalies
detector.train_anomaly_detector()

# Classify issues
anomalies = detector.classify_issues()

# Generate recommendations
recommendations = detector.generate_recommendations(anomalies)

# Visualize results
detector.visualize_anomalies('WeeksOfStockT1', 'TargetAchievement')
detector.visualize_with_pca()
```

## 📈 MLflow Integration

This project is integrated with MLflow for experiment tracking and model management, particularly optimized for Databricks environments.

### Using MLflow

```python
from mlflow.utils import mlflow_utils

# Start an MLflow run
with mlflow_utils.start_run(experiment_name="supply_chain_anomaly_detection"):
    # Track parameters
    mlflow_utils.log_parameters({
        'anomaly_contamination': 0.05,
        'n_estimators': 100,
        'feature_count': len(detector.numerical_features)
    })
    
    # Run your model
    detector.train_anomaly_detector()
    anomalies = detector.classify_issues()
    
    # Log metrics
    mlflow_utils.log_metrics({
        'anomaly_count': anomalies['is_anomaly'].sum(),
        'anomaly_percentage': (anomalies['is_anomaly'].sum() / len(anomalies)) * 100
    })
    
    # Log models
    mlflow_utils.log_model(detector, "supply_chain_detector")
```

## 📁 Project Structure

```
supply-chain-anomaly-detection/
├── data/               # Data directory
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
├── mlflow/             # MLflow utilities
├── tests/              # Test suite
└── docs/               # Documentation
```

## 📝 Data Dictionary

| Feature | Description |
|---------|-------------|
| SellTo | Units sold to end customers |
| SellThru | Units sold to distributors |
| T2Inventory | Tier 2 inventory levels |
| DistributorInventory | Inventory at distributor level |
| Backlog | Unfulfilled orders |
| Shipments | Units shipped |
| AgedInventory | Inventory over a certain age threshold |
| WeeksOfStockT1 | Weeks of inventory at Tier 1 |
| WeeksOfStockT2 | Weeks of inventory at Tier 2 |
| NumCompetitors | Number of competitors in market |
| PricePositioning | Price relative to competition (100 = parity) |
| TargetQty | Target sales quantity |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the AI Inventory team License - see the LICENSE file for details.

## 📞 Contact

Project Owner - [JEDDI Mohammed](mailto:mohammed.jeddi@hp.com)

Project Link: [https://github.azc.ext.hp.com/mohammed-jeddi/supply-chain-anomaly-detection](https://github.com/yourusername/supply-chain-anomaly-detection)
