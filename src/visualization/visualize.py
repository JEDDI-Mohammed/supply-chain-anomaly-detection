"""
Visualization tools for supply chain anomaly detection.

This module provides visualization functions for anomalies, issue types, and PCA-based analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyVisualizer:
    """
    Visualization tools for supply chain anomaly detection.
    
    This class provides methods to create visualizations of anomalies and
    their classifications, helping to interpret and communicate results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # Configure default plot styles
        plt.style.use('seaborn-v0_8-darkgrid')
        self.default_palette = 'Set2'
        self.anomaly_palette = {'Normal': '#1f77b4', 'Anomaly': '#d62728'}
        
    def visualize_anomalies(self, data, column1, column2, figsize=(10, 6)):
        """
        Create a scatter plot visualization of anomalies based on two features.
        
        Args:
            data (pd.DataFrame): DataFrame containing data with anomaly flags
            column1 (str): Name of the column for the x-axis
            column2 (str): Name of the column for the y-axis
            figsize (tuple, optional): Figure size in inches
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Anomaly detection must be performed first. 'is_anomaly' column not found.")
            
        # Ensure columns exist in the data
        if column1 not in data.columns:
            raise ValueError(f"Column '{column1}' not found in the data")
        if column2 not in data.columns:
            raise ValueError(f"Column '{column2}' not found in the data")
        
        # Create a categorical column for better legend display
        anomaly_status = data['is_anomaly'].map({0: 'Normal', 1: 'Anomaly'})
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create the scatter plot with explicit hue order to ensure consistent colors
        scatter = sns.scatterplot(
            x=data[column1],
            y=data[column2],
            hue=anomaly_status,
            palette=self.anomaly_palette,
            alpha=0.7,
            s=80,  # Slightly larger points for better visibility
            ax=ax
        )
        
        # Set the title and labels
        ax.set_title(f'Anomaly Detection: {column1} vs {column2}', fontsize=14)
        ax.set_xlabel(column1, fontsize=12)
        ax.set_ylabel(column2, fontsize=12)
        
        # Fix the legend to explicitly show Normal and Anomaly with correct colors
        handles, labels = scatter.get_legend_handles_labels()
        ax.legend(handles, labels, title='Status')
        
        # Add count of anomalies as text
        anomaly_count = data['is_anomaly'].sum()
        normal_count = len(data) - anomaly_count
        anomaly_pct = (anomaly_count / len(data)) * 100
        
        info_text = (
            f"Total points: {len(data)}\n"
            f"Normal: {normal_count} ({100-anomaly_pct:.1f}%)\n"
            f"Anomalies: {anomaly_count} ({anomaly_pct:.1f}%)"
        )
        
        ax.annotate(
            info_text, 
            xy=(0.02, 0.97), 
            xycoords='axes fraction',
            va='top', 
            ha='left', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # If issue_type is available, create a second visualization for anomalies by issue type
        if 'issue_type' in data.columns and anomaly_count > 0:
            self._visualize_issue_types(data, column1, column2, figsize)
        
        return fig
    
    def _visualize_issue_types(self, data, column1, column2, figsize):
        """
        Create a visualization of anomalies colored by issue type.
        
        Args:
            data (pd.DataFrame): DataFrame containing data with anomaly flags and issue types
            column1 (str): Name of the column for the x-axis
            column2 (str): Name of the column for the y-axis
            figsize (tuple): Figure size in inches
        """
        # Filter only the anomalies
        anomalies = data[data['is_anomaly'] == 1].copy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a scatter plot with issue types
        scatter = sns.scatterplot(
            x=anomalies[column1],
            y=anomalies[column2],
            hue=anomalies['issue_type'],
            palette='viridis',
            s=100,
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title(f'Issue Types: {column1} vs {column2}', fontsize=14)
        ax.set_xlabel(column1, fontsize=12)
        ax.set_ylabel(column2, fontsize=12)
        ax.legend(title='Issue Type')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_with_pca(self, anomalies, numerical_features, figsize=(18, 8)):
        """
        Visualize anomalies and their issue types using Principal Component Analysis (PCA).
        
        This helps to validate whether the issue categorization is well-detected by
        showing how well separated the different issue types are in a reduced dimensional space.
        
        Args:
            anomalies (pd.DataFrame): DataFrame containing only anomalies with issue types
            numerical_features (list): List of numerical feature names to use for PCA
            figsize (tuple, optional): Figure size in inches
            
        Returns:
            tuple: (PCA model, DataFrame with PCA components added)
        """
        if 'issue_type' not in anomalies.columns:
            raise ValueError("Issue classification must be performed first.")
        
        if len(anomalies) < 3:
            logger.warning("Not enough anomalies for PCA visualization (minimum 3 required).")
            return None, anomalies
        
        # Select the numerical features for PCA
        X = anomalies[numerical_features].copy()
        
        # Handle missing values if any
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA results to the anomalies dataframe
        pca_anomalies = anomalies.copy()
        pca_anomalies['PCA1'] = X_pca[:, 0]
        pca_anomalies['PCA2'] = X_pca[:, 1]
        
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: PCA with issue types
        # Create a scatter plot with different colors for each issue type
        issue_types = pca_anomalies['issue_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(issue_types)))
        
        for i, issue_type in enumerate(issue_types):
            mask = pca_anomalies['issue_type'] == issue_type
            ax1.scatter(
                pca_anomalies.loc[mask, 'PCA1'], 
                pca_anomalies.loc[mask, 'PCA2'],
                c=[colors[i]],
                label=issue_type,
                alpha=0.7,
                s=80,
                edgecolors='k'
            )
        
        # Add title and labels
        ax1.set_title('PCA Visualization of Anomalies by Issue Type', fontsize=14)
        ax1.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax1.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax1.legend(title='Issue Type')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add feature correlation with PCA components as arrows
        feature_names = numerical_features
        pca_components = pca.components_
        
        # Calculate scaling factor for arrows
        arrow_scale = 5
        
        # Only show top contributing features to avoid clutter
        n_top_features = min(10, len(feature_names))
        feature_importance = np.sum(np.abs(pca_components), axis=0)
        top_indices = np.argsort(feature_importance)[-n_top_features:]
        
        for i in top_indices:
            ax1.arrow(
                0, 0,
                pca_components[0, i] * arrow_scale,
                pca_components[1, i] * arrow_scale,
                head_width=0.1,
                head_length=0.1,
                fc='gray',
                ec='gray',
                alpha=0.5
            )
            ax1.text(
                pca_components[0, i] * arrow_scale * 1.15,
                pca_components[1, i] * arrow_scale * 1.15,
                feature_names[i],
                fontsize=8,
                ha='center',
                va='center',
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8)
            )
        
        # Plot 2: PCA with priority
        # Create a scatter plot with different colors for priority
        if 'Priority' in pca_anomalies.columns:
            priorities = pca_anomalies['Priority']
        else:
            # If Priority column doesn't exist, use anomaly_score to create one
            pca_anomalies['Priority'] = pca_anomalies['anomaly_score'].apply(
                lambda x: 'High' if x > 0.7 else 'Medium'
            )
            priorities = pca_anomalies['Priority']
        
        unique_priorities = np.unique(priorities)
        priority_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green', 'Unknown': 'gray'}
        
        for priority in unique_priorities:
            if priority in priority_colors:
                color = priority_colors[priority]
            else:
                color = 'blue'  # Default color
                
            mask = priorities == priority
            ax2.scatter(
                pca_anomalies.loc[mask, 'PCA1'], 
                pca_anomalies.loc[mask, 'PCA2'],
                c=color,
                label=priority,
                alpha=0.7,
                s=80,
                edgecolors='k'
            )
        
        # Add title and labels
        ax2.set_title('PCA Visualization of Anomalies by Priority', fontsize=14)
        ax2.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax2.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        ax2.legend(title='Priority')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate and print total explained variance
        total_variance = pca.explained_variance_ratio_.sum() * 100
        plt.figtext(0.5, 0.01, f'Total Explained Variance: {total_variance:.2f}%', 
                    ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.07)
        plt.show()
        
        # Optionally, we can also create a 3D PCA plot if we have enough anomalies
        if len(anomalies) >= 4:
            self._create_3d_pca_plot(pca_anomalies, X_scaled)
        
        logger.info(f"PCA visualization created with {len(pca_anomalies)} anomalies and {len(issue_types)} issue types")
        
        return pca, pca_anomalies
    
    def _create_3d_pca_plot(self, anomalies, X_scaled):
        """
        Create a 3D PCA plot to further visualize the separation of issue types.
        
        Args:
            anomalies (pd.DataFrame): DataFrame containing anomalies with issue types
            X_scaled (np.ndarray): Scaled feature matrix for PCA
        """
        # Apply PCA with 3 components
        pca3d = PCA(n_components=3)
        X_pca3d = pca3d.fit_transform(X_scaled)
        
        # Add PCA results to the dataframe
        anomalies.loc[:, 'PCA1_3D'] = X_pca3d[:, 0]
        anomalies.loc[:, 'PCA2_3D'] = X_pca3d[:, 1]
        anomalies.loc[:, 'PCA3_3D'] = X_pca3d[:, 2]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each issue type
        issue_types = anomalies['issue_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(issue_types)))
        
        for i, issue_type in enumerate(issue_types):
            mask = anomalies['issue_type'] == issue_type
            ax.scatter(
                anomalies.loc[mask, 'PCA1_3D'], 
                anomalies.loc[mask, 'PCA2_3D'],
                anomalies.loc[mask, 'PCA3_3D'],
                c=[colors[i]],
                label=issue_type,
                alpha=0.7,
                s=60,
                edgecolors='k'
            )
        
        # Add title and labels
        ax.set_title('3D PCA Visualization of Anomalies by Issue Type', fontsize=14)
        ax.set_xlabel(f'PC1 ({pca3d.explained_variance_ratio_[0]:.2%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca3d.explained_variance_ratio_[1]:.2%})', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca3d.explained_variance_ratio_[2]:.2%})', fontsize=10)
        ax.legend(title='Issue Type')
        
        # Calculate and add total explained variance
        total_variance = pca3d.explained_variance_ratio_.sum() * 100
        plt.figtext(0.5, 0.01, f'Total Explained Variance (3D): {total_variance:.2f}%', 
                    ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.07)
        plt.show()
    
    def analyze_pca_results(self, pca, pca_anomalies):
        """
        Analyze PCA results to validate the quality of issue categorization.
        
        This provides quantitative metrics on how well separated the issue types are.
        
        Args:
            pca: Fitted PCA model
            pca_anomalies (pd.DataFrame): DataFrame of anomalies with PCA components added
            
        Returns:
            dict: Dictionary of analysis metrics
        """
        # Ensure we have the necessary data
        if 'PCA1' not in pca_anomalies.columns or 'PCA2' not in pca_anomalies.columns:
            raise ValueError("PCA components not found in DataFrame")
        
        if 'issue_type' not in pca_anomalies.columns:
            raise ValueError("Issue type column not found in DataFrame")
        
        # Initialize results dictionary
        results = {
            'total_explained_variance': pca.explained_variance_ratio_.sum() * 100,
            'issue_types': pca_anomalies['issue_type'].unique().tolist(),
            'counts_by_issue': pca_anomalies['issue_type'].value_counts().to_dict(),
            'silhouette_score': None,
            'separation_quality': None,
            'feature_importance': None,
            'issue_centroid_distances': None
        }
        
        # Only calculate silhouette score if we have multiple issue types and each type has multiple samples
        issue_types = pca_anomalies['issue_type'].unique()
        valid_for_silhouette = len(issue_types) > 1
        
        for issue_type in issue_types:
            count = sum(pca_anomalies['issue_type'] == issue_type)
            if count < 2:
                valid_for_silhouette = False
                break
        
        if valid_for_silhouette:
            try:
                # Create labels from issue types
                issue_labels = pca_anomalies['issue_type'].astype('category').cat.codes
                
                # Calculate silhouette score using PCA components
                X_pca = pca_anomalies[['PCA1', 'PCA2']].values
                silhouette_avg = silhouette_score(X_pca, issue_labels)
                results['silhouette_score'] = silhouette_avg
                
                # Interpret silhouette score
                if silhouette_avg > 0.5:
                    results['separation_quality'] = "Good separation between issue types"
                elif silhouette_avg > 0.25:
                    results['separation_quality'] = "Moderate separation between issue types"
                else:
                    results['separation_quality'] = "Poor separation between issue types"
            except Exception as e:
                logger.error(f"Error calculating silhouette score: {str(e)}")
                results['separation_quality'] = "Could not calculate separation quality"
        else:
            results['separation_quality'] = "Not enough samples per issue type for silhouette score"
        
        # Calculate feature importance based on PCA components
        numerical_features = list(pca.feature_names_in_) if hasattr(pca, 'feature_names_in_') else [
            f'Feature_{i}' for i in range(pca.components_.shape[1])
        ]
        
        pca_components = pca.components_
        
        feature_importance = pd.DataFrame({
            'Feature': numerical_features,
            'PC1_Importance': np.abs(pca_components[0]),
            'PC2_Importance': np.abs(pca_components[1]),
            'Total_Importance': np.abs(pca_components[0]) + np.abs(pca_components[1])
        })
        
        feature_importance = feature_importance.sort_values('Total_Importance', ascending=False)
        results['feature_importance'] = feature_importance.head(10).to_dict('records')
        
        # Calculate centroid distances between issue types
        if len(issue_types) > 1:
            centroid_distances = {}
            
            for i, type1 in enumerate(issue_types):
                type1_data = pca_anomalies[pca_anomalies['issue_type'] == type1]
                type1_centroid = type1_data[['PCA1', 'PCA2']].mean().values
                
                for j, type2 in enumerate(issue_types[i+1:], i+1):
                    type2_data = pca_anomalies[pca_anomalies['issue_type'] == type2]
                    type2_centroid = type2_data[['PCA1', 'PCA2']].mean().values
                    
                    distance = np.linalg.norm(type1_centroid - type2_centroid)
                    centroid_distances[f"{type1} vs {type2}"] = float(distance)
            
            results['issue_centroid_distances'] = centroid_distances
        
        # Print summarized analysis
        logger.info("\nPCA Analysis for Issue Categorization Validation:")
        logger.info(f"Total explained variance: {results['total_explained_variance']:.2f}%")
        logger.info(f"Issue types identified: {len(results['issue_types'])}")
        
        if results['silhouette_score'] is not None:
            logger.info(f"Silhouette score: {results['silhouette_score']:.3f}")
        
        logger.info(f"Separation quality: {results['separation_quality']}")
        
        logger.info("\nTop features driving categorization:")
        top_features = sorted(results['feature_importance'], key=lambda x: x['Total_Importance'], reverse=True)[:5]
        for feature in top_features:
            logger.info(f"  - {feature['Feature']}: {feature['Total_Importance']:.3f}")
        
        if results['issue_centroid_distances']:
            logger.info("\nIssue type separation distances:")
            # Sort by distance for clearer presentation
            sorted_distances = sorted(results['issue_centroid_distances'].items(), key=lambda x: x[1], reverse=True)
            
            for pair, distance in sorted_distances:
                quality = "Well separated" if distance > 1.5 else "Potentially overlapping" if distance < 0.8 else "Moderately separated"
                logger.info(f"  - {pair}: {distance:.2f} ({quality})")
        
        return results