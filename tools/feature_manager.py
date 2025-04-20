# tools/feature_manager.py
"""
CLI tool for managing and inspecting features in the feature store.
"""
import argparse
import pandas as pd
import os
import sys
import logging
import numpy as np

# Add parent directory to path to access project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_store_factory import FeatureStoreFactory
from data.fetching import load_sample_data
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('f1_prediction.feature_manager')

def list_features(args):
    """List all available features."""
    logger = setup_logging()
    logger.info("Listing features")
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(args.config)
    
    # Get feature list
    features_df = feature_store.list_features(tags=args.tags)
    
    # Display features
    if args.output:
        features_df.to_csv(args.output, index=False)
        print(f"Features saved to {args.output}")
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(features_df)
    
    return features_df

def calculate_features(args):
    """Calculate and display features."""
    logger = setup_logging()
    logger.info(f"Calculating features: {args.features}")
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(args.config)
    
    # Load sample data or specified data file
    if args.input:
        if args.input.endswith('.csv'):
            data = pd.read_csv(args.input)
        else:
            logger.error(f"Unsupported file format: {args.input}")
            return
    else:
        data = load_sample_data(data_type='qualifying')
    
    logger.info(f"Loaded data with {len(data)} rows")
    
    # Calculate features
    features = feature_store.get_features(data, feature_names=args.features.split(','))
    
    # Display features
    if args.output:
        features.to_csv(args.output, index=False)
        print(f"Features saved to {args.output}")
    else:
        # Display subset if too many columns
        if len(features.columns) > 10 and not args.all_columns:
            print("First 10 columns of features:")
            print(features.iloc[:, :10])
            print(f"...and {len(features.columns) - 10} more columns")
        else:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 120)
            print(features)
    
    return features

def visualize_features(args):
    """Visualize features."""
    logger = setup_logging()
    logger.info(f"Visualizing features: {args.features}")
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(args.config)
    
    # Load sample data or specified data file
    if args.input:
        if args.input.endswith('.csv'):
            data = pd.read_csv(args.input)
        else:
            logger.error(f"Unsupported file format: {args.input}")
            return
    else:
        data = load_sample_data(data_type='qualifying')
    
    logger.info(f"Loaded data with {len(data)} rows")
    
    # Calculate features
    features = feature_store.get_features(data, feature_names=args.features.split(','))
    
    # Create output directory if needed
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Identify numeric columns for visualization
    numeric_cols = features.select_dtypes(include=['number']).columns.tolist()
    
    if not numeric_cols:
        logger.warning("No numeric columns found for visualization")
        return
    
    # Create visualizations
    if args.plot_type == 'correlation':
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation = features[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm', 
                   linewidths=0.5, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, 'correlation_heatmap.png'), dpi=300)
            logger.info(f"Saved correlation heatmap to {args.output_dir}")
        else:
            plt.show()
    
    elif args.plot_type == 'distribution':
        # Feature distributions
        for col in numeric_cols[:min(9, len(numeric_cols))]:  # Limit to 9 columns
            plt.figure(figsize=(10, 6))
            sns.histplot(features[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            
            if args.output_dir:
                plt.savefig(os.path.join(args.output_dir, f'distribution_{col}.png'), dpi=300)
            else:
                plt.show()
        
        if args.output_dir:
            logger.info(f"Saved distribution plots to {args.output_dir}")
    
    elif args.plot_type == 'pairplot':
        # Pairplot of features
        # Limit to a manageable number of features
        if len(numeric_cols) > 5:
            logger.warning(f"Limiting pairplot to first 5 numeric features out of {len(numeric_cols)}")
            plot_cols = numeric_cols[:5]
        else:
            plot_cols = numeric_cols
        
        plt.figure(figsize=(15, 15))
        sns.pairplot(features[plot_cols])
        plt.tight_layout()
        
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, 'feature_pairplot.png'), dpi=300)
            logger.info(f"Saved pairplot to {args.output_dir}")
        else:
            plt.show()
    
    else:
        logger.error(f"Unknown plot type: {args.plot_type}")

def clear_cache(args):
    """Clear feature cache."""
    logger = setup_logging()
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(args.config)
    
    # Clear specific features if specified
    if args.features:
        feature_names = args.features.split(',')
        logger.info(f"Clearing cache for features: {feature_names}")
        feature_store.clear_cache(feature_names)
    else:
        # Clear all features
        logger.info("Clearing entire feature cache")
        feature_store.clear_cache()
    
    print("Cache cleared successfully")

def show_dependencies(args):
    """Show feature dependencies."""
    logger = setup_logging()
    logger.info(f"Showing dependencies for feature: {args.feature}")
    
    # Create feature store
    feature_store = FeatureStoreFactory.create_feature_store(args.config)
    
    # Get dependencies
    try:
        dependencies = feature_store.get_feature_dependencies(args.feature)
        
        # Print dependencies tree
        def print_tree(node, level=0):
            indent = "  " * level
            prefix = "└─ " if level > 0 else ""
            
            if 'circular' in node and node['circular']:
                print(f"{indent}{prefix}{node['name']} [CIRCULAR REFERENCE]")
                return
            
            if 'missing' in node and node['missing']:
                print(f"{indent}{prefix}{node['name']} [MISSING]")
                return
            
            print(f"{indent}{prefix}{node['name']}")
            
            if 'dependencies' in node:
                for dep in node['dependencies']:
                    print_tree(dep, level + 1)
        
        print_tree(dependencies)
        
    except ValueError as e:
        logger.error(f"Error: {e}")

def main():
    """Main entry point for the feature manager CLI."""
    parser = argparse.ArgumentParser(description='F1 Prediction Feature Manager')
    parser.add_argument('--config', default='config/feature_store_config.yaml',
                      help='Path to feature store configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List features command
    list_parser = subparsers.add_parser('list', help='List available features')
    list_parser.add_argument('--tags', nargs='+', help='Filter features by tags')
    list_parser.add_argument('--output', help='Save output to CSV file')
    
    # Calculate features command
    calc_parser = subparsers.add_parser('calculate', help='Calculate features')
    calc_parser.add_argument('features', help='Comma-separated list of features to calculate')
    calc_parser.add_argument('--input', help='Input data file (CSV)')
    calc_parser.add_argument('--output', help='Save output to CSV file')
    calc_parser.add_argument('--all-columns', action='store_true', 
                           help='Display all columns (default is to show only first 10)')
    
    # Visualize features command
    vis_parser = subparsers.add_parser('visualize', help='Visualize features')
    vis_parser.add_argument('features', help='Comma-separated list of features to visualize')
    vis_parser.add_argument('--input', help='Input data file (CSV)')
    vis_parser.add_argument('--output-dir', help='Directory to save visualizations')
    vis_parser.add_argument('--plot-type', choices=['correlation', 'distribution', 'pairplot'],
                          default='correlation', help='Type of visualization')
    
    # Clear cache command
    clear_parser = subparsers.add_parser('clear-cache', help='Clear feature cache')
    clear_parser.add_argument('--features', help='Comma-separated list of features to clear (default: all)')
    
    # Show dependencies command
    deps_parser = subparsers.add_parser('dependencies', help='Show feature dependencies')
    deps_parser.add_argument('feature', help='Feature to show dependencies for')
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'list':
        list_features(args)
    elif args.command == 'calculate':
        calculate_features(args)
    elif args.command == 'visualize':
        visualize_features(args)
    elif args.command == 'clear-cache':
        clear_cache(args)
    elif args.command == 'dependencies':
        show_dependencies(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()