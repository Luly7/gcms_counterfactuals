#!/usr/bin/env python3
"""
Complete feature extraction pipeline combining all 37 features.

This module combines:
- 31 molecular descriptors (from RDKit)
- 1 column identifier (categorical encoding)
- 5 experimental features (temperature + flow rate)

Total: 37 features for XGBoost model

Author: Lourdes Castleton
Project: CS 6610 Capstone - GC-MS RT Prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .molecular_descriptors import MolecularDescriptorExtractor
from .experimental_features import ExperimentalFeatureExtractor

try:
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class CompleteFeatureExtractor:
    """
    Extract all 37 features for retention time prediction.
    
    Combines:
    - 31 molecular descriptors (RDKit)
    - 1 column identifier  
    - 5 experimental features (temp program + flow rate)
    """
    
    # Column name to ID mapping
    COLUMN_MAPPING = {
        'HP-5MS': 0, 'DB-5': 1, 'DB-1': 2, 'Rtx-1': 3,
        'SPB-1': 4, 'ZB-5': 5, 'BPX5': 6, 'Rtx-5': 7,
        'DB-17': 8, 'DB-35': 9, 'Rtx-35': 10,
        'DB-624': 11, 'Rtx-624': 12, 'DB-WAX': 13, 'DB-VRX': 14
    }
    
    def __init__(self):
        """Initialize extractors."""
        self.mol_extractor = MolecularDescriptorExtractor()
        self.exp_extractor = ExperimentalFeatureExtractor()
        
        logger.info("Initialized CompleteFeatureExtractor (37 features)")
    
    def encode_column(self, column_name: str) -> int:
        """
        Encode column name as integer ID.
        
        Args:
            column_name: GC column name (e.g., 'HP-5MS')
        
        Returns:
            Column ID (0-14)
        """
        if column_name not in self.COLUMN_MAPPING:
            logger.warning(f"Unknown column: {column_name}, defaulting to 0")
            return 0
        return self.COLUMN_MAPPING[column_name]
    
    def extract_single(self,
                      smiles: str,
                      column_name: str,
                      temp_program: str,
                      flow_rate: str) -> Optional[Dict[str, float]]:
        """
        Extract all 37 features for a single measurement.
        
        Args:
            smiles: Molecule SMILES string
            column_name: GC column name (e.g., 'HP-5MS')
            temp_program: Temperature program (e.g., '60-280C at 10C/min')
            flow_rate: Flow rate (e.g., '1.0 mL/min')
        
        Returns:
            Dictionary with all 37 features
        """
        try:
            # Extract molecular features (31)
            mol_features = self.mol_extractor.extract_features(smiles)
            if mol_features is None:
                return None
            
            # Encode column (1)
            column_id = self.encode_column(column_name)
            
            # Extract experimental features (5)
            exp_features = self.exp_extractor.extract_all_features(temp_program, flow_rate)
            
            # Combine all features
            all_features = {
                **mol_features,
                'column_id': column_id,
                **exp_features
            }
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def extract_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all 37 features for multiple measurements.
        
        Args:
            df: DataFrame with columns:
                - smiles: Molecule SMILES
                - column_type: Column name
                - temperature_program: Temp program string
                - flow_rate: Flow rate string
        
        Returns:
            DataFrame with all 37 features + original columns
        """
        logger.info(f"Extracting features for {len(df)} measurements...")
        
        # Validate required columns
        required = ['smiles', 'column_type', 'temperature_program', 'flow_rate']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract molecular features (31)
        logger.info("  Extracting molecular descriptors (31 features)...")
        mol_features_df = self.mol_extractor.extract_batch(df['smiles'].tolist())
        
        # Remove the 'smiles' column from mol_features_df (already in original df)
        mol_features_df = mol_features_df.drop(columns=['smiles'])
        
        # Encode columns (1)
        logger.info("  Encoding column types (1 feature)...")
        column_ids = df['column_type'].apply(self.encode_column)
        
        # Extract experimental features (5)
        logger.info("  Extracting experimental features (5 features)...")
        exp_features_df = self.exp_extractor.extract_batch(
            df['temperature_program'].tolist(),
            df['flow_rate'].tolist()
        )
        
        # Combine all features
        logger.info("  Combining all features...")
        result_df = pd.concat([
            df.reset_index(drop=True),
            mol_features_df.reset_index(drop=True),
            pd.DataFrame({'column_id': column_ids}).reset_index(drop=True),
            exp_features_df.reset_index(drop=True)
        ], axis=1)
        
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
        logger.info(f"Extraction complete: {len(result_df)} measurements, {result_df.shape[1]} columns")
        
        return result_df
    
    def get_feature_columns(self) -> List[str]:
        """
        Get ordered list of all 37 feature column names for model input.
        
        Returns:
            List of 37 feature names in correct order
        """
        return (
            self.mol_extractor.DESCRIPTOR_NAMES +
            ['column_id'] +
            self.exp_extractor.FEATURE_NAMES
        )


def main():
    """Example usage and testing."""
    print("=" * 70)
    print("Complete Feature Extraction Pipeline - Test")
    print("=" * 70)
    print()
    
    extractor = CompleteFeatureExtractor()
    
    # Test single extraction
    print("Test 1: Single measurement feature extraction")
    print("-" * 70)
    
    features = extractor.extract_single(
        smiles='c1ccccc1',  # Benzene
        column_name='HP-5MS',
        temp_program='60-280C at 10C/min',
        flow_rate='1.0 mL/min'
    )
    
    if features:
        print(f"Extracted {len(features)} features for benzene on HP-5MS")
        print("\nSample features:")
        print(f"  molecular_weight: {features['molecular_weight']:.2f}")
        print(f"  logp:             {features['logp']:.2f}")
        print(f"  column_id:        {features['column_id']}")
        print(f"  start_temp:       {features['start_temp']:.1f}")
        print(f"  heating_rate:     {features['heating_rate']:.1f}")
        print(f"  flow_rate:        {features['flow_rate']:.1f}")
    
    # Test batch extraction
    print("\n" + "=" * 70)
    print("Test 2: Batch feature extraction")
    print("-" * 70)
    
    # Create sample dataset
    test_data = pd.DataFrame({
        'smiles': ['c1ccccc1', 'CCO', 'Cc1ccccc1', 'CC(=O)C'],
        'column_type': ['HP-5MS', 'DB-5', 'HP-5MS', 'DB-WAX'],
        'temperature_program': [
            '60-280C at 10C/min',
            '40-240C at 8C/min',
            '60-300C at 15C/min',
            '50-280C at 10C/min'
        ],
        'flow_rate': ['1.0 mL/min', '1.2 mL/min', '1.5 mL/min', '1.0 mL/min'],
        'retention_time': [4.2, 2.1, 6.1, 2.5]  # Ground truth (not used in features)
    })
    
    print(f"Input data: {len(test_data)} measurements")
    print("\nInput DataFrame:")
    print(test_data)
    
    # Extract features
    result = extractor.extract_batch(test_data)
    
    print(f"\nOutput: {result.shape[0]} measurements, {result.shape[1]} total columns")
    
    # Get just the 37 feature columns
    feature_cols = extractor.get_feature_columns()
    print(f"\n37 feature columns for model input:")
    print(feature_cols)
    
    # Show feature matrix
    X = result[feature_cols]
    print(f"\nFeature matrix shape: {X.shape}")
    print("\nFirst row of features:")
    print(X.iloc[0])
    
    # Save test output
    output_file = "test_complete_features.csv"
    result.to_csv(output_file, index=False)
    print(f"\nSaved complete output to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FEATURE SUMMARY:")
    print("-" * 70)
    print(f"Molecular descriptors:    31 features (RDKit)")
    print(f"Column encoding:          1 feature (categorical)")
    print(f"Experimental conditions:  5 features (temp + flow)")
    print(f"{'─' * 70}")
    print(f"TOTAL:                    37 features")
    print("\nThese 37 features enabled R^2 = 0.9714 with XGBoost!")
    print("=" * 70)


if __name__ == "__main__":
    main()
