
#!/usr/bin/env python3
"""
Extract 31 molecular descriptors using RDKit (2023+ compatible).

Compatible with RDKit 2023.09+ API changes.

Author: Lourdes Castleton
Project: CS 6610 Capstone - GC-MS RT Prediction
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors, Fragments
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class MolecularDescriptorExtractor:
    """Extract comprehensive molecular features using RDKit 2023+."""
    
    DESCRIPTOR_NAMES = [
        'molecular_weight', 'logp', 'tpsa', 'num_rotatable_bonds',
        'num_h_acceptors', 'num_h_donors', 'num_aromatic_rings',
        'num_aliphatic_rings', 'num_saturated_rings', 'num_heteroatoms',
        'fraction_csp3', 'num_bridgehead_atoms', 'chi0v', 'chi1v',
        'kappa1', 'kappa2', 'balabanj', 'hallkieralpha', 'molmr',
        'labuteasa', 'peoe_vsa1', 'peoe_vsa2', 'peoe_vsa3', 'peoe_vsa4',
        'peoe_vsa5', 'smr_vsa1', 'smr_vsa2', 'slogp_vsa1', 'slogp_vsa2',
        'max_partial_charge', 'min_partial_charge'
    ]
    
    def __init__(self):
        """Initialize the descriptor extractor."""
        self.num_features = len(self.DESCRIPTOR_NAMES)
        logger.info(f"Initialized MolecularDescriptorExtractor with {self.num_features} features")
    
    def extract_features(self, smiles: str) -> Optional[Dict[str, float]]:
        """Extract all 31 molecular descriptors from a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None
            
            features = {}
            
            # ===== BASIC MOLECULAR PROPERTIES =====
            features['molecular_weight'] = Descriptors.ExactMolWt(mol)
            features['logp'] = Crippen.MolLogP(mol)
            features['tpsa'] = Descriptors.TPSA(mol)
            features['num_rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
            features['num_h_acceptors'] = Lipinski.NumHAcceptors(mol)
            features['num_h_donors'] = Lipinski.NumHDonors(mol)
            features['num_aromatic_rings'] = Lipinski.NumAromaticRings(mol)
            
            # ===== STRUCTURAL PROPERTIES =====
            features['num_aliphatic_rings'] = Lipinski.NumAliphaticRings(mol)
            features['num_saturated_rings'] = Lipinski.NumSaturatedRings(mol)
            features['num_heteroatoms'] = Lipinski.NumHeteroatoms(mol)
            features['fraction_csp3'] = Descriptors.FractionCSP3(mol)
            
            # Bridgehead atoms - calculate manually for RDKit 2023+
            features['num_bridgehead_atoms'] = self._count_bridgehead_atoms(mol)
            
            # ===== TOPOLOGICAL INDICES =====
            features['chi0v'] = GraphDescriptors.Chi0v(mol)
            features['chi1v'] = GraphDescriptors.Chi1v(mol)
            features['kappa1'] = GraphDescriptors.Kappa1(mol)
            features['kappa2'] = GraphDescriptors.Kappa2(mol)
            features['balabanj'] = GraphDescriptors.BalabanJ(mol)
            features['hallkieralpha'] = GraphDescriptors.HallKierAlpha(mol)
            
            # ===== ELECTRONIC PROPERTIES =====
            features['molmr'] = Crippen.MolMR(mol)
            
            # ===== SURFACE PROPERTIES =====
            features['labuteasa'] = Descriptors.LabuteASA(mol)
            features['peoe_vsa1'] = Descriptors.PEOE_VSA1(mol)
            features['peoe_vsa2'] = Descriptors.PEOE_VSA2(mol)
            features['peoe_vsa3'] = Descriptors.PEOE_VSA3(mol)
            features['peoe_vsa4'] = Descriptors.PEOE_VSA4(mol)
            features['peoe_vsa5'] = Descriptors.PEOE_VSA5(mol)
            features['smr_vsa1'] = Descriptors.SMR_VSA1(mol)
            features['smr_vsa2'] = Descriptors.SMR_VSA2(mol)
            features['slogp_vsa1'] = Descriptors.SlogP_VSA1(mol)
            features['slogp_vsa2'] = Descriptors.SlogP_VSA2(mol)
            
            # ===== PARTIAL CHARGES =====
            AllChem.ComputeGasteigerCharges(mol)
            charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') 
                      for i in range(mol.GetNumAtoms())]
            
            charges = [c if not np.isnan(c) else 0.0 for c in charges]
            
            if charges:
                features['max_partial_charge'] = max(charges)
                features['min_partial_charge'] = min(charges)
            else:
                features['max_partial_charge'] = 0.0
                features['min_partial_charge'] = 0.0
            
            # Replace any remaining NaN values
            for key in features:
                if np.isnan(features[key]) or np.isinf(features[key]):
                    features[key] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {smiles}: {e}")
            return None
    
    def _count_bridgehead_atoms(self, mol):
        """Count bridgehead atoms manually (RDKit 2023+ compatible)."""
        try:
            ri = mol.GetRingInfo()
            atom_rings = ri.AtomRings()
            
            if not atom_rings or len(atom_rings) < 2:
                return 0
            
            bridgehead_count = 0
            for atom_idx in range(mol.GetNumAtoms()):
                rings_with_atom = [ring for ring in atom_rings if atom_idx in ring]
                if len(rings_with_atom) >= 2:
                    bridgehead_count += 1
            
            return bridgehead_count
        except:
            return 0
    
    def extract_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """Extract features for multiple molecules."""
        results = []
        
        logger.info(f"Processing {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(smiles_list):
            if i > 0 and i % 100 == 0:
                logger.info(f"  Processed {i}/{len(smiles_list)} molecules")
            
            features = self.extract_features(smiles)
            if features:
                features['smiles'] = smiles
                results.append(features)
            else:
                logger.warning(f"  Skipped invalid molecule {i}: {smiles}")
        
        logger.info(f"Successfully processed {len(results)}/{len(smiles_list)} molecules")
        
        df = pd.DataFrame(results)
        
        cols = ['smiles'] + self.DESCRIPTOR_NAMES
        df = df[cols]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return self.DESCRIPTOR_NAMES.copy()


def main():
    """Example usage and testing."""
    print("=" * 60)
    print("Molecular Descriptor Extractor - Test")
    print("=" * 60)
    print()
    
    extractor = MolecularDescriptorExtractor()
    
    test_molecules = {
        'Benzene': 'c1ccccc1',
        'Ethanol': 'CCO',
        'Toluene': 'Cc1ccccc1',
        'Acetone': 'CC(=O)C',
        'Acetic acid': 'CC(=O)O'
    }
    
    print("Testing individual molecule extraction:")
    print("-" * 60)
    
    for name, smiles in test_molecules.items():
        features = extractor.extract_features(smiles)
        if features:
            print(f"\n{name} ({smiles}):")
            print(f"  MW:    {features['molecular_weight']:.2f} Da")
            print(f"  LogP:  {features['logp']:.2f}")
            print(f"  TPSA:  {features['tpsa']:.2f} Ų")
            print(f"  H-acc: {features['num_h_acceptors']:.0f}")
            print(f"  H-don: {features['num_h_donors']:.0f}")
            print(f"  Ar:    {features['num_aromatic_rings']:.0f}")
    
    print("\n" + "=" * 60)
    print("Testing batch extraction:")
    print("-" * 60)
    
    smiles_list = list(test_molecules.values())
    df = extractor.extract_batch(smiles_list)
    
    print(f"\nExtracted features for {len(df)} molecules")
    print(f"Features per molecule: {len(extractor.DESCRIPTOR_NAMES)}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    output_file = "test_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
