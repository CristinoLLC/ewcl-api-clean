#!/usr/bin/env python3
"""
EWCL Feature Extractor - Production Version
==========================================

Extracts all features needed for EWCLv1 (249 features) and EWCLv1-E (505 features).
Handles sequence features, PSSM data, and ProtT5 embeddings with proper fallbacks.
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Amino acid properties (Kyte-Doolittle scale and others)
AA_PROPERTIES = {
    'hydropathy': {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2, 'X': 0.0
    },
    'polarity': {
        'A': 0.0, 'R': 1.0, 'N': 1.0, 'D': 1.0, 'C': 0.0, 'Q': 1.0, 'E': 1.0,
        'G': 0.0, 'H': 1.0, 'I': 0.0, 'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0,
        'P': 0.0, 'S': 1.0, 'T': 1.0, 'W': 0.0, 'Y': 1.0, 'V': 0.0, 'X': 0.0
    },
    'charge_pH7': {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
        'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
        'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0, 'X': 0
    },
    'vdw_volume': {
        'A': 67, 'R': 148, 'N': 96, 'D': 91, 'C': 86, 'Q': 114, 'E': 109,
        'G': 48, 'H': 118, 'I': 124, 'L': 124, 'K': 135, 'M': 124, 'F': 135,
        'P': 90, 'S': 73, 'T': 93, 'W': 163, 'Y': 141, 'V': 105, 'X': 100
    },
    'flexibility': {
        'A': 0.25, 'R': 0.55, 'N': 0.56, 'D': 0.68, 'C': 0.19, 'Q': 0.56, 'E': 0.65,
        'G': 0.8, 'H': 0.5, 'I': 0.18, 'L': 0.19, 'K': 0.54, 'M': 0.26, 'F': 0.2,
        'P': 0.0, 'S': 0.53, 'T': 0.45, 'W': 0.2, 'Y': 0.24, 'V': 0.18, 'X': 0.4
    },
    'bulkiness': {
        'A': 0.61, 'R': 0.68, 'N': 0.63, 'D': 0.64, 'C': 0.68, 'Q': 0.68, 'E': 0.68,
        'G': 0.0, 'H': 0.68, 'I': 0.68, 'L': 0.68, 'K': 0.68, 'M': 0.68, 'F': 0.68,
        'P': 0.39, 'S': 0.53, 'T': 0.59, 'W': 0.68, 'Y': 0.68, 'V': 0.64, 'X': 0.5
    },
    'helix_prop': {
        'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.7, 'Q': 1.11, 'E': 1.51,
        'G': 0.57, 'H': 1.0, 'I': 1.08, 'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13,
        'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06, 'X': 1.0
    },
    'sheet_prop': {
        'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19, 'Q': 1.1, 'E': 0.37,
        'G': 0.75, 'H': 0.87, 'I': 1.6, 'L': 1.3, 'K': 0.74, 'M': 1.05, 'F': 1.38,
        'P': 0.55, 'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.7, 'X': 1.0
    }
}

class EWCLFeatureExtractor:
    """Production EWCL feature extractor supporting both EWCLv1 and EWCLv1-E models."""
    
    def __init__(self, embeddings_dir: Optional[Path] = None, pssm_data: Optional[pd.DataFrame] = None):
        """
        Initialize the feature extractor.
        
        Args:
            embeddings_dir: Path to ProtT5 embeddings directory (for EWCLv1-E)
            pssm_data: Preloaded PSSM data DataFrame (optional)
        """
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.pssm_data = pssm_data
        
    def extract_sequence_features(self, sequence: str, protein_id: str) -> pd.DataFrame:
        """
        Extract all features needed for EWCL models from a protein sequence.
        
        Args:
            sequence: Protein sequence string
            protein_id: UniProt accession (used for PSSM/embeddings lookup)
            
        Returns:
            DataFrame with 1-based residue_index and all required features
        """
        # Normalize sequence
        sequence = sequence.upper().replace('U', 'C').replace('*', 'X')
        L = len(sequence)
        
        if L == 0:
            return pd.DataFrame()
        
        # Initialize DataFrame with 1-based indexing
        df = pd.DataFrame({
            'residue_index': np.arange(1, L + 1, dtype='int64'),
            'aa': list(sequence)
        })
        
        # 1. Basic amino acid properties
        df = self._add_basic_properties(df, sequence)
        
        # 2. Window-based features (5, 11, 25, 50, 100 residue windows)
        df = self._add_window_features(df, sequence)
        
        # 3. Composition features
        df = self._add_composition_features(df, sequence)
        
        # 4. Sequence complexity features
        df = self._add_complexity_features(df, sequence)
        
        # 5. Poly-amino acid runs
        df = self._add_poly_runs(df, sequence)
        
        # 6. One-hot amino acid encoding
        df = self._add_aa_encoding(df, sequence)
        
        # 7. PSSM features (with fallback)
        df = self._add_pssm_features(df, protein_id, sequence)
        
        # 8. ProtT5 embeddings (for EWCLv1-E, with fallback)
        df = self._add_embedding_features(df, protein_id, sequence)
        
        return df
    
    def _add_basic_properties(self, df: pd.DataFrame, sequence: str) -> pd.DataFrame:
        """Add basic amino acid properties."""
        for prop_name, prop_dict in AA_PROPERTIES.items():
            df[prop_name] = [prop_dict.get(aa, prop_dict['X']) for aa in sequence]
        
        # Unknown amino acid indicator
        df['is_unknown_aa'] = (df['aa'] == 'X').astype(float)
        
        # SCD (Sequence Charge Decoration) - simplified version
        charges = np.array([AA_PROPERTIES['charge_pH7'].get(aa, 0) for aa in sequence])
        scd_values = []
        for i in range(len(sequence)):
            # Simple SCD approximation: local charge density
            window = charges[max(0, i-5):min(len(charges), i+6)]
            scd_values.append(np.abs(window).mean())
        df['scd_local'] = scd_values
        
        return df
    
    def _add_window_features(self, df: pd.DataFrame, sequence: str) -> pd.DataFrame:
        """Add sliding window features for multiple window sizes."""
        windows = [5, 11, 25, 50, 100]
        
        for window_size in windows:
            suffix = f"_w{window_size}"
            pad = window_size // 2
            
            for prop_name in ['hydropathy', 'polarity', 'vdw_volume', 'flexibility', 
                             'bulkiness', 'helix_prop', 'sheet_prop', 'charge_pH7']:
                values = np.array([AA_PROPERTIES[prop_name].get(aa, AA_PROPERTIES[prop_name]['X']) 
                                 for aa in sequence])
                
                # Pad sequence for window calculations
                padded = np.pad(values, pad, mode='edge')
                
                # Calculate window statistics
                means, stds, mins, maxs = [], [], [], []
                for i in range(len(sequence)):
                    window_vals = padded[i:i+window_size]
                    means.append(np.mean(window_vals))
                    stds.append(np.std(window_vals))
                    mins.append(np.min(window_vals))
                    maxs.append(np.max(window_vals))
                
                # Fix feature naming to match model expectations
                if prop_name == 'hydropathy':
                    prop_short = 'hydro'
                elif prop_name == 'polarity':
                    prop_short = 'polar'
                elif prop_name == 'vdw_volume':
                    prop_short = 'vdw'
                elif prop_name == 'flexibility':
                    prop_short = 'flex'
                elif prop_name == 'bulkiness':
                    prop_short = 'bulk'
                elif prop_name == 'helix_prop':
                    prop_short = 'helix_prop'
                elif prop_name == 'sheet_prop':
                    prop_short = 'sheet_prop'
                elif prop_name == 'charge_pH7':
                    prop_short = 'charge'
                else:
                    prop_short = prop_name
                
                df[f"{prop_short}{suffix}_mean"] = means
                df[f"{prop_short}{suffix}_std"] = stds
                df[f"{prop_short}{suffix}_min"] = mins
                df[f"{prop_short}{suffix}_max"] = maxs
            
            # Window-specific features
            df[f"entropy{suffix}"] = self._calculate_entropy(sequence, window_size)
            df[f"low_complex{suffix}"] = self._calculate_low_complexity(sequence, window_size)
            df[f"comp_bias{suffix}"] = self._calculate_composition_bias(sequence, window_size)
            df[f"uversky_dist{suffix}"] = self._calculate_uversky_distance(sequence, window_size)
        
        return df
    
    def _calculate_entropy(self, sequence: str, window_size: int) -> List[float]:
        """Calculate Shannon entropy in sliding windows."""
        pad = window_size // 2
        padded_seq = 'X' * pad + sequence + 'X' * pad
        
        entropies = []
        for i in range(len(sequence)):
            window = padded_seq[i:i+window_size]
            # Calculate Shannon entropy
            aa_counts = {}
            for aa in window:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            entropy = 0.0
            for count in aa_counts.values():
                p = count / len(window)
                if p > 0:
                    entropy -= p * np.log2(p)
            entropies.append(entropy)
        
        return entropies
    
    def _calculate_low_complexity(self, sequence: str, window_size: int) -> List[float]:
        """Calculate low complexity score in sliding windows."""
        pad = window_size // 2
        padded_seq = 'X' * pad + sequence + 'X' * pad
        
        complexity_scores = []
        for i in range(len(sequence)):
            window = padded_seq[i:i+window_size]
            # Simple complexity: number of unique amino acids / window size
            unique_aas = len(set(window))
            complexity = unique_aas / window_size
            complexity_scores.append(1.0 - complexity)  # Low complexity = high score
        
        return complexity_scores
    
    def _calculate_composition_bias(self, sequence: str, window_size: int) -> List[float]:
        """Calculate composition bias in sliding windows."""
        pad = window_size // 2
        padded_seq = 'X' * pad + sequence + 'X' * pad
        
        bias_scores = []
        for i in range(len(sequence)):
            window = padded_seq[i:i+window_size]
            # Composition bias: deviation from expected uniform distribution
            aa_counts = {}
            for aa in window:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            # Calculate chi-square-like statistic
            expected = window_size / 20  # Assuming 20 standard amino acids
            bias = 0.0
            for count in aa_counts.values():
                bias += (count - expected) ** 2 / expected
            bias_scores.append(bias / window_size)
        
        return bias_scores
    
    def _calculate_uversky_distance(self, sequence: str, window_size: int) -> List[float]:
        """Calculate Uversky plot distance in sliding windows."""
        pad = window_size // 2
        padded_seq = 'X' * pad + sequence + 'X' * pad
        
        distances = []
        for i in range(len(sequence)):
            window = padded_seq[i:i+window_size]
            
            # Calculate mean hydropathy and charge for window
            hydro_sum = sum(AA_PROPERTIES['hydropathy'].get(aa, 0) for aa in window)
            charge_sum = sum(abs(AA_PROPERTIES['charge_pH7'].get(aa, 0)) for aa in window)
            
            mean_hydro = hydro_sum / len(window)
            mean_charge = charge_sum / len(window)
            
            # Distance from Uversky boundary (simplified)
            # Boundary approximately: charge = 2.785 * hydropathy - 1.151
            boundary_charge = 2.785 * mean_hydro - 1.151
            distance = mean_charge - boundary_charge
            distances.append(distance)
        
        return distances
    
    def _add_composition_features(self, df: pd.DataFrame, sequence: str) -> pd.DataFrame:
        """Add global sequence composition features."""
        L = len(sequence)
        
        # Individual amino acid compositions
        for aa in 'ARNDCQEGHILKMFPSTWYV':
            df[f'comp_{aa}'] = sequence.count(aa) / L
        
        # Grouped compositions
        aromatic = 'FWY'
        positive = 'RK'
        negative = 'DE'
        polar = 'NQST'
        aliphatic = 'AILV'
        
        df['comp_frac_aromatic'] = sum(sequence.count(aa) for aa in aromatic) / L
        df['comp_frac_positive'] = sum(sequence.count(aa) for aa in positive) / L
        df['comp_frac_negative'] = sum(sequence.count(aa) for aa in negative) / L
        df['comp_frac_polar'] = sum(sequence.count(aa) for aa in polar) / L
        df['comp_frac_aliphatic'] = sum(sequence.count(aa) for aa in aliphatic) / L
        df['comp_frac_proline'] = sequence.count('P') / L
        df['comp_frac_glycine'] = sequence.count('G') / L
        
        return df
    
    def _add_complexity_features(self, df: pd.DataFrame, sequence: str) -> pd.DataFrame:
        """Add sequence complexity and disorder-related features."""
        # aa_encoded feature - indicates if amino acid is standard (not X)
        df['aa_encoded'] = (df['aa'] != 'X').astype(float)
        return df
    
    def _add_poly_runs(self, df: pd.DataFrame, sequence: str) -> pd.DataFrame:
        """Add poly-amino acid run features."""
        poly_aas = ['P', 'E', 'K', 'Q', 'S', 'G', 'D', 'N']
        
        for aa in poly_aas:
            # Find runs of 3 or more consecutive amino acids
            in_run = [False] * len(sequence)
            i = 0
            while i < len(sequence):
                if sequence[i] == aa:
                    run_start = i
                    while i < len(sequence) and sequence[i] == aa:
                        i += 1
                    run_length = i - run_start
                    if run_length >= 3:
                        for j in range(run_start, i):
                            in_run[j] = True
                else:
                    i += 1
            
            df[f'in_poly_{aa}_run_ge3'] = [float(x) for x in in_run]
        
        return df
    
    def _add_aa_encoding(self, df: pd.DataFrame, sequence: str) -> pd.DataFrame:
        """Skip one-hot encoding here - it's handled in PSSM features to avoid duplication."""
        # This method is intentionally empty - amino acid encoding is handled in _add_pssm_features
        # to avoid feature duplication between one-hot encoding and PSSM features
        return df
    
    def _add_pssm_features(self, df: pd.DataFrame, protein_id: str, sequence: str) -> pd.DataFrame:
        """Add PSSM features with fallback to one-hot encoding (universal model compatibility)."""
        standard_aas = 'ARNDCQEGHILKMFPSTWYV'
        
        # Try to load PSSM data for this protein
        has_pssm = False
        if self.pssm_data is not None and len(self.pssm_data) > 0:
            try:
                # Look for PSSM data in the provided DataFrame
                protein_pssm = self.pssm_data[self.pssm_data['protein_id'] == protein_id]
                if not protein_pssm.empty:
                    has_pssm = True
                    # Add PSSM scores for each amino acid
                    for aa in standard_aas:
                        if aa in protein_pssm.columns:
                            # Align PSSM data with sequence positions
                            pssm_scores = protein_pssm[aa].values
                            if len(pssm_scores) == len(sequence):
                                df[aa] = pssm_scores
                            else:
                                # Length mismatch - use sequence-based fallback
                                df[aa] = (df['aa'] == aa).astype(float)
                        else:
                            df[aa] = (df['aa'] == aa).astype(float)
                    
                    # Add PSSM-derived features if available
                    if 'pssm_entropy' in protein_pssm.columns:
                        df['pssm_entropy'] = protein_pssm['pssm_entropy'].values[:len(sequence)]
                    if 'pssm_max_score' in protein_pssm.columns:
                        df['pssm_max_score'] = protein_pssm['pssm_max_score'].values[:len(sequence)]
                    if 'pssm_variance' in protein_pssm.columns:
                        df['pssm_variance'] = protein_pssm['pssm_variance'].values[:len(sequence)]
            except Exception as e:
                # If any error accessing PSSM data, fall back to one-hot encoding
                has_pssm = False
        
        # Fallback: use one-hot encoding if no PSSM data (UNIVERSAL MODEL COMPATIBILITY)
        if not has_pssm:
            # Add one-hot encoded amino acid features (this is what the universal models expect)
            for aa in standard_aas:
                if aa not in df.columns:
                    df[aa] = (df['aa'] == aa).astype(float)
            
            # Default PSSM-derived features
            if 'pssm_entropy' not in df.columns:
                df['pssm_entropy'] = 2.0  # Average entropy
            if 'pssm_max_score' not in df.columns:
                df['pssm_max_score'] = 0.0  # Neutral score
            if 'pssm_variance' not in df.columns:
                df['pssm_variance'] = 1.0  # Default variance
        
        df['has_pssm_data'] = float(has_pssm)
        return df
    
    def _add_embedding_features(self, df: pd.DataFrame, protein_id: str, sequence: str) -> pd.DataFrame:
        """Add ProtT5 embedding features for EWCLv1-E model with robust fallback handling."""
        has_embeddings = False
        
        # Define the specific embedding dimensions used by EWCLv1-E
        embedding_dims = [0, 1, 8, 9, 10, 13, 14, 20, 22, 28, 31, 32, 41, 47, 48, 53, 54, 59, 60, 63, 66, 72, 75, 79, 81, 84, 92, 96, 99, 100, 103, 106, 109, 112, 113, 117, 133, 136, 141, 144, 146, 153, 163, 165, 169, 170, 172, 178, 180, 182, 188, 189, 194, 204, 205, 214, 226, 229, 232, 233, 240, 243, 245, 246, 250, 256, 261, 262, 269, 273, 274, 276, 282, 284, 286, 288, 289, 294, 300, 306, 312, 318, 320, 323, 330, 346, 351, 354, 355, 357, 358, 366, 372, 376, 380, 391, 397, 402, 403, 405, 407, 409, 412, 419, 424, 426, 428, 434, 439, 442, 443, 448, 461, 468, 470, 483, 485, 489, 491, 505, 511, 513, 516, 518, 521, 524, 532, 541, 542, 546, 552, 554, 555, 569, 578, 579, 589, 600, 608, 612, 613, 614, 615, 617, 625, 627, 629, 635, 644, 646, 652, 655, 660, 663, 670, 675, 678, 683, 686, 689, 691, 693, 697, 698, 700, 706, 708, 710, 725, 727, 728, 736, 739, 741, 743, 747, 749, 751, 757, 758, 759, 761, 766, 773, 785, 789, 792, 793, 796, 802, 804, 805, 807, 808, 811, 813, 815, 817, 822, 828, 834, 835, 837, 845, 852, 858, 862, 869, 870, 872, 874, 875, 878, 879, 881, 885, 886, 888, 891, 892, 893, 897, 898, 904, 911, 912, 919, 924, 930, 934, 938, 944, 947, 948, 949, 955, 958, 967, 970, 985, 986, 991, 995, 1008, 1009, 1011, 1016, 1017, 1018, 1022]
        
        # Try to load embedding data with multiple fallback strategies
        if self.embeddings_dir and self.embeddings_dir.exists():
            # Try exact protein ID match first
            embedding_files = [
                f"{protein_id}.npy",
                f"{protein_id}_embeddings.npy",
                f"{protein_id.split('|')[0]}.npy",  # Handle pipe-separated IDs
                f"{protein_id.split('_')[0]}.npy",  # Handle underscore-separated IDs
            ]
            
            for filename in embedding_files:
                embedding_file = self.embeddings_dir / filename
                if embedding_file.exists():
                    try:
                        embeddings = np.load(embedding_file)
                        
                        # Validate embedding dimensions
                        if len(embeddings.shape) != 2:
                            print(f"Warning: Unexpected embedding shape for {protein_id}: {embeddings.shape}")
                            continue
                            
                        # Handle sequence length mismatch with intelligent truncation/padding
                        if embeddings.shape[0] != len(sequence):
                            print(f"Warning: Embedding length mismatch for {protein_id}: {embeddings.shape[0]} vs {len(sequence)}")
                            
                            if embeddings.shape[0] > len(sequence):
                                # Truncate embeddings to match sequence length
                                embeddings = embeddings[:len(sequence)]
                            else:
                                # Pad embeddings with mean values
                                mean_embedding = np.mean(embeddings, axis=0)
                                padding_needed = len(sequence) - embeddings.shape[0]
                                padding = np.tile(mean_embedding, (padding_needed, 1))
                                embeddings = np.vstack([embeddings, padding])
                        
                        has_embeddings = True
                        
                        # Add the specific embedding dimensions
                        for dim in embedding_dims:
                            if dim < embeddings.shape[1]:
                                df[f'emb_{dim}'] = embeddings[:, dim]
                            else:
                                # Handle missing dimensions with zero padding
                                df[f'emb_{dim}'] = 0.0
                        
                        # Add statistical features derived from embeddings
                        df['emb_mean'] = np.mean(embeddings, axis=1)
                        df['emb_std'] = np.std(embeddings, axis=1)
                        df['emb_max'] = np.max(embeddings, axis=1)
                        df['emb_min'] = np.min(embeddings, axis=1)
                        df['emb_norm'] = np.linalg.norm(embeddings, axis=1)
                        
                        # Additional embedding-derived features
                        df['emb_variance'] = np.var(embeddings, axis=1)
                        df['emb_skewness'] = self._calculate_skewness(embeddings)
                        df['emb_kurtosis'] = self._calculate_kurtosis(embeddings)
                        
                        break  # Successfully loaded embeddings
                        
                    except Exception as e:
                        print(f"Warning: Could not load embeddings from {filename} for {protein_id}: {e}")
                        continue
        
        # Fallback: Generate synthetic embeddings if no real embeddings available
        if not has_embeddings:
            print(f"No embeddings found for {protein_id}, using sequence-based fallback")
            
            # Create synthetic embeddings based on sequence properties
            synthetic_embeddings = self._generate_synthetic_embeddings(sequence)
            
            # Add all required embedding features with synthetic values
            for dim in embedding_dims:
                if dim < synthetic_embeddings.shape[1]:
                    df[f'emb_{dim}'] = synthetic_embeddings[:, dim]
                else:
                    df[f'emb_{dim}'] = 0.0
            
            # Add statistical features from synthetic embeddings
            df['emb_mean'] = np.mean(synthetic_embeddings, axis=1)
            df['emb_std'] = np.std(synthetic_embeddings, axis=1)
            df['emb_max'] = np.max(synthetic_embeddings, axis=1)
            df['emb_min'] = np.min(synthetic_embeddings, axis=1)
            df['emb_norm'] = np.linalg.norm(synthetic_embeddings, axis=1)
            df['emb_variance'] = np.var(synthetic_embeddings, axis=1)
            df['emb_skewness'] = self._calculate_skewness(synthetic_embeddings)
            df['emb_kurtosis'] = self._calculate_kurtosis(synthetic_embeddings)
        
        # Add metadata about embedding quality
        df['has_embedding_data'] = float(has_embeddings)
        df['embedding_quality'] = 1.0 if has_embeddings else 0.5  # Quality score
        
        return df
    
    def _generate_synthetic_embeddings(self, sequence: str) -> np.ndarray:
        """Generate synthetic embeddings based on amino acid properties when real embeddings are unavailable."""
        # Create a feature vector for each residue based on amino acid properties
        L = len(sequence)
        n_features = 1024  # Standard ProtT5 embedding dimension
        
        synthetic_embeddings = np.zeros((L, n_features))
        
        for i, aa in enumerate(sequence):
            # Base features on amino acid properties
            base_features = np.array([
                AA_PROPERTIES['hydropathy'].get(aa, 0),
                AA_PROPERTIES['polarity'].get(aa, 0),
                AA_PROPERTIES['charge_pH7'].get(aa, 0),
                AA_PROPERTIES['vdw_volume'].get(aa, 100),
                AA_PROPERTIES['flexibility'].get(aa, 0.4),
                AA_PROPERTIES['bulkiness'].get(aa, 0.5),
                AA_PROPERTIES['helix_prop'].get(aa, 1.0),
                AA_PROPERTIES['sheet_prop'].get(aa, 1.0),
            ])
            
            # Expand to full dimension with noise and positional encoding
            for j in range(n_features):
                if j < len(base_features):
                    synthetic_embeddings[i, j] = base_features[j]
                else:
                    # Add controlled noise based on position and property combinations
                    noise = np.sin(i * 0.1 + j * 0.01) * 0.1
                    prop_combination = (base_features[j % len(base_features)] + 
                                      base_features[(j + 1) % len(base_features)]) * 0.5
                    synthetic_embeddings[i, j] = prop_combination + noise
        
        return synthetic_embeddings
    
    def _calculate_skewness(self, embeddings: np.ndarray) -> List[float]:
        """Calculate skewness for each position's embedding vector."""
        from scipy import stats
        try:
            return [stats.skew(row) if len(row) > 2 else 0.0 for row in embeddings]
        except:
            return [0.0] * embeddings.shape[0]
    
    def _calculate_kurtosis(self, embeddings: np.ndarray) -> List[float]:
        """Calculate kurtosis for each position's embedding vector."""
        from scipy import stats
        try:
            return [stats.kurtosis(row) if len(row) > 3 else 0.0 for row in embeddings]
        except:
            return [0.0] * embeddings.shape[0]