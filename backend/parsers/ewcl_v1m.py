"""
EWCL v1-M (Machine Learning) Feature Extractor
Handles all 255 features including PSSM data for machine learning predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import Counter
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EWCLv1MParser:
    """
    EWCL v1-M feature parser for machine learning predictions.
    Extracts 255 features including base sequence features and PSSM data.
    """
    
    # Amino acid properties
    HYDROPATHY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
        'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    POLARITY = {
        'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0, 'Q': 1, 'E': 1,
        'G': 0, 'H': 1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
        'P': 0, 'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
    }
    
    VDW_VOLUME = {
        'A': 67, 'R': 148, 'N': 96, 'D': 91, 'C': 86, 'Q': 114, 'E': 109,
        'G': 48, 'H': 118, 'I': 124, 'L': 124, 'K': 135, 'M': 124, 'F': 135,
        'P': 90, 'S': 73, 'T': 93, 'W': 163, 'Y': 141, 'V': 105
    }
    
    FLEXIBILITY = {
        'A': 0.25, 'R': 0.25, 'N': 0.25, 'D': 0.25, 'C': 0.25, 'Q': 0.25, 'E': 0.25,
        'G': 0.00, 'H': 0.25, 'I': 0.25, 'L': 0.25, 'K': 0.25, 'M': 0.25, 'F': 0.25,
        'P': 0.50, 'S': 0.25, 'T': 0.25, 'W': 0.25, 'Y': 0.25, 'V': 0.25
    }
    
    BULKINESS = {
        'A': 11.5, 'R': 14.28, 'N': 12.82, 'D': 11.68, 'C': 13.46, 'Q': 14.45, 'E': 13.57,
        'G': 3.4, 'H': 13.69, 'I': 21.4, 'L': 21.4, 'K': 15.71, 'M': 16.25, 'F': 19.8,
        'P': 17.43, 'S': 9.47, 'T': 15.77, 'W': 27.68, 'Y': 18.03, 'V': 21.57
    }
    
    HELIX_PROP = {
        'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70, 'Q': 1.11, 'E': 1.51,
        'G': 0.57, 'H': 1.00, 'I': 1.08, 'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13,
        'P': 0.57, 'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
    }
    
    SHEET_PROP = {
        'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19, 'Q': 1.10, 'E': 0.37,
        'G': 0.75, 'H': 0.87, 'I': 1.60, 'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38,
        'P': 0.55, 'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
    }
    
    CHARGE_PH7 = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0, 'Q': 0, 'E': -1,
        'G': 0, 'H': 0.1, 'I': 0, 'L': 0, 'K': 1, 'M': 0, 'F': 0,
        'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    
    # Amino acid classifications
    HYDROPHOBIC = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y'}
    POLAR = {'S', 'T', 'N', 'Q', 'C'}
    CHARGED = {'D', 'E', 'K', 'R', 'H'}
    AROMATIC = {'F', 'W', 'Y', 'H'}
    POSITIVE = {'K', 'R', 'H'}
    NEGATIVE = {'D', 'E'}
    ALIPHATIC = {'A', 'V', 'L', 'I'}
    
    def __init__(self):
        """Initialize the EWCLv1-M parser."""
        self.feature_order = self._get_feature_order()
    
    def _get_feature_order(self) -> List[str]:
        """Get the exact feature order for EWCLv1-M model (255 features)."""
        return [
            # Base features (235 features)
            "is_unknown_aa", "hydropathy", "polarity", "vdw_volume", "flexibility", 
            "bulkiness", "helix_prop", "sheet_prop", "charge_pH7", "scd_local",
            
            # Window features (w5, w11, w25, w50, w100)
            *[f"{prop}_w{w}_{stat}" for w in [5, 11, 25, 50, 100] 
              for prop in ["hydro", "polar", "vdw", "flex", "bulk", "helix_prop", "sheet_prop", "charge"]
              for stat in ["mean", "std", "min", "max"]],
            
            *[f"{prop}_w{w}" for w in [5, 11, 25, 50, 100] 
              for prop in ["entropy", "low_complex", "comp_bias", "uversky_dist"]],
            
            # Amino acid composition
            *[f"comp_{aa}" for aa in "DYFMVRPALITWQNKEGSHC"],
            "comp_frac_aromatic", "comp_frac_positive", "comp_frac_negative", 
            "comp_frac_polar", "comp_frac_aliphatic", "comp_frac_proline", "comp_frac_glycine",
            
            # Poly runs
            *[f"in_poly_{aa}_run_ge3" for aa in "PEKQSGDN"],
            
            # Amino acid encoding (20 features for one-hot encoding)
            *list("ARNDCQEGHILKMFPSTWYV"),
            
            # PSSM features (31 features)
            "pssm_entropy", "pssm_max_score", "pssm_variance", "pssm_native",
            "pssm_top1", "pssm_top2", "pssm_gap", "pssm_sum_hydrophobic", 
            "pssm_sum_polar", "pssm_sum_charged"
        ]
    
    def parse_variant(self, sequence: str, position: int, original_aa: str, 
                     variant_aa: str, pssm_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Parse a single variant and extract all 255 EWCLv1-M features.
        
        Args:
            sequence: Full protein sequence
            position: 1-based position of the variant
            original_aa: Original amino acid
            variant_aa: Variant amino acid
            pssm_data: Optional PSSM data dictionary
            
        Returns:
            Dictionary with all 255 features
        """
        try:
            # Convert to 0-based indexing
            pos_idx = position - 1
            
            # Validate inputs
            if pos_idx < 0 or pos_idx >= len(sequence):
                raise ValueError(f"Position {position} out of range for sequence length {len(sequence)}")
            
            if sequence[pos_idx].upper() != original_aa.upper():
                logger.warning(f"Sequence mismatch at position {position}: expected {original_aa}, found {sequence[pos_idx]}")
            
            # Create mutated sequence
            mutated_sequence = sequence[:pos_idx] + variant_aa + sequence[pos_idx + 1:]
            
            # Extract all features
            features = {}
            
            # Basic amino acid features
            features.update(self._extract_basic_features(variant_aa, pos_idx, sequence))
            
            # Window-based features
            features.update(self._extract_window_features(mutated_sequence, pos_idx))
            
            # Composition features
            features.update(self._extract_composition_features(mutated_sequence))
            
            # Poly run features
            features.update(self._extract_poly_run_features(mutated_sequence, pos_idx))
            
            # Amino acid encoding (one-hot)
            features.update(self._extract_aa_encoding(variant_aa))
            
            # PSSM features
            features.update(self._extract_pssm_features(pssm_data, pos_idx, variant_aa))
            
            # Ensure we have exactly 255 features
            if len(features) != 255:
                logger.warning(f"Expected 255 features, got {len(features)}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error parsing variant {original_aa}{position}{variant_aa}: {str(e)}")
            raise
    
    def _extract_basic_features(self, aa: str, pos_idx: int, sequence: str) -> Dict[str, float]:
        """Extract basic amino acid properties."""
        aa = aa.upper()
        
        features = {
            'is_unknown_aa': 1.0 if aa not in self.HYDROPATHY else 0.0,
            'hydropathy': self.HYDROPATHY.get(aa, 0.0),
            'polarity': self.POLARITY.get(aa, 0.0),
            'vdw_volume': self.VDW_VOLUME.get(aa, 0.0),
            'flexibility': self.FLEXIBILITY.get(aa, 0.0),
            'bulkiness': self.BULKINESS.get(aa, 0.0),
            'helix_prop': self.HELIX_PROP.get(aa, 0.0),
            'sheet_prop': self.SHEET_PROP.get(aa, 0.0),
            'charge_pH7': self.CHARGE_PH7.get(aa, 0.0),
        }
        
        # Simple sequence complexity for local region (±5 residues)
        start = max(0, pos_idx - 5)
        end = min(len(sequence), pos_idx + 6)
        local_seq = sequence[start:end]
        features['scd_local'] = len(set(local_seq)) / len(local_seq) if local_seq else 0.0
        
        return features
    
    def _extract_window_features(self, sequence: str, pos_idx: int) -> Dict[str, float]:
        """Extract window-based statistical features."""
        features = {}
        
        windows = [5, 11, 25, 50, 100]
        properties = {
            'hydro': self.HYDROPATHY,
            'polar': self.POLARITY,
            'vdw': self.VDW_VOLUME,
            'flex': self.FLEXIBILITY,
            'bulk': self.BULKINESS,
            'helix_prop': self.HELIX_PROP,
            'sheet_prop': self.SHEET_PROP,
            'charge': self.CHARGE_PH7
        }
        
        for window in windows:
            half_window = window // 2
            start = max(0, pos_idx - half_window)
            end = min(len(sequence), pos_idx + half_window + 1)
            window_seq = sequence[start:end]
            
            if not window_seq:
                continue
            
            # Property-based features
            for prop_name, prop_dict in properties.items():
                values = [prop_dict.get(aa.upper(), 0.0) for aa in window_seq]
                
                if values:
                    features[f"{prop_name}_w{window}_mean"] = np.mean(values)
                    features[f"{prop_name}_w{window}_std"] = np.std(values)
                    features[f"{prop_name}_w{window}_min"] = np.min(values)
                    features[f"{prop_name}_w{window}_max"] = np.max(values)
                else:
                    features[f"{prop_name}_w{window}_mean"] = 0.0
                    features[f"{prop_name}_w{window}_std"] = 0.0
                    features[f"{prop_name}_w{window}_min"] = 0.0
                    features[f"{prop_name}_w{window}_max"] = 0.0
            
            # Sequence complexity features
            features[f"entropy_w{window}"] = self._calculate_entropy(window_seq)
            features[f"low_complex_w{window}"] = self._calculate_low_complexity(window_seq)
            features[f"comp_bias_w{window}"] = self._calculate_composition_bias(window_seq)
            features[f"uversky_dist_w{window}"] = self._calculate_uversky_distance(window_seq)
        
        return features
    
    def _extract_composition_features(self, sequence: str) -> Dict[str, float]:
        """Extract amino acid composition features."""
        features = {}
        seq_len = len(sequence)
        
        if seq_len == 0:
            return {f"comp_{aa}": 0.0 for aa in "DYFMVRPALITWQNKEGSHC"}
        
        # Count amino acids
        aa_counts = Counter(sequence.upper())
        
        # Individual amino acid compositions
        for aa in "DYFMVRPALITWQNKEGSHC":
            features[f"comp_{aa}"] = aa_counts.get(aa, 0) / seq_len
        
        # Functional group compositions
        features["comp_frac_aromatic"] = sum(aa_counts.get(aa, 0) for aa in self.AROMATIC) / seq_len
        features["comp_frac_positive"] = sum(aa_counts.get(aa, 0) for aa in self.POSITIVE) / seq_len
        features["comp_frac_negative"] = sum(aa_counts.get(aa, 0) for aa in self.NEGATIVE) / seq_len
        features["comp_frac_polar"] = sum(aa_counts.get(aa, 0) for aa in self.POLAR) / seq_len
        features["comp_frac_aliphatic"] = sum(aa_counts.get(aa, 0) for aa in self.ALIPHATIC) / seq_len
        features["comp_frac_proline"] = aa_counts.get('P', 0) / seq_len
        features["comp_frac_glycine"] = aa_counts.get('G', 0) / seq_len
        
        return features
    
    def _extract_poly_run_features(self, sequence: str, pos_idx: int) -> Dict[str, float]:
        """Extract poly amino acid run features."""
        features = {}
        
        for aa in "PEKQSGDN":
            features[f"in_poly_{aa}_run_ge3"] = 0.0
            
            # Check if position is in a poly run of ≥3 residues
            if pos_idx < len(sequence):
                # Find runs of the amino acid
                i = 0
                while i < len(sequence):
                    if sequence[i].upper() == aa:
                        run_start = i
                        while i < len(sequence) and sequence[i].upper() == aa:
                            i += 1
                        run_end = i
                        
                        # Check if run is ≥3 and contains our position
                        if (run_end - run_start >= 3 and 
                            run_start <= pos_idx < run_end):
                            features[f"in_poly_{aa}_run_ge3"] = 1.0
                            break
                    else:
                        i += 1
        
        return features
    
    def _extract_aa_encoding(self, aa: str) -> Dict[str, float]:
        """Extract one-hot amino acid encoding."""
        features = {}
        aa = aa.upper()
        
        for amino_acid in "ARNDCQEGHILKMFPSTWYV":
            features[amino_acid] = 1.0 if aa == amino_acid else 0.0
        
        return features
    
    def _extract_pssm_features(self, pssm_data: Optional[Dict], pos_idx: int, 
                              variant_aa: str) -> Dict[str, float]:
        """Extract PSSM-based features."""
        features = {
            "pssm_entropy": 0.0,
            "pssm_max_score": 0.0,
            "pssm_variance": 0.0,
            "pssm_native": 0.0,
            "pssm_top1": 0.0,
            "pssm_top2": 0.0,
            "pssm_gap": 0.0,
            "pssm_sum_hydrophobic": 0.0,
            "pssm_sum_polar": 0.0,
            "pssm_sum_charged": 0.0
        }
        
        if not pssm_data or pos_idx >= len(pssm_data.get('scores', [])):
            return features
        
        try:
            position_scores = pssm_data['scores'][pos_idx]
            aa_order = "ARNDCQEGHILKMFPSTWYV"
            
            if len(position_scores) >= 20:
                scores = np.array(position_scores[:20])
                
                # Basic PSSM statistics
                features["pssm_entropy"] = self._calculate_pssm_entropy(scores)
                features["pssm_max_score"] = float(np.max(scores))
                features["pssm_variance"] = float(np.var(scores))
                
                # Variant-specific scores
                if variant_aa.upper() in aa_order:
                    aa_idx = aa_order.index(variant_aa.upper())
                    features["pssm_native"] = float(scores[aa_idx])
                
                # Top scores
                sorted_scores = np.sort(scores)[::-1]
                features["pssm_top1"] = float(sorted_scores[0])
                features["pssm_top2"] = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
                
                # Gap penalty (if available)
                if len(position_scores) > 20:
                    features["pssm_gap"] = float(position_scores[20])
                
                # Functional group sums
                features["pssm_sum_hydrophobic"] = sum(scores[aa_order.index(aa)] 
                                                     for aa in self.HYDROPHOBIC if aa in aa_order)
                features["pssm_sum_polar"] = sum(scores[aa_order.index(aa)] 
                                               for aa in self.POLAR if aa in aa_order)
                features["pssm_sum_charged"] = sum(scores[aa_order.index(aa)] 
                                                 for aa in self.CHARGED if aa in aa_order)
                
        except Exception as e:
            logger.warning(f"Error extracting PSSM features: {str(e)}")
        
        return features
    
    def _calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of sequence."""
        if not sequence:
            return 0.0
        
        counts = Counter(sequence.upper())
        total = len(sequence)
        entropy = 0.0
        
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_low_complexity(self, sequence: str) -> float:
        """Calculate low complexity score."""
        if not sequence:
            return 0.0
        
        # Simple implementation: fraction of most common amino acid
        if len(sequence) == 0:
            return 0.0
        
        counts = Counter(sequence.upper())
        max_count = max(counts.values()) if counts else 0
        return max_count / len(sequence)
    
    def _calculate_composition_bias(self, sequence: str) -> float:
        """Calculate composition bias score."""
        if not sequence:
            return 0.0
        
        # Measure deviation from uniform distribution
        expected_freq = 1.0 / 20  # Uniform distribution
        counts = Counter(sequence.upper())
        total = len(sequence)
        
        bias = 0.0
        for aa in "ARNDCQEGHILKMFPSTWYV":
            observed_freq = counts.get(aa, 0) / total
            bias += abs(observed_freq - expected_freq)
        
        return bias
    
    def _calculate_uversky_distance(self, sequence: str) -> float:
        """Calculate Uversky plot distance (disorder prediction)."""
        if not sequence:
            return 0.0
        
        # Simplified Uversky calculation
        hydrophobic_count = sum(1 for aa in sequence.upper() if aa in self.HYDROPHOBIC)
        charged_count = sum(1 for aa in sequence.upper() if aa in self.CHARGED)
        
        if len(sequence) == 0:
            return 0.0
        
        hydrophobic_frac = hydrophobic_count / len(sequence)
        charged_frac = charged_count / len(sequence)
        
        # Distance from folded protein boundary
        return abs(hydrophobic_frac - 0.45) + abs(charged_frac - 0.25)
    
    def _calculate_pssm_entropy(self, scores: np.ndarray) -> float:
        """Calculate entropy from PSSM scores."""
        try:
            # Convert scores to probabilities
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            probabilities = exp_scores / np.sum(exp_scores)
            
            # Calculate entropy
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            return entropy
        except:
            return 0.0

    def parse_multiple_variants(self, variants_data: List[Dict]) -> List[Dict[str, float]]:
        """
        Parse multiple variants efficiently.
        
        Args:
            variants_data: List of variant dictionaries with keys:
                - sequence: protein sequence
                - position: 1-based position
                - original_aa: original amino acid
                - variant_aa: variant amino acid
                - pssm_data: optional PSSM data
        
        Returns:
            List of feature dictionaries
        """
        results = []
        
        for variant in variants_data:
            try:
                features = self.parse_variant(
                    sequence=variant['sequence'],
                    position=variant['position'], 
                    original_aa=variant['original_aa'],
                    variant_aa=variant['variant_aa'],
                    pssm_data=variant.get('pssm_data')
                )
                results.append(features)
            except Exception as e:
                logger.error(f"Error parsing variant {variant}: {str(e)}")
                # Return zero features for failed variants
                results.append({feature: 0.0 for feature in self.feature_order})
        
        return results

    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to ordered numpy array."""
        return np.array([features.get(feature, 0.0) for feature in self.feature_order])

# Create global parser instance
ewcl_v1m_parser = EWCLv1MParser()