"""
NEGASYS Data Loader Module
Reads real conjoint data in TV case format (.01b utility files, .01c competitor files)

Author: P.V. (Sundar) Balakrishnan
Version: 5.0
Date: December 2025

This module provides functions to load real conjoint utility data and competitor 
product definitions from the TV case format used in PRODLINE.

File Formats:
-------------
Utility File (.01b):
- Line 1: n_consumers, n_attributes, levels_attr_1, levels_attr_2, ...
- Lines 2 to n+1: Part-worths per consumer (one-hot encoded, flattened across all levels)
- Remaining lines: Attribute level labels (one per line)

Competition File (.01c):
- Line 1: n_competitors, n_attributes, levels_attr_1, levels_attr_2, ...
- Lines 2 to n+1: One-hot encoded product profiles per competitor
- Remaining lines: Attribute level labels (one per line)

Household Pairing:
------------------
For NEGASYS, individual consumers are paired into households:
- Consecutive pairs: rows 1-2 become HH 0, rows 3-4 become HH 1, etc.
- If odd number of consumers, the last one is duplicated to form a household
- Alpha (bargaining power) defaults to 0.5 but can be specified separately
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import io

# Import core data structures
from negasys_core_v3 import (
    ProductProfile, Household, HouseholdUtilityData
)


# =============================================================================
# DATA STRUCTURES FOR LOADED DATA
# =============================================================================

@dataclass
class LoadedUtilityData:
    """Container for raw loaded utility data before household pairing."""
    num_consumers: int
    num_attributes: int
    attribute_levels: List[int]
    partworths: np.ndarray  # Shape: (num_consumers, num_attributes, max_levels)
    attribute_names: Optional[List[str]] = None
    level_names: Optional[List[List[str]]] = None


@dataclass
class LoadedCompetitorData:
    """Container for loaded competitor products."""
    num_competitors: int
    num_attributes: int
    attribute_levels: List[int]
    products: List[ProductProfile]
    attribute_names: Optional[List[str]] = None
    level_names: Optional[List[List[str]]] = None


# =============================================================================
# UTILITY FILE PARSER (.01b format)
# =============================================================================

def load_utility_file_tv_format(filepath_or_content: str, 
                                 is_file_content: bool = False) -> LoadedUtilityData:
    """
    Load conjoint utility data from TV-style .01b format.
    
    Args:
        filepath_or_content: Either a file path or raw file content string
        is_file_content: If True, first arg is content; if False, it's a path
    
    Returns:
        LoadedUtilityData with individual-level part-worths
    
    File Format:
        Line 1: n_consumers, n_attributes, levels_1, levels_2, ...
        Lines 2 to n+1: Flattened one-hot part-worths (comma-separated)
        Remaining lines: Level names (one per line, grouped by attribute)
    """
    if is_file_content:
        lines = filepath_or_content.strip().split('\n')
    else:
        with open(filepath_or_content, 'r') as f:
            lines = f.readlines()
    
    # Clean lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Parse header
    header_parts = lines[0].split(',')
    header_parts = [p.strip() for p in header_parts if p.strip()]
    
    num_consumers = int(header_parts[0])
    num_attributes = int(header_parts[1])
    attribute_levels = [int(header_parts[2 + k]) for k in range(num_attributes)]
    
    total_levels = sum(attribute_levels)
    max_levels = max(attribute_levels)
    
    # Parse part-worths
    partworths = np.zeros((num_consumers, num_attributes, max_levels))
    
    for c_idx in range(num_consumers):
        line_idx = 1 + c_idx
        if line_idx >= len(lines):
            raise ValueError(f"Expected {num_consumers} consumer rows, got {c_idx}")
        
        values = [float(v.strip()) for v in lines[line_idx].split(',') if v.strip()]
        
        if len(values) < total_levels:
            raise ValueError(
                f"Consumer {c_idx}: expected {total_levels} values, got {len(values)}"
            )
        
        # Unpack flattened one-hot into matrix form
        val_idx = 0
        for k in range(num_attributes):
            for lv in range(attribute_levels[k]):
                partworths[c_idx, k, lv] = values[val_idx]
                val_idx += 1
    
    # Parse attribute and level names (if present)
    attribute_names = []
    level_names = [[] for _ in range(num_attributes)]
    
    label_start = 1 + num_consumers
    if label_start < len(lines):
        label_lines = lines[label_start:]
        
        # First num_attributes labels are attribute names (or first level of each)
        # Then levels follow in sequence
        label_idx = 0
        for k in range(num_attributes):
            attr_level_names = []
            for lv in range(attribute_levels[k]):
                if label_idx < len(label_lines):
                    name = label_lines[label_idx].split(',')[0].strip()
                    name = name.strip('"').replace('""', '"')  # Handle quoted strings
                    attr_level_names.append(name)
                    label_idx += 1
                else:
                    attr_level_names.append(f"Level_{lv+1}")
            
            level_names[k] = attr_level_names
            # Use first level name or generate attribute name
            if attr_level_names:
                attribute_names.append(f"Attribute_{k+1}")
    
    # Generate default names if not parsed
    if not attribute_names:
        attribute_names = [f"Attribute_{k+1}" for k in range(num_attributes)]
    
    return LoadedUtilityData(
        num_consumers=num_consumers,
        num_attributes=num_attributes,
        attribute_levels=attribute_levels,
        partworths=partworths,
        attribute_names=attribute_names,
        level_names=level_names if any(level_names) else None
    )


# =============================================================================
# COMPETITOR FILE PARSER (.01c format)
# =============================================================================

def load_competitor_file_tv_format(filepath_or_content: str,
                                    is_file_content: bool = False) -> LoadedCompetitorData:
    """
    Load competitor products from TV-style .01c format.
    
    Args:
        filepath_or_content: Either a file path or raw file content string
        is_file_content: If True, first arg is content; if False, it's a path
    
    Returns:
        LoadedCompetitorData with competitor ProductProfiles
    
    File Format:
        Line 1: n_competitors, n_attributes, levels_1, levels_2, ...
        Lines 2 to n+1: One-hot encoded product profiles
        Remaining lines: Level names (optional)
    """
    if is_file_content:
        lines = filepath_or_content.strip().split('\n')
    else:
        with open(filepath_or_content, 'r') as f:
            lines = f.readlines()
    
    # Clean lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Parse header
    header_parts = lines[0].split(',')
    header_parts = [p.strip() for p in header_parts if p.strip()]
    
    num_competitors = int(header_parts[0])
    num_attributes = int(header_parts[1])
    attribute_levels = [int(header_parts[2 + k]) for k in range(num_attributes)]
    
    total_levels = sum(attribute_levels)
    
    # Parse products (one-hot encoded)
    products = []
    
    for p_idx in range(num_competitors):
        line_idx = 1 + p_idx
        if line_idx >= len(lines):
            raise ValueError(f"Expected {num_competitors} product rows, got {p_idx}")
        
        values = [int(float(v.strip())) for v in lines[line_idx].split(',') if v.strip()]
        
        if len(values) < total_levels:
            raise ValueError(
                f"Product {p_idx}: expected {total_levels} values, got {len(values)}"
            )
        
        # Decode one-hot to attribute level indices
        attrs = np.zeros(num_attributes, dtype=np.int32)
        val_idx = 0
        
        for k in range(num_attributes):
            # Find which level is set to 1
            found = False
            for lv in range(attribute_levels[k]):
                if values[val_idx + lv] == 1:
                    attrs[k] = lv  # 0-indexed level
                    found = True
                    break
            
            if not found:
                # Default to first level if no 1 found
                attrs[k] = 0
            
            val_idx += attribute_levels[k]
        
        products.append(ProductProfile(attributes=attrs))
    
    # Parse level names (if present)
    attribute_names = [f"Attribute_{k+1}" for k in range(num_attributes)]
    level_names = [[] for _ in range(num_attributes)]
    
    label_start = 1 + num_competitors
    if label_start < len(lines):
        label_lines = lines[label_start:]
        label_idx = 0
        
        for k in range(num_attributes):
            attr_level_names = []
            for lv in range(attribute_levels[k]):
                if label_idx < len(label_lines):
                    name = label_lines[label_idx].split(',')[0].strip()
                    name = name.strip('"').replace('""', '"')
                    attr_level_names.append(name)
                    label_idx += 1
                else:
                    attr_level_names.append(f"Level_{lv+1}")
            level_names[k] = attr_level_names
    
    return LoadedCompetitorData(
        num_competitors=num_competitors,
        num_attributes=num_attributes,
        attribute_levels=attribute_levels,
        products=products,
        attribute_names=attribute_names,
        level_names=level_names if any(level_names) else None
    )


# =============================================================================
# HOUSEHOLD PAIRING
# =============================================================================

def pair_consumers_to_households(
    utility_data: LoadedUtilityData,
    competitors: List[ProductProfile],
    alpha_values: Optional[List[float]] = None,
    alpha_default: float = 0.5,
    pairing_mode: str = "consecutive"
) -> HouseholdUtilityData:
    """
    Convert individual consumer utilities to NEGASYS household format.
    
    Pairing Modes:
    - "consecutive": Pair rows 0-1, 2-3, 4-5, etc.
    - "random": Randomly pair consumers (requires even number)
    
    If odd number of consumers, the last consumer is duplicated.
    
    Args:
        utility_data: LoadedUtilityData from load_utility_file_tv_format
        competitors: List of competitor ProductProfiles (for initial status quo)
        alpha_values: Optional list of alpha values per household
        alpha_default: Default alpha if not specified
        pairing_mode: How to pair consumers ("consecutive" or "random")
    
    Returns:
        HouseholdUtilityData ready for NEGASYS optimization
    """
    num_consumers = utility_data.num_consumers
    num_attributes = utility_data.num_attributes
    attribute_levels = utility_data.attribute_levels
    partworths = utility_data.partworths
    
    # Handle odd number of consumers
    if num_consumers % 2 == 1:
        # Duplicate the last consumer
        new_partworths = np.zeros((num_consumers + 1, num_attributes, partworths.shape[2]))
        new_partworths[:num_consumers] = partworths
        new_partworths[num_consumers] = partworths[num_consumers - 1]  # Duplicate last
        partworths = new_partworths
        num_consumers += 1
    
    num_households = num_consumers // 2
    
    # Create pairing indices
    if pairing_mode == "random":
        import random
        indices = list(range(num_consumers))
        random.shuffle(indices)
        pairs = [(indices[2*i], indices[2*i+1]) for i in range(num_households)]
    else:  # consecutive
        pairs = [(2*i, 2*i+1) for i in range(num_households)]
    
    # Create households
    households = []
    
    for h_id, (idx_H, idx_W) in enumerate(pairs):
        # Get alpha for this household
        if alpha_values and h_id < len(alpha_values):
            alpha = alpha_values[h_id]
        else:
            alpha = alpha_default
        
        # Assign random status quo from competitors
        if competitors:
            sq_idx = h_id % len(competitors)
            status_quo = competitors[sq_idx].copy()
        else:
            # No competitors provided; use random status quo
            sq_attrs = np.array([
                np.random.randint(0, attribute_levels[k])
                for k in range(num_attributes)
            ], dtype=np.int32)
            status_quo = ProductProfile(attributes=sq_attrs)
        
        household = Household(
            id=h_id,
            partworths_H=partworths[idx_H],
            partworths_W=partworths[idx_W],
            status_quo=status_quo,
            alpha=alpha,
            weight=1.0
        )
        households.append(household)
    
    return HouseholdUtilityData(
        num_households=num_households,
        num_attributes=num_attributes,
        attribute_levels=attribute_levels,
        households=households,
        attribute_names=utility_data.attribute_names,
        level_names=utility_data.level_names
    )


# =============================================================================
# COMBINED LOADER FUNCTION
# =============================================================================

def load_negasys_data_tv_format(
    utility_filepath_or_content: str,
    competitor_filepath_or_content: str,
    utility_is_content: bool = False,
    competitor_is_content: bool = False,
    alpha_values: Optional[List[float]] = None,
    alpha_default: float = 0.5,
    pairing_mode: str = "consecutive"
) -> Tuple[HouseholdUtilityData, List[ProductProfile], Dict[str, Any]]:
    """
    Load complete NEGASYS data from TV-format files.
    
    This is the main entry point for loading real data.
    
    Args:
        utility_filepath_or_content: Path or content of utility file (.01b)
        competitor_filepath_or_content: Path or content of competitor file (.01c)
        utility_is_content: True if utility arg is content string
        competitor_is_content: True if competitor arg is content string
        alpha_values: Optional list of alpha per household
        alpha_default: Default alpha value
        pairing_mode: "consecutive" or "random"
    
    Returns:
        Tuple of:
        - HouseholdUtilityData ready for NEGASYS
        - List of competitor ProductProfiles (for Stage 0)
        - Metadata dict with parsing info
    """
    # Load utilities
    utility_data = load_utility_file_tv_format(
        utility_filepath_or_content, 
        is_file_content=utility_is_content
    )
    
    # Load competitors
    competitor_data = load_competitor_file_tv_format(
        competitor_filepath_or_content,
        is_file_content=competitor_is_content
    )
    
    # Validate compatibility
    if utility_data.num_attributes != competitor_data.num_attributes:
        raise ValueError(
            f"Attribute count mismatch: utility has {utility_data.num_attributes}, "
            f"competitor has {competitor_data.num_attributes}"
        )
    
    if utility_data.attribute_levels != competitor_data.attribute_levels:
        raise ValueError(
            f"Attribute levels mismatch: utility has {utility_data.attribute_levels}, "
            f"competitor has {competitor_data.attribute_levels}"
        )
    
    # Pair consumers into households
    household_data = pair_consumers_to_households(
        utility_data=utility_data,
        competitors=competitor_data.products,
        alpha_values=alpha_values,
        alpha_default=alpha_default,
        pairing_mode=pairing_mode
    )
    
    # Use competitor level names if utility didn't have them
    if not household_data.level_names and competitor_data.level_names:
        household_data.level_names = competitor_data.level_names
    
    # Build metadata
    metadata = {
        'original_num_consumers': utility_data.num_consumers,
        'num_households': household_data.num_households,
        'num_attributes': household_data.num_attributes,
        'attribute_levels': household_data.attribute_levels,
        'num_competitors': competitor_data.num_competitors,
        'pairing_mode': pairing_mode,
        'alpha_default': alpha_default,
        'attribute_names': household_data.attribute_names,
        'level_names': household_data.level_names
    }
    
    return household_data, competitor_data.products, metadata


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def describe_loaded_data(household_data: HouseholdUtilityData,
                         competitors: List[ProductProfile],
                         metadata: Dict[str, Any]) -> str:
    """Generate a summary description of loaded data."""
    lines = []
    lines.append("=" * 60)
    lines.append("NEGASYS DATA SUMMARY (TV Format)")
    lines.append("=" * 60)
    
    lines.append(f"\n[Source Data]")
    lines.append(f"  Original consumers: {metadata['original_num_consumers']}")
    lines.append(f"  Pairing mode: {metadata['pairing_mode']}")
    
    lines.append(f"\n[Household Structure]")
    lines.append(f"  Households: {metadata['num_households']}")
    lines.append(f"  Attributes: {metadata['num_attributes']}")
    lines.append(f"  Levels: {metadata['attribute_levels']}")
    
    lines.append(f"\n[Attributes and Levels]")
    attr_names = metadata.get('attribute_names', [])
    level_names = metadata.get('level_names', [])
    
    for k in range(metadata['num_attributes']):
        attr_name = attr_names[k] if k < len(attr_names) else f"Attribute_{k+1}"
        lines.append(f"  {attr_name}:")
        if level_names and k < len(level_names):
            for lv, lv_name in enumerate(level_names[k]):
                lines.append(f"    Level {lv}: {lv_name}")
        else:
            for lv in range(metadata['attribute_levels'][k]):
                lines.append(f"    Level {lv}: (unnamed)")
    
    lines.append(f"\n[Competitor Products]")
    lines.append(f"  Count: {metadata['num_competitors']}")
    for i, prod in enumerate(competitors):
        lines.append(f"  Competitor {i+1}: {prod.attributes.tolist()}")
    
    # Sample household info
    lines.append(f"\n[Sample Household (HH 0)]")
    if household_data.households:
        h = household_data.households[0]
        lines.append(f"  Alpha: {h.alpha:.2f}")
        lines.append(f"  Status quo: {h.status_quo.attributes.tolist()}")
        lines.append(f"  H part-worths shape: {h.partworths_H.shape}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def validate_data_consistency(household_data: HouseholdUtilityData,
                              competitors: List[ProductProfile]) -> List[str]:
    """
    Validate loaded data for consistency.
    
    Returns list of warning messages (empty if no issues).
    """
    warnings = []
    
    # Check attribute dimensions
    for h in household_data.households:
        if h.partworths_H.shape[0] != household_data.num_attributes:
            warnings.append(f"HH {h.id}: H partworths dimension mismatch")
        if h.partworths_W.shape[0] != household_data.num_attributes:
            warnings.append(f"HH {h.id}: W partworths dimension mismatch")
    
    # Check competitor attributes
    for i, prod in enumerate(competitors):
        if len(prod.attributes) != household_data.num_attributes:
            warnings.append(f"Competitor {i}: attribute count mismatch")
    
    # Check alpha range
    for h in household_data.households:
        if not 0 <= h.alpha <= 1:
            warnings.append(f"HH {h.id}: alpha {h.alpha} out of [0,1] range")
    
    return warnings


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test with sample data
    print("NEGASYS Data Loader Module - Test")
    print("=" * 60)
    
    # Sample utility content (TV format)
    sample_utility = """4,2,2,3,,,,
0.3,0.7,0.2,0.5,0.3
0.4,0.6,0.1,0.4,0.5
0.5,0.5,0.3,0.3,0.4
0.6,0.4,0.4,0.2,0.4
Level_A1
Level_A2
Level_B1
Level_B2
Level_B3"""
    
    sample_competitor = """2,2,2,3,,,,
1,0,1,0,0
0,1,0,0,1
Level_A1
Level_A2
Level_B1
Level_B2
Level_B3"""
    
    print("\n[Loading sample data...]")
    
    try:
        household_data, competitors, metadata = load_negasys_data_tv_format(
            utility_filepath_or_content=sample_utility,
            competitor_filepath_or_content=sample_competitor,
            utility_is_content=True,
            competitor_is_content=True,
            alpha_default=0.5,
            pairing_mode="consecutive"
        )
        
        print(describe_loaded_data(household_data, competitors, metadata))
        
        # Validate
        warnings = validate_data_consistency(household_data, competitors)
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        else:
            print("\nNo validation warnings.")
        
        print("\nTest passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
