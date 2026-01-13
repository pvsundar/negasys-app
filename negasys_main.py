"""
NEGASYS Main Orchestrator
Integrates Stage 0 (Status Quo Assignment) with Core v3 Optimization

Author: P.V. (Sundar) Balakrishnan
Version: 4.0
Date: December 2025

This module provides a clean entry point for running the complete NEGASYS pipeline:
1. Generate or load household data
2. Define competitor products (Stage 0)
3. Assign status quo via random or Nash-optimal selection
4. Run product line optimization (GA or DP)
5. Export results in multiple formats

New Features:
- Integrated Stage 0 status quo assignment
- Downloadable synthetic household data (CSV)
- Downloadable optimization results (CSV)
- Batch comparison of all 4 decision rules
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
import time
import io

# Import core modules
from negasys_core_v3 import (
    AggregationRule, ObjectiveType, TransformationType,
    ProductProfile, Household, HouseholdUtilityData,
    ProductLine, NEGASYSConfig, GAParams, OptimizationResult,
    run_negasys_ga, generate_synthetic_household_data,
    UtilityEngine
)


# =============================================================================
# STATUS QUO ASSIGNMENT (Stage 0 - Integrated)
# =============================================================================

class StatusQuoMode(IntEnum):
    """Status quo assignment modes."""
    RANDOM = 1              # Random assignment from competitors
    NASH_ARGMAX = 2         # Nash-optimal choice from competitors


@dataclass
class Stage0Result:
    """Results from Stage 0 status quo assignment."""
    mode: str
    num_households: int
    num_competitors: int
    product_counts: Dict[int, int]
    product_shares: Dict[int, float]
    competitor_products: List[ProductProfile]


class BaselineUtilityEngine:
    """
    Utility engine for Stage 0 baseline choice (no prior status quo).
    Normalizes utilities relative to the competitor set.
    """
    
    def __init__(self, config: NEGASYSConfig):
        self.config = config
    
    def compute_raw_utility(self, household: Household, product: ProductProfile, 
                           individual: str) -> float:
        partworths = household.partworths_H if individual == 'H' else household.partworths_W
        utility = 0.0
        for k, level in enumerate(product.attributes):
            utility += partworths[k, level]
        return utility
    
    def compute_normalized_utility(self, household: Household, product: ProductProfile,
                                   choice_set: List[ProductProfile], individual: str) -> float:
        eps = self.config.epsilon
        utilities = [self.compute_raw_utility(household, p, individual) for p in choice_set]
        min_u, max_u = min(utilities), max(utilities)
        focal_u = self.compute_raw_utility(household, product, individual)
        
        if max_u - min_u < eps:
            return 0.5
        return (focal_u - min_u + eps) / (max_u - min_u + eps)
    
    def compute_household_utility_baseline(self, household: Household, 
                                           product: ProductProfile,
                                           choice_set: List[ProductProfile]) -> float:
        v_H = self.compute_normalized_utility(household, product, choice_set, 'H')
        v_W = self.compute_normalized_utility(household, product, choice_set, 'W')
        alpha = household.alpha
        
        if self.config.aggregation_rule == AggregationRule.NASH:
            return v_H * v_W
        elif self.config.aggregation_rule == AggregationRule.ROTH:
            if v_H <= 0 or v_W <= 0:
                return self.config.epsilon ** 2
            return (v_H ** alpha) * (v_W ** (1 - alpha))
        elif self.config.aggregation_rule == AggregationRule.LINEAR:
            return alpha * v_H + (1 - alpha) * v_W
        elif self.config.aggregation_rule == AggregationRule.MIN:
            return min(v_H, v_W)
        else:
            return alpha * v_H + (1 - alpha) * v_W


def create_competitor_set(num_attributes: int, attribute_levels: List[int],
                         profiles: List[List[int]]) -> List[ProductProfile]:
    """Create competitor ProductProfiles from attribute level specifications."""
    competitor_products = []
    for profile in profiles:
        if len(profile) != num_attributes:
            raise ValueError(f"Profile {profile} has wrong number of attributes")
        for k, level in enumerate(profile):
            if level < 0 or level >= attribute_levels[k]:
                raise ValueError(f"Invalid level {level} for attribute {k}")
        attrs = np.array(profile, dtype=np.int32)
        competitor_products.append(ProductProfile(attributes=attrs))
    return competitor_products


def generate_random_competitors(num_competitors: int, num_attributes: int,
                               attribute_levels: List[int], 
                               seed: Optional[int] = None) -> List[ProductProfile]:
    """Generate random competitor products for simulation studies."""
    if seed is not None:
        np.random.seed(seed)
    
    competitors = []
    used = set()
    attempts = 0
    
    while len(competitors) < num_competitors and attempts < num_competitors * 100:
        attrs = np.array([np.random.randint(0, attribute_levels[k]) 
                         for k in range(num_attributes)], dtype=np.int32)
        key = tuple(attrs)
        if key not in used:
            used.add(key)
            competitors.append(ProductProfile(attributes=attrs))
        attempts += 1
    
    return competitors


def assign_status_quo(households: List[Household], 
                     competitor_products: List[ProductProfile],
                     config: NEGASYSConfig,
                     mode: StatusQuoMode = StatusQuoMode.NASH_ARGMAX,
                     seed: Optional[int] = None) -> Stage0Result:
    """
    Stage 0: Assign status quo products from competitor set.
    Modifies households in place.
    """
    import random
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    product_counts = {i: 0 for i in range(len(competitor_products))}
    
    if mode == StatusQuoMode.RANDOM:
        for household in households:
            idx = random.randint(0, len(competitor_products) - 1)
            household.status_quo = competitor_products[idx].copy()
            product_counts[idx] += 1
    else:  # NASH_ARGMAX
        engine = BaselineUtilityEngine(config)
        for household in households:
            best_utility = -np.inf
            best_idx = 0
            for idx, product in enumerate(competitor_products):
                A = engine.compute_household_utility_baseline(
                    household, product, competitor_products)
                if A > best_utility:
                    best_utility = A
                    best_idx = idx
                elif A == best_utility and random.random() < 0.5:
                    best_idx = idx
            household.status_quo = competitor_products[best_idx].copy()
            product_counts[best_idx] += 1
    
    num_households = len(households)
    product_shares = {i: count / num_households for i, count in product_counts.items()}
    
    return Stage0Result(
        mode="random_from_competitors" if mode == StatusQuoMode.RANDOM else "nash_argmax_from_competitors",
        num_households=num_households,
        num_competitors=len(competitor_products),
        product_counts=product_counts,
        product_shares=product_shares,
        competitor_products=competitor_products
    )


# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================

def export_household_data_csv(household_data: HouseholdUtilityData) -> str:
    """
    Export household data to CSV format for Excel/R testing.
    Returns CSV string.
    
    Format: One row per household-member with:
    - HH_ID, member (H/W), part-worths, alpha, status_quo attributes
    """
    rows = []
    
    for h in household_data.households:
        # Get attribute names
        attr_names = household_data.attribute_names or \
                    [f"attr_{k+1}" for k in range(household_data.num_attributes)]
        
        # Build part-worth column names
        pw_cols = []
        for k in range(household_data.num_attributes):
            for lv in range(household_data.attribute_levels[k]):
                pw_cols.append(f"{attr_names[k]}_lv{lv+1}")
        
        # Status quo columns
        sq_cols = [f"sq_{attr_names[k]}" for k in range(household_data.num_attributes)]
        
        # Row for H
        row_H = {
            'HH_ID': h.id,
            'member': 'H',
            'alpha': h.alpha,
            'weight': h.weight
        }
        
        # Add part-worths for H
        pw_idx = 0
        for k in range(household_data.num_attributes):
            for lv in range(household_data.attribute_levels[k]):
                row_H[pw_cols[pw_idx]] = h.partworths_H[k, lv]
                pw_idx += 1
        
        # Add status quo
        for k in range(household_data.num_attributes):
            row_H[sq_cols[k]] = h.status_quo.attributes[k] + 1  # 1-indexed
        
        rows.append(row_H)
        
        # Row for W
        row_W = {
            'HH_ID': h.id,
            'member': 'W',
            'alpha': h.alpha,
            'weight': h.weight
        }
        
        pw_idx = 0
        for k in range(household_data.num_attributes):
            for lv in range(household_data.attribute_levels[k]):
                row_W[pw_cols[pw_idx]] = h.partworths_W[k, lv]
                pw_idx += 1
        
        for k in range(household_data.num_attributes):
            row_W[sq_cols[k]] = h.status_quo.attributes[k] + 1
        
        rows.append(row_W)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def export_optimization_results_csv(result: OptimizationResult,
                                   household_data: HouseholdUtilityData,
                                   config: NEGASYSConfig) -> str:
    """
    Export optimization results to CSV format.
    
    Format: One row per product with attributes, margins, shares, etc.
    """
    rows = []
    attr_names = household_data.attribute_names or \
                [f"attr_{k+1}" for k in range(household_data.num_attributes)]
    
    for i, product in enumerate(result.best_line.products):
        row = {
            'product_id': i + 1,
        }
        
        # Attribute levels (1-indexed)
        for k in range(household_data.num_attributes):
            row[attr_names[k]] = product.attributes[k] + 1
        
        # Margins and shares
        row['cost'] = product.cost
        row['price'] = product.price
        row['margin'] = product.margin if product.margin != 0 else config.default_margin
        
        if result.best_line.share_per_product:
            row['market_share_pct'] = round(result.best_line.share_per_product[i], 2)
        
        rows.append(row)
    
    # Add summary row
    summary = {
        'product_id': 'TOTAL',
        'market_share_pct': round(result.total_share, 2),
        'cost': '',
        'price': '',
        'margin': '',
    }
    
    # Add blank attribute columns
    for k in range(household_data.num_attributes):
        summary[attr_names[k]] = ''
    
    rows.append(summary)
    
    # Add metrics row
    metrics = {
        'product_id': 'METRICS',
        'market_share_pct': '',
        'cost': f'Profit: ${result.total_profit:.2f}',
        'price': f'Avg U_H: {result.avg_utility_H:.4f}' if result.avg_utility_H else '',
        'margin': f'Avg U_W: {result.avg_utility_W:.4f}' if result.avg_utility_W else '',
    }
    for k in range(household_data.num_attributes):
        metrics[attr_names[k]] = ''
    
    rows.append(metrics)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# =============================================================================
# BATCH COMPARISON OF DECISION RULES
# =============================================================================

@dataclass
class RuleComparisonResult:
    """Results from comparing multiple decision rules."""
    rule_name: str
    aggregation_rule: AggregationRule
    total_share: float
    total_profit: float
    avg_utility_H: Optional[float]
    avg_utility_W: Optional[float]
    utility_dispersion: Optional[float]
    fairness_score: Optional[float]
    product_line: List[List[int]]  # Attribute vectors
    share_per_product: List[float]
    generations: int
    elapsed_seconds: float


def run_all_decision_rules(household_data: HouseholdUtilityData,
                          base_config: NEGASYSConfig,
                          ga_params: GAParams,
                          progress_callback=None) -> List[RuleComparisonResult]:
    """
    Run optimization for all 4 decision rules and return comparison.
    
    Rules tested:
    1. Nash Product
    2. Generalized Nash (Roth)
    3. Linear Weighted
    4. Min (Rawlsian)
    """
    rules = [
        ("Nash Product", AggregationRule.NASH),
        ("Generalized Nash (Roth)", AggregationRule.ROTH),
        ("Linear Weighted", AggregationRule.LINEAR),
        ("Min (Rawlsian)", AggregationRule.MIN),
    ]
    
    results = []
    
    for rule_name, rule_enum in rules:
        if progress_callback:
            progress_callback(f"Running {rule_name}...")
        
        # Create config for this rule
        config = NEGASYSConfig(
            aggregation_rule=rule_enum,
            transformation=base_config.transformation,
            epsilon=base_config.epsilon,
            num_products=base_config.num_products,
            market_size=base_config.market_size,
            default_margin=base_config.default_margin,
            objective=base_config.objective,
            fairness_weight=base_config.fairness_weight
        )
        
        # Run optimization
        opt_result = run_negasys_ga(household_data, config, ga_params)
        
        # Extract product line definition
        product_line = [p.attributes.tolist() for p in opt_result.best_line.products]
        
        # Compute dispersion
        dispersion = None
        if opt_result.avg_utility_H is not None and opt_result.avg_utility_W is not None:
            dispersion = abs(opt_result.avg_utility_H - opt_result.avg_utility_W)
        
        comparison = RuleComparisonResult(
            rule_name=rule_name,
            aggregation_rule=rule_enum,
            total_share=opt_result.total_share,
            total_profit=opt_result.total_profit,
            avg_utility_H=opt_result.avg_utility_H,
            avg_utility_W=opt_result.avg_utility_W,
            utility_dispersion=dispersion,
            fairness_score=opt_result.best_line.fairness_score,
            product_line=product_line,
            share_per_product=opt_result.best_line.share_per_product or [],
            generations=opt_result.generations,
            elapsed_seconds=opt_result.elapsed_seconds
        )
        results.append(comparison)
    
    return results


def export_rule_comparison_csv(comparisons: List[RuleComparisonResult],
                               household_data: HouseholdUtilityData) -> str:
    """Export rule comparison results to CSV."""
    rows = []
    
    attr_names = household_data.attribute_names or \
                [f"attr_{k+1}" for k in range(household_data.num_attributes)]
    
    for comp in comparisons:
        row = {
            'decision_rule': comp.rule_name,
            'total_market_share_pct': round(comp.total_share, 2),
            'total_profit': round(comp.total_profit, 2),
            'avg_utility_H': round(comp.avg_utility_H, 4) if comp.avg_utility_H else None,
            'avg_utility_W': round(comp.avg_utility_W, 4) if comp.avg_utility_W else None,
            'utility_dispersion': round(comp.utility_dispersion, 4) if comp.utility_dispersion else None,
            'fairness_score': round(comp.fairness_score, 4) if comp.fairness_score else None,
            'generations': comp.generations,
            'time_seconds': round(comp.elapsed_seconds, 2),
        }
        
        # Add product line definition
        for p_idx, product_attrs in enumerate(comp.product_line):
            row[f'product_{p_idx+1}_attrs'] = str(product_attrs)
            if p_idx < len(comp.share_per_product):
                row[f'product_{p_idx+1}_share_pct'] = round(comp.share_per_product[p_idx], 2)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

@dataclass
class NEGASYSPipelineConfig:
    """Configuration for the full NEGASYS pipeline."""
    # Data generation
    num_households: int = 100
    num_attributes: int = 4
    attribute_levels: List[int] = None
    preference_correlation: float = 0.4
    alpha_mean: float = 0.5
    alpha_std: float = 0.15
    
    # Stage 0 settings
    num_competitors: int = 4
    competitor_profiles: Optional[List[List[int]]] = None
    status_quo_mode: StatusQuoMode = StatusQuoMode.NASH_ARGMAX
    
    # Optimization settings
    aggregation_rule: AggregationRule = AggregationRule.ROTH
    objective: ObjectiveType = ObjectiveType.MARKET_SHARE
    num_products: int = 2
    market_size: float = 1000.0
    default_margin: float = 1.0
    fairness_weight: float = 0.0
    
    # GA settings
    population_size: int = 100
    max_generations: int = 300
    mutation_rate: float = 0.1
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.attribute_levels is None:
            self.attribute_levels = [3] * self.num_attributes


@dataclass
class NEGASYSPipelineResult:
    """Complete results from NEGASYS pipeline."""
    # Data
    household_data: HouseholdUtilityData
    
    # Stage 0
    stage0_result: Stage0Result
    
    # Optimization
    optimization_result: OptimizationResult
    
    # Exports
    household_csv: str
    results_csv: str
    
    # Optional comparison
    rule_comparison: Optional[List[RuleComparisonResult]] = None
    comparison_csv: Optional[str] = None
    
    # Timing
    total_elapsed_seconds: float = 0.0


def run_negasys_pipeline(config: NEGASYSPipelineConfig,
                        run_rule_comparison: bool = False,
                        progress_callback=None) -> NEGASYSPipelineResult:
    """
    Run the complete NEGASYS pipeline.
    
    Steps:
    1. Generate synthetic household data
    2. Generate or use provided competitor products
    3. Assign status quo (Stage 0)
    4. Run optimization
    5. Optionally run comparison across all decision rules
    6. Generate export CSVs
    """
    start_time = time.time()
    
    # Step 1: Generate household data
    if progress_callback:
        progress_callback("Generating household data...")
    
    household_data = generate_synthetic_household_data(
        num_households=config.num_households,
        num_attributes=config.num_attributes,
        attribute_levels=config.attribute_levels,
        preference_correlation=config.preference_correlation,
        alpha_mean=config.alpha_mean,
        alpha_std=config.alpha_std,
        seed=config.seed
    )
    
    # Step 2: Create competitor products
    if progress_callback:
        progress_callback("Creating competitor products...")
    
    if config.competitor_profiles:
        competitors = create_competitor_set(
            config.num_attributes,
            config.attribute_levels,
            config.competitor_profiles
        )
    else:
        competitors = generate_random_competitors(
            config.num_competitors,
            config.num_attributes,
            config.attribute_levels,
            seed=config.seed + 1000
        )
    
    # Step 3: Stage 0 - Assign status quo
    if progress_callback:
        progress_callback(f"Stage 0: Assigning status quo ({config.status_quo_mode.name})...")
    
    negasys_config = NEGASYSConfig(
        aggregation_rule=config.aggregation_rule,
        objective=config.objective,
        num_products=config.num_products,
        market_size=config.market_size,
        default_margin=config.default_margin,
        fairness_weight=config.fairness_weight
    )
    
    stage0_result = assign_status_quo(
        households=household_data.households,
        competitor_products=competitors,
        config=negasys_config,
        mode=config.status_quo_mode,
        seed=config.seed
    )
    
    # Step 4: Run optimization
    if progress_callback:
        progress_callback("Running GA optimization...")
    
    ga_params = GAParams(
        population_size=config.population_size,
        max_generations=config.max_generations,
        mutation_rate=config.mutation_rate,
        seed=config.seed
    )
    
    optimization_result = run_negasys_ga(
        household_data, negasys_config, ga_params,
        progress_callback=lambda g, f: progress_callback(f"Gen {g}: {f:.2f}") if progress_callback and g % 50 == 0 else None
    )
    
    # Step 5: Optional rule comparison
    rule_comparison = None
    comparison_csv = None
    
    if run_rule_comparison:
        if progress_callback:
            progress_callback("Running comparison across all decision rules...")
        
        rule_comparison = run_all_decision_rules(
            household_data, negasys_config, ga_params,
            progress_callback=progress_callback
        )
        comparison_csv = export_rule_comparison_csv(rule_comparison, household_data)
    
    # Step 6: Generate exports
    if progress_callback:
        progress_callback("Generating export files...")
    
    household_csv = export_household_data_csv(household_data)
    results_csv = export_optimization_results_csv(
        optimization_result, household_data, negasys_config
    )
    
    total_elapsed = time.time() - start_time
    
    return NEGASYSPipelineResult(
        household_data=household_data,
        stage0_result=stage0_result,
        optimization_result=optimization_result,
        household_csv=household_csv,
        results_csv=results_csv,
        rule_comparison=rule_comparison,
        comparison_csv=comparison_csv,
        total_elapsed_seconds=total_elapsed
    )


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def print_results(result: NEGASYSPipelineResult):
    """Print summary of pipeline results."""
    print("\n" + "=" * 60)
    print("NEGASYS PIPELINE RESULTS")
    print("=" * 60)
    
    print(f"\n[Data Generation]")
    print(f"  Households: {result.household_data.num_households}")
    print(f"  Attributes: {result.household_data.num_attributes}")
    print(f"  Levels: {result.household_data.attribute_levels}")
    
    print(f"\n[Stage 0: Status Quo Assignment]")
    print(f"  Mode: {result.stage0_result.mode}")
    print(f"  Competitors: {result.stage0_result.num_competitors}")
    print(f"  Distribution: {result.stage0_result.product_shares}")
    
    print(f"\n[Optimization Results]")
    print(f"  Objective: {result.optimization_result.objective_name}")
    print(f"  Market Share: {result.optimization_result.total_share:.1f}%")
    print(f"  Profit: ${result.optimization_result.total_profit:,.2f}")
    print(f"  Generations: {result.optimization_result.generations}")
    print(f"  Time: {result.optimization_result.elapsed_seconds:.2f}s")
    
    print(f"\n[Optimal Product Line]")
    for i, product in enumerate(result.optimization_result.best_line.products):
        share = result.optimization_result.best_line.share_per_product[i] \
                if result.optimization_result.best_line.share_per_product else 0
        print(f"  Product {i+1}: {product.attributes.tolist()} -> {share:.1f}%")
    
    if result.rule_comparison:
        print(f"\n[Decision Rule Comparison]")
        print("-" * 60)
        for comp in result.rule_comparison:
            print(f"  {comp.rule_name}:")
            print(f"    Share: {comp.total_share:.1f}%, Profit: ${comp.total_profit:.2f}")
    
    print(f"\n[Total Pipeline Time: {result.total_elapsed_seconds:.2f}s]")
    print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("NEGASYS Main Orchestrator v4.0")
    print("=" * 60)
    
    # Configure pipeline
    config = NEGASYSPipelineConfig(
        num_households=100,
        num_attributes=4,
        attribute_levels=[3, 4, 3, 2],
        preference_correlation=0.4,
        alpha_mean=0.5,
        num_competitors=4,
        status_quo_mode=StatusQuoMode.NASH_ARGMAX,
        aggregation_rule=AggregationRule.ROTH,
        objective=ObjectiveType.MARKET_SHARE,
        num_products=2,
        population_size=50,
        max_generations=200,
        seed=42
    )
    
    # Run pipeline with rule comparison
    result = run_negasys_pipeline(
        config,
        run_rule_comparison=True,
        progress_callback=lambda msg: print(f"  {msg}")
    )
    
    # Print results
    print_results(result)
    
    # Save exports
    with open("household_data.csv", "w") as f:
        f.write(result.household_csv)
    print("\nSaved: household_data.csv")
    
    with open("optimization_results.csv", "w") as f:
        f.write(result.results_csv)
    print("Saved: optimization_results.csv")
    
    if result.comparison_csv:
        with open("rule_comparison.csv", "w") as f:
            f.write(result.comparison_csv)
        print("Saved: rule_comparison.csv")
    
    print("\nPipeline complete!")
