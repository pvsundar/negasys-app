"""
NEGASYS: Negotiated Household Product Line Design Optimizer
Core Module - Version 3.0 (GA fixes)

Authors: P.V. (Sundar) Balakrishnan
Version: 3.0
Date: December 2025

FIXES in v3.0:
- Fixed GA convergence issues that caused static 55% market share
- Improved fitness computation and selection pressure
- Better margin handling for profit vs market share objectives
- Enhanced progress tracking and convergence detection
- Cleaned up objective handling throughout the pipeline

A decision support system for optimal product line design with dyadic household
decision-making. Extends PRODLINE to model households where two decision-makers
(e.g., H and W) jointly select products based on cooperative bargaining rules.

Supported Aggregation Rules:
- Nash Product: U_H Ã— U_W
- Generalized Nash (Roth): U_H^Î± Ã— U_W^(1-Î±)
- Linear Weighted: Î±Â·U_H + (1-Î±)Â·U_W
- Min (Rawlsian): min{U_H, U_W}

References:
- Nash, J. (1950). The Bargaining Problem. Econometrica, 18(2), 155-162.
- Roth, A.E. (1979). Axiomatic Models of Bargaining. Springer-Verlag.
- Balakrishnan & Jacob (1996). Management Science, 42(8), 1105-1117.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict, Any
from enum import IntEnum
import random
from abc import ABC, abstractmethod


# =============================================================================
# CONSTANTS
# =============================================================================

class AggregationRule(IntEnum):
    """Household utility aggregation rules."""
    NASH = 1           # Pure Nash product: U_H Ã— U_W
    ROTH = 2           # Generalized Nash: U_H^Î± Ã— U_W^(1-Î±)
    LINEAR = 3         # Linear weighted: Î±Â·U_H + (1-Î±)Â·U_W
    MIN = 4            # Rawlsian: min{U_H, U_W}
    MAX = 5            # Utilitarian max: max{U_H, U_W}
    CES = 6            # CES: (Î±Â·U_H^Ï + (1-Î±)Â·U_W^Ï)^(1/Ï)


class TransformationType(IntEnum):
    """Utility transformation methods."""
    NONE = 0           # Raw utilities (for linear rule)
    GAINS = 1          # Gains over status quo
    NORMALIZED = 2     # Normalized to (0, 1]
    SHIFTED = 3        # Shifted to positive


class ObjectiveType(IntEnum):
    """Optimization objective types."""
    PROFIT = 1         # Maximize profit (requires margins)
    MARKET_SHARE = 2   # Maximize market share
    WEIGHTED = 3       # Weighted combination of profit and fairness


class InitializationType(IntEnum):
    """GA population initialization strategies."""
    RANDOM = 1
    DIVERSIFIED = 2
    SEEDED = 3


class SelectionType(IntEnum):
    """GA selection methods."""
    TOURNAMENT = 1
    ROULETTE = 2
    RANK = 3
    DETERMINISTIC = 4


class CrossoverType(IntEnum):
    """GA crossover methods."""
    SINGLE_POINT = 1
    TWO_POINT = 2
    UNIFORM = 3
    PRODUCT_SWAP = 4


class PopulationMaintenance(IntEnum):
    """GA population maintenance strategies."""
    GENERATIONAL = 1
    STEADY_STATE = 2
    ELITIST = 3


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ProductProfile:
    """A single product defined by attribute levels."""
    attributes: np.ndarray           # Shape: (num_attributes,), values are level indices (0-based)
    cost: float = 0.0
    price: float = 0.0
    margin: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.attributes, list):
            self.attributes = np.array(self.attributes, dtype=np.int32)
        if self.price > 0 and self.cost >= 0:
            self.margin = self.price - self.cost
    
    def copy(self) -> 'ProductProfile':
        return ProductProfile(
            attributes=self.attributes.copy(),
            cost=self.cost,
            price=self.price,
            margin=self.margin
        )
    
    def __hash__(self):
        return hash(tuple(self.attributes))
    
    def __eq__(self, other):
        if not isinstance(other, ProductProfile):
            return False
        return np.array_equal(self.attributes, other.attributes)


@dataclass
class Household:
    """A household with two decision-makers."""
    id: int
    partworths_H: np.ndarray         # Shape: (num_attributes, max_levels)
    partworths_W: np.ndarray         # Shape: (num_attributes, max_levels)
    status_quo: ProductProfile       # Current product owned
    alpha: float = 0.5               # Bargaining power (0 = W dominates, 1 = H dominates)
    weight: float = 1.0              # Household weight in market
    
    # Optional covariates for alpha modeling
    income_share_H: Optional[float] = None
    expertise_H: Optional[float] = None
    expertise_W: Optional[float] = None


@dataclass
class HouseholdUtilityData:
    """Container for all household utility data."""
    num_households: int
    num_attributes: int
    attribute_levels: List[int]      # Number of levels per attribute
    households: List[Household]
    
    # Optional metadata
    attribute_names: Optional[List[str]] = None
    level_names: Optional[List[List[str]]] = None
    
    @property
    def total_levels(self) -> int:
        return sum(self.attribute_levels)


@dataclass
class ProductLine:
    """A product line (set of products) being evaluated."""
    products: List[ProductProfile]
    fitness: float = 0.0
    market_share: float = 0.0
    profit: float = 0.0
    share_per_product: Optional[List[float]] = None
    utility_H_avg: Optional[float] = None
    utility_W_avg: Optional[float] = None
    fairness_score: Optional[float] = None
    
    def copy(self) -> 'ProductLine':
        return ProductLine(
            products=[p.copy() for p in self.products],
            fitness=self.fitness,
            market_share=self.market_share,
            profit=self.profit,
            share_per_product=self.share_per_product.copy() if self.share_per_product else None,
            utility_H_avg=self.utility_H_avg,
            utility_W_avg=self.utility_W_avg,
            fairness_score=self.fairness_score
        )
    
    @property
    def num_products(self) -> int:
        return len(self.products)


@dataclass
class NEGASYSConfig:
    """Configuration for NEGASYS optimization."""
    # Aggregation settings
    aggregation_rule: AggregationRule = AggregationRule.ROTH
    transformation: TransformationType = TransformationType.NORMALIZED
    epsilon: float = 0.01            # Smoothing constant for normalization
    
    # CES parameter (only used if aggregation_rule == CES)
    ces_rho: float = 0.5
    
    # Product line settings
    num_products: int = 2
    
    # Market settings
    market_size: float = 1000.0
    
    # Cost/Price settings
    cost_function: Optional[Callable[[ProductProfile], float]] = None
    price_function: Optional[Callable[[ProductProfile], float]] = None
    default_margin: float = 1.0      # Default margin when no cost/price functions
    
    # Objective settings
    objective: ObjectiveType = ObjectiveType.MARKET_SHARE  # What to optimize
    
    # Fairness settings
    fairness_weight: float = 0.0     # Weight on fairness objective (0 = pure profit/share)
    fairness_threshold: Optional[float] = None  # Constraint on |U_H - U_W|


@dataclass
class GAParams:
    """Genetic Algorithm parameters."""
    population_size: int = 100
    max_generations: int = 500
    
    # Initialization
    init_type: InitializationType = InitializationType.DIVERSIFIED
    
    # Selection
    selection_type: SelectionType = SelectionType.TOURNAMENT
    tournament_size: int = 3
    
    # Crossover
    crossover_type: CrossoverType = CrossoverType.TWO_POINT
    crossover_rate: float = 0.8
    
    # Mutation
    mutation_rate: float = 0.05
    
    # Population maintenance
    maintenance_type: PopulationMaintenance = PopulationMaintenance.ELITIST
    elitism_count: int = 2
    
    # Convergence
    convergence_window: int = 20      # Reduced for faster convergence detection
    convergence_threshold: float = 0.0001  # Tighter threshold
    
    # Random seed
    seed: Optional[int] = None


@dataclass
class DPParams:
    """Dynamic Programming parameters."""
    max_products: int = 4
    beam_width: int = 100            # For beam search approximation
    use_exact: bool = True           # Use exact DP if problem is small enough
    exact_threshold: int = 20        # Max products for exact enumeration


@dataclass
class OptimizationResult:
    """Results from NEGASYS optimization."""
    best_line: ProductLine
    solver_type: str
    generations: int
    total_share: float
    total_profit: float
    convergence_history: Optional[List[float]] = None
    
    # Track what metric was optimized
    objective_name: str = "fitness"
    
    # Household-level results
    household_choices: Optional[np.ndarray] = None  # Which product each household chose
    utility_distribution_H: Optional[np.ndarray] = None
    utility_distribution_W: Optional[np.ndarray] = None
    
    # Fairness metrics
    avg_utility_H: Optional[float] = None
    avg_utility_W: Optional[float] = None
    utility_dispersion: Optional[float] = None
    
    # Computation time
    elapsed_seconds: float = 0.0


# =============================================================================
# UTILITY COMPUTATION
# =============================================================================

class UtilityEngine:
    """Engine for computing individual and household utilities."""
    
    def __init__(self, config: NEGASYSConfig):
        self.config = config
    
    def compute_raw_utility(
        self,
        household: Household,
        product: ProductProfile,
        individual: str  # 'H' or 'W'
    ) -> float:
        """Compute additive part-worth utility for an individual."""
        partworths = household.partworths_H if individual == 'H' else household.partworths_W
        
        utility = 0.0
        for k, level in enumerate(product.attributes):
            utility += partworths[k, level]
        
        return utility
    
    def compute_utility_gains(
        self,
        household: Household,
        product: ProductProfile,
        individual: str
    ) -> float:
        """Compute utility gain relative to status quo."""
        U_product = self.compute_raw_utility(household, product, individual)
        U_sq = self.compute_raw_utility(household, household.status_quo, individual)
        return U_product - U_sq
    
    def compute_transformed_utility(
        self,
        household: Household,
        product: ProductProfile,
        choice_set: List[ProductProfile],
        individual: str
    ) -> float:
        """Compute Nash-compatible transformed utility in (0, 1]."""
        eps = self.config.epsilon
        
        # Compute all gains over status quo
        gains = []
        for p in choice_set:
            gain = self.compute_utility_gains(household, p, individual)
            gains.append(gain)
        
        # Also include status quo itself (which has gain = 0)
        gains.append(0.0)
        
        min_gain = min(gains)
        max_gain = max(gains)
        
        # Focal product gain
        focal_gain = self.compute_utility_gains(household, product, individual)
        
        # Transform to (0, 1]
        if max_gain - min_gain < eps:
            # All utilities are essentially equal
            return 0.5
        
        v = (focal_gain - min_gain + eps) / (max_gain - min_gain + eps)
        return max(v, eps)  # Ensure strictly positive
    
    def compute_household_utility(
        self,
        household: Household,
        product: ProductProfile,
        choice_set: List[ProductProfile]
    ) -> Tuple[float, float, float]:
        """
        Compute aggregated household utility.
        
        Returns:
            (aggregated_utility, v_H, v_W)
        """
        # Get transformed utilities
        if self.config.transformation == TransformationType.NORMALIZED:
            v_H = self.compute_transformed_utility(household, product, choice_set, 'H')
            v_W = self.compute_transformed_utility(household, product, choice_set, 'W')
        elif self.config.transformation == TransformationType.GAINS:
            v_H = self.compute_utility_gains(household, product, 'H')
            v_W = self.compute_utility_gains(household, product, 'W')
            # Ensure positive for multiplicative rules
            if self.config.aggregation_rule in [AggregationRule.NASH, AggregationRule.ROTH]:
                v_H = max(v_H, self.config.epsilon)
                v_W = max(v_W, self.config.epsilon)
        else:
            v_H = self.compute_raw_utility(household, product, 'H')
            v_W = self.compute_raw_utility(household, product, 'W')
        
        alpha = household.alpha
        
        # Apply aggregation rule
        if self.config.aggregation_rule == AggregationRule.NASH:
            A = v_H * v_W
        
        elif self.config.aggregation_rule == AggregationRule.ROTH:
            # Handle edge cases for power function
            if v_H <= 0 or v_W <= 0:
                A = self.config.epsilon * self.config.epsilon
            else:
                try:
                    A = (v_H ** alpha) * (v_W ** (1 - alpha))
                except (OverflowError, ValueError):
                    A = self.config.epsilon
        
        elif self.config.aggregation_rule == AggregationRule.LINEAR:
            A = alpha * v_H + (1 - alpha) * v_W
        
        elif self.config.aggregation_rule == AggregationRule.MIN:
            A = min(v_H, v_W)
        
        elif self.config.aggregation_rule == AggregationRule.MAX:
            A = max(v_H, v_W)
        
        elif self.config.aggregation_rule == AggregationRule.CES:
            rho = self.config.ces_rho
            if abs(rho) < 1e-6:  # rho â‰ˆ 0
                # Cobb-Douglas (same as Nash with Î±=0.5)
                try:
                    A = (v_H ** alpha) * (v_W ** (1 - alpha))
                except (OverflowError, ValueError):
                    A = self.config.epsilon
            else:
                try:
                    A = (alpha * (v_H ** rho) + (1 - alpha) * (v_W ** rho)) ** (1 / rho)
                except (OverflowError, ValueError):
                    A = self.config.epsilon
        
        else:
            raise ValueError(f"Unknown aggregation rule: {self.config.aggregation_rule}")
        
        return A, v_H, v_W


# =============================================================================
# MARKET SHARE & PROFIT EVALUATION
# =============================================================================

class MarketEvaluator:
    """Evaluates product lines for market share and profit."""
    
    def __init__(
        self,
        household_data: HouseholdUtilityData,
        config: NEGASYSConfig
    ):
        self.household_data = household_data
        self.config = config
        self.utility_engine = UtilityEngine(config)
    
    def evaluate_product_line(
        self,
        line: ProductLine,
        compute_details: bool = False
    ) -> ProductLine:
        """
        Evaluate a product line for market share and profit.
        
        Args:
            line: ProductLine to evaluate
            compute_details: If True, compute detailed household-level results
        
        Returns:
            ProductLine with fitness, shares, and profit populated
        """
        num_products = len(line.products)
        num_households = self.household_data.num_households
        
        # Initialize counters
        product_counts = np.zeros(num_products + 1)  # Last slot for status quo
        total_weighted = 0.0
        
        # Track utilities for fairness computation
        utility_H_sum = 0.0
        utility_W_sum = 0.0
        
        # Optional detailed tracking
        if compute_details:
            household_choices = np.zeros(num_households, dtype=np.int32)
            utility_H_all = np.zeros(num_households)
            utility_W_all = np.zeros(num_households)
        
        # Evaluate each household
        for h_idx, household in enumerate(self.household_data.households):
            # Build choice set
            choice_set = line.products + [household.status_quo]
            
            # Find best option
            best_utility = -np.inf
            best_choice = num_products  # Default to status quo
            best_v_H = 0.0
            best_v_W = 0.0
            
            for p_idx, product in enumerate(line.products):
                A, v_H, v_W = self.utility_engine.compute_household_utility(
                    household, product, choice_set
                )
                if A > best_utility:
                    best_utility = A
                    best_choice = p_idx
                    best_v_H = v_H
                    best_v_W = v_W
            
            # Compare to status quo
            A_sq, v_H_sq, v_W_sq = self.utility_engine.compute_household_utility(
                household, household.status_quo, choice_set
            )
            if A_sq >= best_utility:
                best_choice = num_products  # Status quo
                best_v_H = v_H_sq
                best_v_W = v_W_sq
            
            # Update counters
            product_counts[best_choice] += household.weight
            total_weighted += household.weight
            
            if best_choice < num_products:  # Purchased from line
                utility_H_sum += best_v_H * household.weight
                utility_W_sum += best_v_W * household.weight
            
            if compute_details:
                household_choices[h_idx] = best_choice
                utility_H_all[h_idx] = best_v_H
                utility_W_all[h_idx] = best_v_W
        
        # Compute market shares
        shares = product_counts[:num_products] / total_weighted if total_weighted > 0 else np.zeros(num_products)
        sq_share = product_counts[num_products] / total_weighted if total_weighted > 0 else 0
        total_share = shares.sum()
        
        # Compute profit - use default_margin if no margin set
        profit = 0.0
        for p_idx, product in enumerate(line.products):
            margin = product.margin if abs(product.margin) > 1e-6 else self.config.default_margin
            profit += self.config.market_size * margin * shares[p_idx]
        
        # Compute fairness metrics
        buyers_weight = total_weighted * total_share
        if buyers_weight > 1e-6:
            avg_H = utility_H_sum / buyers_weight
            avg_W = utility_W_sum / buyers_weight
            fairness = 1.0 - abs(avg_H - avg_W)  # Higher is better
        else:
            avg_H = 0.0
            avg_W = 0.0
            fairness = 1.0
        
        # Compute fitness based on objective type
        if self.config.objective == ObjectiveType.PROFIT:
            base_fitness = profit
        elif self.config.objective == ObjectiveType.MARKET_SHARE:
            base_fitness = total_share * 100  # As percentage for scale
        else:  # WEIGHTED
            base_fitness = profit
        
        # Apply fairness weight
        fitness = base_fitness * (1 - self.config.fairness_weight) + \
                  fairness * self.config.fairness_weight * self.config.market_size
        
        # Update line
        line.fitness = fitness
        line.market_share = total_share * 100  # As percentage
        line.profit = profit
        line.share_per_product = (shares * 100).tolist()
        line.utility_H_avg = avg_H
        line.utility_W_avg = avg_W
        line.fairness_score = fairness
        
        return line


# =============================================================================
# GENETIC ALGORITHM OPTIMIZER
# =============================================================================

class NEGASYS_GA:
    """Genetic Algorithm optimizer for NEGASYS."""
    
    def __init__(
        self,
        household_data: HouseholdUtilityData,
        config: NEGASYSConfig,
        ga_params: GAParams
    ):
        self.household_data = household_data
        self.config = config
        self.params = ga_params
        self.evaluator = MarketEvaluator(household_data, config)
        
        # Set random seed
        if ga_params.seed is not None:
            random.seed(ga_params.seed)
            np.random.seed(ga_params.seed)
        
        # Precompute attribute info
        self.num_attributes = household_data.num_attributes
        self.attribute_levels = household_data.attribute_levels
    
    def run(
        self,
        progress_callback: Optional[Callable[[int, float], None]] = None
    ) -> OptimizationResult:
        """Run the genetic algorithm optimization."""
        import time
        start_time = time.time()
        
        # Initialize population with better diversity
        population = self._initialize_population()
        
        # Evaluate initial population
        for individual in population:
            self.evaluator.evaluate_product_line(individual)
        
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        best_line = population[0].copy()
        convergence_history = [best_line.fitness]
        
        # Track improvement for early stopping
        last_improvement_gen = 0
        
        # Main evolution loop
        for gen in range(1, self.params.max_generations + 1):
            new_population = []
            
            # Elitism - ensure we keep the best
            elite_count = min(self.params.elitism_count, len(population))
            for i in range(elite_count):
                new_population.append(population[i].copy())
            
            # Generate offspring
            while len(new_population) < self.params.population_size:
                # Selection
                parent1 = self._select(population)
                parent2 = self._select(population)
                
                # Crossover with higher probability for diversity
                if random.random() < self.params.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation - ensure it actually changes something
                self._mutate(child1)
                if len(new_population) + 1 < self.params.population_size:
                    self._mutate(child2)
                
                # Evaluate
                self.evaluator.evaluate_product_line(child1)
                new_population.append(child1)
                
                if len(new_population) < self.params.population_size:
                    self.evaluator.evaluate_product_line(child2)
                    new_population.append(child2)
            
            # Replace population
            population = new_population[:self.params.population_size]
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best
            if population[0].fitness > best_line.fitness + 1e-8:  # Require meaningful improvement
                best_line = population[0].copy()
                last_improvement_gen = gen
            
            convergence_history.append(best_line.fitness)
            
            # Progress callback
            if progress_callback:
                progress_callback(gen, best_line.fitness)
            
            # Check convergence
            if self._check_convergence(convergence_history) or \
               (gen - last_improvement_gen > 100):  # No improvement for 100 gens
                break
        
        elapsed = time.time() - start_time
        
        # Final detailed evaluation
        self.evaluator.evaluate_product_line(best_line, compute_details=True)
        
        # Determine objective name for reporting
        if self.config.objective == ObjectiveType.PROFIT:
            objective_name = "Profit ($)"
        elif self.config.objective == ObjectiveType.MARKET_SHARE:
            objective_name = "Market Share (%)"
        else:
            objective_name = "Weighted Objective"
        
        return OptimizationResult(
            best_line=best_line,
            solver_type="GA",
            generations=len(convergence_history) - 1,
            total_share=best_line.market_share,
            total_profit=best_line.profit,
            convergence_history=convergence_history,
            objective_name=objective_name,
            avg_utility_H=best_line.utility_H_avg,
            avg_utility_W=best_line.utility_W_avg,
            elapsed_seconds=elapsed
        )
    
    def _initialize_population(self) -> List[ProductLine]:
        """Initialize GA population with better diversity."""
        population = []
        used_combinations = set()
        
        # First, try to create truly diverse population
        max_attempts = self.params.population_size * 10
        attempts = 0
        
        while len(population) < self.params.population_size and attempts < max_attempts:
            products = []
            line_signature = []
            
            for _ in range(self.config.num_products):
                product = self._random_product()
                products.append(product)
                line_signature.append(tuple(product.attributes))
            
            # Check if this combination is new
            combo_key = tuple(sorted(line_signature))
            if combo_key not in used_combinations:
                used_combinations.add(combo_key)
                population.append(ProductLine(products=products))
            
            attempts += 1
        
        # Fill remaining slots with random products if needed
        while len(population) < self.params.population_size:
            products = [self._random_product() for _ in range(self.config.num_products)]
            population.append(ProductLine(products=products))
        
        return population
    
    def _random_product(self) -> ProductProfile:
        """Generate a random product profile with proper margin."""
        attributes = np.array([
            random.randint(0, self.attribute_levels[k] - 1)
            for k in range(self.num_attributes)
        ], dtype=np.int32)
        
        product = ProductProfile(attributes=attributes)
        
        # Apply cost/price functions if provided, otherwise use default margin
        if self.config.cost_function:
            product.cost = self.config.cost_function(product)
        if self.config.price_function:
            product.price = self.config.price_function(product)
        
        # Ensure margin is set
        if abs(product.margin) < 1e-6:
            product.margin = self.config.default_margin
        
        return product
    
    def _select(self, population: List[ProductLine]) -> ProductLine:
        """Select a parent using tournament selection with better diversity."""
        if self.params.selection_type == SelectionType.TOURNAMENT:
            tournament_size = min(self.params.tournament_size, len(population))
            candidates = random.sample(population, tournament_size)
            
            # Add some randomness to prevent premature convergence
            if random.random() < 0.1:  # 10% chance to select randomly
                return random.choice(candidates).copy()
            else:
                return max(candidates, key=lambda x: x.fitness).copy()
        
        # Other selection methods...
        elif self.params.selection_type == SelectionType.ROULETTE:
            # Fitness-proportional selection with minimum threshold
            fitnesses = [max(ind.fitness, 1e-6) for ind in population]
            total_fitness = sum(fitnesses)
            
            if total_fitness <= 0:
                return random.choice(population).copy()
            
            pick = random.random() * total_fitness
            current = 0
            for ind, fitness in zip(population, fitnesses):
                current += fitness
                if current >= pick:
                    return ind.copy()
            return population[-1].copy()
        
        else:  # Default to first (best)
            return population[0].copy()
    
    def _crossover(
        self,
        parent1: ProductLine,
        parent2: ProductLine
    ) -> Tuple[ProductLine, ProductLine]:
        """Perform crossover between two parents with enhanced variation."""
        n = self.config.num_products
        
        # Use uniform crossover for better mixing
        child1_products = []
        child2_products = []
        
        for i in range(n):
            if random.random() < 0.5:
                child1_products.append(parent1.products[i].copy())
                child2_products.append(parent2.products[i].copy())
            else:
                child1_products.append(parent2.products[i].copy())
                child2_products.append(parent1.products[i].copy())
        
        # Also mix attributes within products occasionally
        for i in range(n):
            if random.random() < 0.3:  # 30% chance for attribute-level crossover
                for k in range(self.num_attributes):
                    if random.random() < 0.5:
                        # Swap this attribute between children
                        temp = child1_products[i].attributes[k]
                        child1_products[i].attributes[k] = child2_products[i].attributes[k]
                        child2_products[i].attributes[k] = temp
        
        return ProductLine(products=child1_products), ProductLine(products=child2_products)
    
    def _mutate(self, line: ProductLine) -> None:
        """Apply mutation to a product line with guaranteed change."""
        mutation_occurred = False
        
        for product in line.products:
            for k in range(self.num_attributes):
                if random.random() < self.params.mutation_rate:
                    old_level = product.attributes[k]
                    # Ensure mutation actually changes the value
                    possible_levels = [l for l in range(self.attribute_levels[k]) if l != old_level]
                    if possible_levels:
                        product.attributes[k] = random.choice(possible_levels)
                        mutation_occurred = True
            
            # Recompute cost/price/margin
            if self.config.cost_function:
                product.cost = self.config.cost_function(product)
            if self.config.price_function:
                product.price = self.config.price_function(product)
            
            # Ensure margin is set
            if abs(product.margin) < 1e-6:
                product.margin = self.config.default_margin
        
        # If no mutation occurred and mutation rate > 0, force at least one change
        if not mutation_occurred and self.params.mutation_rate > 0:
            product_idx = random.randint(0, len(line.products) - 1)
            attr_idx = random.randint(0, self.num_attributes - 1)
            old_level = line.products[product_idx].attributes[attr_idx]
            possible_levels = [l for l in range(self.attribute_levels[attr_idx]) if l != old_level]
            if possible_levels:
                line.products[product_idx].attributes[attr_idx] = random.choice(possible_levels)
    
    def _check_convergence(self, history: List[float]) -> bool:
        """Check if GA has converged."""
        window = self.params.convergence_window
        if len(history) < window:
            return False
        
        recent = history[-window:]
        range_val = max(recent) - min(recent)
        
        if abs(max(recent)) < 1e-10:
            return True
        
        return range_val < self.params.convergence_threshold * abs(max(recent))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_negasys_ga(
    household_data: HouseholdUtilityData,
    config: NEGASYSConfig,
    ga_params: GAParams,
    progress_callback: Optional[Callable[[int, float], None]] = None
) -> OptimizationResult:
    """Convenience function to run NEGASYS GA optimization."""
    optimizer = NEGASYS_GA(household_data, config, ga_params)
    return optimizer.run(progress_callback)


# =============================================================================
# FILE PARSERS & DATA GENERATION
# =============================================================================

def generate_synthetic_household_data(
    num_households: int,
    num_attributes: int,
    attribute_levels: List[int],
    preference_correlation: float = 0.3,
    alpha_mean: float = 0.5,
    alpha_std: float = 0.15,
    seed: Optional[int] = None
) -> HouseholdUtilityData:
    """
    Generate synthetic household data for testing.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    max_levels = max(attribute_levels)
    households = []
    
    for h_id in range(num_households):
        # Generate H partworths with more variation
        partworths_H = np.zeros((num_attributes, max_levels))
        for k in range(num_attributes):
            # Random partworths with higher variance
            raw = np.random.randn(attribute_levels[k]) * 2.0  # Increased variance
            raw = raw - raw.mean()
            partworths_H[k, :attribute_levels[k]] = raw
        
        # Generate W partworths correlated with H but with variation
        partworths_W = np.zeros((num_attributes, max_levels))
        for k in range(num_attributes):
            noise = np.random.randn(attribute_levels[k]) * 2.0
            noise = noise - noise.mean()
            correlated = preference_correlation * partworths_H[k, :attribute_levels[k]] + \
                        np.sqrt(1 - preference_correlation**2) * noise
            partworths_W[k, :attribute_levels[k]] = correlated
        
        # Generate random status quo
        sq_attrs = np.array([
            np.random.randint(0, attribute_levels[k])
            for k in range(num_attributes)
        ], dtype=np.int32)
        
        # Generate alpha with more spread
        alpha = np.clip(np.random.normal(alpha_mean, alpha_std), 0.05, 0.95)
        
        household = Household(
            id=h_id,
            partworths_H=partworths_H,
            partworths_W=partworths_W,
            status_quo=ProductProfile(attributes=sq_attrs),
            alpha=alpha
        )
        households.append(household)
    
    return HouseholdUtilityData(
        num_households=num_households,
        num_attributes=num_attributes,
        attribute_levels=attribute_levels,
        households=households,
        attribute_names=[f"Attribute_{k+1}" for k in range(num_attributes)]
    )


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    # Quick test with synthetic data
    print("NEGASYS Core Module v3.0 - Test Run")
    print("=" * 50)
    
    # Generate synthetic data
    print("\nGenerating synthetic household data...")
    household_data = generate_synthetic_household_data(
        num_households=100,
        num_attributes=4,
        attribute_levels=[3, 4, 3, 2],
        preference_correlation=0.4,
        seed=7149
    )
    print(f"  Households: {household_data.num_households}")
    print(f"  Attributes: {household_data.num_attributes}")
    print(f"  Levels: {household_data.attribute_levels}")
    
    # Configure NEGASYS for market share optimization
    config = NEGASYSConfig(
        aggregation_rule=AggregationRule.ROTH,
        num_products=3,
        market_size=1000.0,
        objective=ObjectiveType.MARKET_SHARE,
        default_margin=1.0
    )
    
    # Configure GA with better parameters
    ga_params = GAParams(
        population_size=50,
        max_generations=200,
        mutation_rate=0.1,  # Higher mutation for diversity
        convergence_window=20,
        seed=7149
    )
    
    # Run optimization
    print("\nRunning GA optimization...")
    result = run_negasys_ga(
        household_data, config, ga_params,
        progress_callback=lambda g, f: print(f"  Gen {g}: {f:.4f}") if g % 25 == 0 else None
    )
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Objective: {result.objective_name}")
    print(f"Generations: {result.generations}")
    print(f"Market Share: {result.total_share:.1f}%")
    print(f"Profit: ${result.total_profit:.2f}")
    print(f"Time: {result.elapsed_seconds:.2f}s")
    
    print(f"\nOptimal Product Line:")
    for i, product in enumerate(result.best_line.products):
        share = result.best_line.share_per_product[i] if result.best_line.share_per_product else 0
        print(f"  Product {i+1}: {product.attributes.tolist()} -> {share:.1f}%")
    
    # Show convergence pattern
    if result.convergence_history and len(result.convergence_history) > 1:
        print(f"\nConvergence Pattern:")
        print(f"  Start: {result.convergence_history[0]:.2f}")
        print(f"  End: {result.convergence_history[-1]:.2f}")
        print(f"  Improvement: {result.convergence_history[-1] - result.convergence_history[0]:.2f}")
    
    print("\n" + "=" * 50)
    print("Test complete! GA should now show proper evolution.")
