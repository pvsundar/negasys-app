"""
NEGASYS: Negotiated Household Product Line Design Optimizer
Streamlit Web Application - Version 5.1

Author: P.V. (Sundar) Balakrishnan
Version: 5.1
Date: January 2026

New Features in v5.1:
- Built-in SmartTVs 4'Us teaching case data (no file uploads needed)
- Three data source options: Synthetic, SmartTVs Case, Upload Files

Features from v5.0:
- Data source switch: Synthetic or Real Data (TV format)
- Upload utility file (.01b) and competitor file (.01c)
- Automatic household pairing from individual conjoint data
- Fixed deprecation warnings (use_container_width -> width)
- Preserved all v4 features: Stage 0, exports, rule comparison
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Optional, List
import io
import tempfile
from pathlib import Path
import os

# Import from main orchestrator
from negasys_main import (
    StatusQuoMode, Stage0Result,
    create_competitor_set, generate_random_competitors, assign_status_quo,
    export_household_data_csv, export_optimization_results_csv,
    run_all_decision_rules, export_rule_comparison_csv,
    RuleComparisonResult
)

# Import from core
from negasys_core_v3 import (
    AggregationRule, ObjectiveType,
    ProductProfile, Household, HouseholdUtilityData,
    ProductLine, NEGASYSConfig, GAParams, OptimizationResult,
    run_negasys_ga, generate_synthetic_household_data
)

# Import data loader
from negasys_data_loader import (
    load_negasys_data_tv_format,
    load_utility_file_tv_format,
    load_competitor_file_tv_format,
    pair_consumers_to_households,
    describe_loaded_data,
    validate_data_consistency
)

NEGASYS_VERSION = "5.1"

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NEGASYS v5.1: Household Product Line Optimizer",
    page_icon="👫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CASE DATA PATHS
# =============================================================================

def get_case_data_path(filename):
    """Get path to built-in case data files."""
    # When running on Streamlit Cloud, files are relative to app location
    script_dir = Path(__file__).parent
    case_path = script_dir / "examples" / "smarttv_case" / filename
    if case_path.exists():
        return case_path
    # Fallback for local development
    fallback = Path("examples") / "smarttv_case" / filename
    if fallback.exists():
        return fallback
    return case_path  # Return original path for error message

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_rule_description(rule):
    descriptions = {
        AggregationRule.NASH: "**Nash Product**: U_H × U_W — Both partners must benefit equally.",
        AggregationRule.ROTH: "**Generalized Nash (Roth)**: U_H^α × U_W^(1-α) — Asymmetric bargaining power.",
        AggregationRule.LINEAR: "**Linear Weighted**: α·U_H + (1-α)·U_W — Simple weighted average.",
        AggregationRule.MIN: "**Min (Rawlsian)**: min{U_H, U_W} — Focus on least satisfied partner.",
    }
    return descriptions.get(rule, "")


def create_fairness_gauge(avg_H, avg_W):
    """Create gauge showing H vs W utility balance."""
    if avg_H is None or avg_W is None or (avg_H + avg_W) == 0:
        balance = 0.5
    else:
        balance = avg_H / (avg_H + avg_W)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=balance,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "H vs W Balance", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 1], 'tickmode': 'linear', 'dtick': 0.2},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 0.4], 'color': '#ff6b6b'},
                {'range': [0.4, 0.6], 'color': '#6bcb77'},
                {'range': [0.6, 1], 'color': '#ff6b6b'}
            ],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 0.5}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_rule_comparison_chart(comparisons: List[RuleComparisonResult]):
    """Create comparison chart for decision rules."""
    rules = [c.rule_name for c in comparisons]
    shares = [c.total_share for c in comparisons]
    profits = [c.total_profit for c in comparisons]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Market Share by Rule', 'Profit by Rule'))
    
    fig.add_trace(go.Bar(x=rules, y=shares, name='Market Share %', marker_color='#667eea'), row=1, col=1)
    fig.add_trace(go.Bar(x=rules, y=profits, name='Profit ($)', marker_color='#6bcb77'), row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(title_text="Market Share (%)", row=1, col=1)
    fig.update_yaxes(title_text="Profit ($)", row=1, col=2)
    
    return fig


def format_product_profile(attrs: np.ndarray, level_names: Optional[List[List[str]]] = None) -> str:
    """Format product attributes with level names if available."""
    if level_names:
        parts = []
        for k, lv in enumerate(attrs):
            if k < len(level_names) and lv < len(level_names[k]):
                parts.append(level_names[k][lv])
            else:
                parts.append(f"L{lv+1}")
        return " | ".join(parts)
    else:
        return str(attrs.tolist())


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.title("👫 NEGASYS v5.1")
    st.markdown("**Negotiated Household Product Line Design Optimizer**")
    st.markdown("*With SmartTVs Case, Real Data Support, and Multi-Rule Comparison*")
    
    # Initialize session state
    if 'household_data' not in st.session_state:
        st.session_state.household_data = None
    if 'stage0_result' not in st.session_state:
        st.session_state.stage0_result = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'rule_comparison' not in st.session_state:
        st.session_state.rule_comparison = None
    if 'competitors' not in st.session_state:
        st.session_state.competitors = None
    if 'data_metadata' not in st.session_state:
        st.session_state.data_metadata = None
    
    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.header("📂 Data Source")
        
        data_source = st.radio(
            "Select Data Source",
            ["SmartTVs 4'Us Case", "Synthetic Data", "Real Data (Upload Files)"],
            help="SmartTVs: built-in teaching case. Synthetic: generate random data. Real Data: upload your own files."
        )
        
        st.divider()
        
        # ==========================================================================
        # SMARTTVS 4'US CASE DATA
        # ==========================================================================
        if data_source == "SmartTVs 4'Us Case":
            st.header("📺 SmartTVs 4'Us Case")
            
            st.markdown("""
            **Built-in Teaching Case Data**
            
            - 200 individuals → 100 households
            - 6 attributes (Brand, Screen, Audio, Parental, Platform, Price)
            - 5 competitor TVs
            
            *No file uploads required!*
            """)
            
            st.divider()
            
            alpha_default = st.slider("Default Bargaining Power (α)", 0.1, 0.9, 0.5, key="case_alpha")
            seed = st.number_input("Random Seed", 0, 9999, 7149, key="case_seed")
            
            st.divider()
            
            st.subheader("Stage 0 Settings")
            sq_mode = st.radio(
                "Status Quo Assignment",
                ["Random from Competitors", "Nash-Optimal from Competitors"],
                key="case_sq_mode"
            )
            status_quo_mode = StatusQuoMode.RANDOM if "Random" in sq_mode else StatusQuoMode.NASH_ARGMAX
        
        # ==========================================================================
        # SYNTHETIC DATA SETTINGS
        # ==========================================================================
        elif data_source == "Synthetic Data":
            st.header("🎲 Synthetic Data Settings")
            
            num_households = st.slider("Number of Households", 20, 500, 100, step=10)
            num_attributes = st.slider("Number of Attributes", 2, 8, 4)
            
            st.write("**Levels per Attribute:**")
            attribute_levels = []
            cols = st.columns(min(num_attributes, 4))
            for i in range(num_attributes):
                with cols[i % 4]:
                    levels = st.number_input(f"Attr {i+1}", 2, 6, 3, key=f"syn_lv_{i}")
                    attribute_levels.append(levels)
            
            pref_corr = st.slider("H-W Preference Correlation", 0.0, 1.0, 0.4)
            alpha_mean = st.slider("Mean Bargaining Power (α)", 0.1, 0.9, 0.5)
            seed = st.number_input("Random Seed", 0, 9999, 7149)
            
            st.divider()
            
            # Stage 0 Settings for Synthetic
            st.header("🏠 Stage 0: Competitors")
            num_competitors = st.slider("Number of Competitors", 2, 10, 4)
            
            sq_mode = st.radio(
                "Status Quo Assignment",
                ["Random from Competitors", "Nash-Optimal from Competitors"],
                key="syn_sq_mode"
            )
            status_quo_mode = StatusQuoMode.RANDOM if "Random" in sq_mode else StatusQuoMode.NASH_ARGMAX
        
        # ==========================================================================
        # REAL DATA SETTINGS (Upload Files)
        # ==========================================================================
        else:  # Real Data (Upload Files)
            st.header("📁 Upload Your Data")
            
            st.markdown("""
            **Required Files:**
            - **Utility file (.01b)**: Conjoint part-worths
            - **Competitor file (.01c)**: Competitor products
            """)
            
            utility_file = st.file_uploader(
                "Upload Utility File (.01b)",
                type=['01b', 'txt', 'csv'],
                help="TV-format utility file with individual conjoint part-worths"
            )
            
            competitor_file = st.file_uploader(
                "Upload Competitor File (.01c)",
                type=['01c', 'txt', 'csv'],
                help="TV-format competitor product definitions"
            )
            
            st.divider()
            
            st.subheader("Household Pairing")
            pairing_mode = st.radio(
                "How to pair consumers into households",
                ["Consecutive (1-2, 3-4, ...)", "Random pairing"],
                help="Consecutive pairs rows 1-2 as HH1, 3-4 as HH2, etc."
            )
            pairing_mode_str = "consecutive" if "Consecutive" in pairing_mode else "random"
            
            alpha_default = st.slider("Default Bargaining Power (α)", 0.1, 0.9, 0.5, key="real_alpha")
            
            seed = st.number_input("Random Seed", 0, 9999, 7149, key="real_seed")
            
            st.divider()
            
            st.subheader("Stage 0 Settings")
            sq_mode = st.radio(
                "Status Quo Assignment",
                ["Random from Competitors", "Nash-Optimal from Competitors"],
                key="real_sq_mode"
            )
            status_quo_mode = StatusQuoMode.RANDOM if "Random" in sq_mode else StatusQuoMode.NASH_ARGMAX
        
        st.divider()
        
        # ==========================================================================
        # OPTIMIZATION SETTINGS (shared)
        # ==========================================================================
        st.header("⚖️ Optimization Settings")
        
        rule_names = {
            "Nash Product": AggregationRule.NASH,
            "Generalized Nash (Roth)": AggregationRule.ROTH,
            "Linear Weighted": AggregationRule.LINEAR,
            "Min (Rawlsian)": AggregationRule.MIN,
        }
        selected_rule = st.selectbox("Aggregation Rule", list(rule_names.keys()), index=1)
        aggregation_rule = rule_names[selected_rule]
        
        num_products = st.number_input("Products in Line", 1, 10, 2)
        market_size = st.number_input("Market Size", 100, 1000000, 1000, step=100)
        
        objective_choice = st.radio("Optimize for", ["Market Share", "Profit"])
        objective_type = ObjectiveType.MARKET_SHARE if objective_choice == "Market Share" else ObjectiveType.PROFIT
        
        default_margin = 1.0
        if objective_type == ObjectiveType.PROFIT:
            default_margin = st.number_input("Default Margin ($)", 0.1, 100.0, 1.0)
        
        fairness_weight = st.slider("Fairness Weight", 0.0, 1.0, 0.0)
        
        st.divider()
        
        st.header("🧬 GA Settings")
        pop_size = st.slider("Population Size", 20, 300, 100)
        max_gen = st.slider("Max Generations", 50, 1000, 300)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.3, 0.1)
    
    # ==========================================================================
    # MAIN PANEL - DATA GENERATION / LOADING
    # ==========================================================================
    
    st.header("Step 1: Load or Generate Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # ==========================================================================
        # SMARTTVS 4'US CASE - LOAD BUTTON
        # ==========================================================================
        if data_source == "SmartTVs 4'Us Case":
            if st.button("📺 Load SmartTVs 4'Us Case Data", type="primary", use_container_width=True):
                with st.spinner("Loading SmartTVs 4'Us case data..."):
                    try:
                        # Load from examples folder
                        utility_path = get_case_data_path("TV-Utility.01b")
                        competitor_path = get_case_data_path("TV-Competition.01c")
                        
                        if not utility_path.exists():
                            st.error(f"Case data not found at {utility_path}. Please ensure examples/smarttv_case/ folder exists in the repository.")
                            st.stop()
                        
                        # Read file contents
                        utility_content = utility_path.read_text()
                        competitor_content = competitor_path.read_text()
                        
                        # Load data
                        household_data, competitors, metadata = load_negasys_data_tv_format(
                            utility_filepath_or_content=utility_content,
                            competitor_filepath_or_content=competitor_content,
                            utility_is_content=True,
                            competitor_is_content=True,
                            alpha_default=alpha_default,
                            pairing_mode="consecutive"
                        )
                        
                        # Create config for Stage 0
                        config = NEGASYSConfig(
                            aggregation_rule=aggregation_rule,
                            objective=objective_type,
                            num_products=num_products,
                            market_size=market_size,
                            default_margin=default_margin
                        )
                        
                        # Run Stage 0
                        stage0_result = assign_status_quo(
                            household_data.households, competitors, config, 
                            status_quo_mode, seed=seed
                        )
                        
                        st.session_state.household_data = household_data
                        st.session_state.competitors = competitors
                        st.session_state.stage0_result = stage0_result
                        st.session_state.result = None
                        st.session_state.rule_comparison = None
                        st.session_state.data_metadata = metadata
                        st.session_state.data_metadata['source'] = 'SmartTVs Case'
                        
                        st.success(
                            f"✅ Loaded SmartTVs 4'Us case: "
                            f"{metadata['num_households']} households, "
                            f"{metadata['num_competitors']} competitors"
                        )
                        
                    except Exception as e:
                        st.error(f"Error loading case data: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # ==========================================================================
        # SYNTHETIC DATA - GENERATE BUTTON
        # ==========================================================================
        elif data_source == "Synthetic Data":
            if st.button("🔄 Generate Synthetic Data + Run Stage 0", type="primary", use_container_width=True):
                with st.spinner("Generating synthetic data..."):
                    try:
                        # Generate household data
                        household_data = generate_synthetic_household_data(
                            num_households=num_households,
                            num_attributes=num_attributes,
                            attribute_levels=attribute_levels,
                            preference_correlation=pref_corr,
                            alpha_mean=alpha_mean,
                            seed=seed
                        )
                        
                        # Generate competitors
                        competitors = generate_random_competitors(
                            num_competitors, num_attributes, attribute_levels, seed=seed+1000
                        )
                        
                        # Create config for Stage 0
                        config = NEGASYSConfig(
                            aggregation_rule=aggregation_rule,
                            objective=objective_type,
                            num_products=num_products,
                            market_size=market_size,
                            default_margin=default_margin
                        )
                        
                        # Run Stage 0
                        stage0_result = assign_status_quo(
                            household_data.households, competitors, config, 
                            status_quo_mode, seed=seed
                        )
                        
                        st.session_state.household_data = household_data
                        st.session_state.competitors = competitors
                        st.session_state.stage0_result = stage0_result
                        st.session_state.result = None
                        st.session_state.rule_comparison = None
                        st.session_state.data_metadata = {
                            'source': 'Synthetic',
                            'num_households': num_households,
                            'num_attributes': num_attributes,
                            'attribute_levels': attribute_levels,
                            'num_competitors': num_competitors
                        }
                        
                        st.success(f"✅ Generated {num_households} households with {num_competitors} competitors")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # ==========================================================================
        # REAL DATA (UPLOAD FILES) - LOAD BUTTON
        # ==========================================================================
        else:  # Real Data (Upload Files)
            if utility_file is not None and competitor_file is not None:
                if st.button("📥 Load Uploaded Data + Run Stage 0", type="primary", use_container_width=True):
                    with st.spinner("Loading and processing uploaded data..."):
                        try:
                            # Read file contents
                            utility_content = utility_file.read().decode('utf-8')
                            competitor_content = competitor_file.read().decode('utf-8')
                            
                            # Load data
                            household_data, competitors, metadata = load_negasys_data_tv_format(
                                utility_filepath_or_content=utility_content,
                                competitor_filepath_or_content=competitor_content,
                                utility_is_content=True,
                                competitor_is_content=True,
                                alpha_default=alpha_default,
                                pairing_mode=pairing_mode_str
                            )
                            
                            # Create config for Stage 0
                            config = NEGASYSConfig(
                                aggregation_rule=aggregation_rule,
                                objective=objective_type,
                                num_products=num_products,
                                market_size=market_size,
                                default_margin=default_margin
                            )
                            
                            # Run Stage 0
                            stage0_result = assign_status_quo(
                                household_data.households, competitors, config, 
                                status_quo_mode, seed=seed
                            )
                            
                            st.session_state.household_data = household_data
                            st.session_state.competitors = competitors
                            st.session_state.stage0_result = stage0_result
                            st.session_state.result = None
                            st.session_state.rule_comparison = None
                            st.session_state.data_metadata = metadata
                            st.session_state.data_metadata['source'] = 'Uploaded'
                            
                            st.success(
                                f"✅ Loaded {metadata['original_num_consumers']} consumers → "
                                f"{metadata['num_households']} households, "
                                f"{metadata['num_competitors']} competitors"
                            )
                            
                            # Show validation warnings
                            warnings = validate_data_consistency(household_data, competitors)
                            if warnings:
                                st.warning("Data validation warnings:")
                                for w in warnings:
                                    st.write(f"⚠️ {w}")
                            
                        except Exception as e:
                            st.error(f"Error loading data: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("👆 Please upload both utility (.01b) and competitor (.01c) files in the sidebar.")
    
    with col2:
        if st.session_state.household_data:
            total_profiles = np.prod(st.session_state.household_data.attribute_levels)
            st.metric("Possible Products", f"{total_profiles:,}")
    
    # ==========================================================================
    # DATA SUMMARY
    # ==========================================================================
    
    if st.session_state.household_data is not None:
        hd = st.session_state.household_data
        s0 = st.session_state.stage0_result
        metadata = st.session_state.data_metadata
        
        st.divider()
        st.header("📊 Data Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Source", metadata.get('source', 'unknown'))
        with col2:
            st.metric("Households", f"{hd.num_households:,}")
        with col3:
            st.metric("Attributes", hd.num_attributes)
        with col4:
            avg_alpha = np.mean([h.alpha for h in hd.households])
            st.metric("Avg α", f"{avg_alpha:.2f}")
        with col5:
            st.metric("Competitors", s0.num_competitors if s0 else "N/A")
        
        # Attribute details
        with st.expander("📋 Attribute Details"):
            attr_data = []
            for k in range(hd.num_attributes):
                attr_name = hd.attribute_names[k] if hd.attribute_names else f"Attr {k+1}"
                num_levels = hd.attribute_levels[k]
                
                if hd.level_names and k < len(hd.level_names):
                    level_list = ", ".join(hd.level_names[k])
                else:
                    level_list = ", ".join([f"Level {l+1}" for l in range(num_levels)])
                
                attr_data.append({
                    'Attribute': attr_name,
                    'Levels': num_levels,
                    'Level Names': level_list
                })
            
            st.dataframe(pd.DataFrame(attr_data), hide_index=True, use_container_width=True)
        
        # Stage 0 results
        with st.expander("🏠 Stage 0 Status Quo Distribution", expanded=True):
            if s0:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Mode:** {s0.mode}")
                    for idx, share in s0.product_shares.items():
                        comp = st.session_state.competitors[idx]
                        profile_str = format_product_profile(
                            comp.attributes, 
                            hd.level_names
                        )
                        st.write(f"**Comp {idx+1}:** {profile_str}")
                        st.write(f"  → {share*100:.1f}% of households")
                
                with col2:
                    shares = list(s0.product_shares.values())
                    labels = [f"Comp {i+1}" for i in range(len(shares))]
                    fig = px.pie(
                        values=[s*100 for s in shares], 
                        names=labels, 
                        title="Status Quo Distribution"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Download household data
        st.subheader("📥 Download Data")
        
        col1, col2 = st.columns(2)
        with col1:
            household_csv = export_household_data_csv(hd)
            st.download_button(
                "📄 Download Household Data (CSV)",
                data=household_csv,
                file_name=f"negasys_households_{int(time.time())}.csv",
                mime="text/csv",
                help="CSV with one row per household-member"
            )
        
        with col2:
            if metadata.get('source') == 'Uploaded':
                st.info(f"Original: {metadata.get('original_num_consumers', '?')} consumers → {hd.num_households} households")
        
        st.divider()
        
        # ==========================================================================
        # OPTIMIZATION
        # ==========================================================================
        
        st.header("Step 2: Run Optimization")
        
        st.markdown(f"""
        **Configuration:**
        - Rule: {selected_rule} | Objective: {objective_choice} | Products: {num_products}
        - Population: {pop_size} | Generations: {max_gen}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            run_single = st.button("🚀 Run Single Optimization", type="primary", use_container_width=True)
        
        with col2:
            run_all = st.button("🔬 Test All 4 Decision Rules", type="secondary", use_container_width=True)
        
        # Single optimization
        if run_single:
            config = NEGASYSConfig(
                aggregation_rule=aggregation_rule,
                objective=objective_type,
                num_products=num_products,
                market_size=market_size,
                default_margin=default_margin,
                fairness_weight=fairness_weight
            )
            
            ga_params = GAParams(
                population_size=pop_size,
                max_generations=max_gen,
                mutation_rate=mutation_rate,
                seed=seed
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(gen, fitness):
                progress_bar.progress(min(gen / max_gen, 1.0))
                status_text.text(f"Generation {gen}/{max_gen}: Best {objective_choice} = {fitness:.3f}")
            
            try:
                result = run_negasys_ga(hd, config, ga_params, progress_callback=progress_callback)
                st.session_state.result = result
                st.session_state.rule_comparison = None
                
                progress_bar.progress(1.0)
                status_text.success(f"✅ Complete in {result.elapsed_seconds:.2f}s")
                
            except Exception as e:
                st.error(f"Optimization failed: {e}")
        
        # Run all 4 rules
        if run_all:
            config = NEGASYSConfig(
                aggregation_rule=aggregation_rule,
                objective=objective_type,
                num_products=num_products,
                market_size=market_size,
                default_margin=default_margin,
                fairness_weight=fairness_weight
            )
            
            ga_params = GAParams(
                population_size=pop_size,
                max_generations=max_gen,
                mutation_rate=mutation_rate,
                seed=seed
            )
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                def progress_callback(msg):
                    status_text.text(msg)
                
                comparisons = run_all_decision_rules(
                    hd, config, ga_params, progress_callback=progress_callback
                )
                
                st.session_state.rule_comparison = comparisons
                st.session_state.result = None
                
                progress_bar.progress(1.0)
                status_text.success("✅ All 4 rules tested!")
                
            except Exception as e:
                st.error(f"Comparison failed: {e}")
        
        st.divider()
        
        # ==========================================================================
        # RESULTS
        # ==========================================================================
        
        # Single result display
        if st.session_state.result:
            result = st.session_state.result
            
            st.header("📈 Optimization Results")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Market Share", f"{result.total_share:.1f}%")
            with col2:
                st.metric("Profit", f"${result.total_profit:,.2f}")
            with col3:
                st.metric("Generations", result.generations)
            with col4:
                st.metric("Time", f"{result.elapsed_seconds:.2f}s")
            with col5:
                if result.convergence_history and len(result.convergence_history) > 1:
                    improvement = result.convergence_history[-1] - result.convergence_history[0]
                    st.metric("Improvement", f"+{improvement:.2f}")
            
            # Product line with level names
            st.subheader("🎯 Optimal Product Line")
            
            products_data = []
            for i, product in enumerate(result.best_line.products):
                row = {'Product': f"Product {i+1}"}
                
                for k in range(hd.num_attributes):
                    attr_name = hd.attribute_names[k] if hd.attribute_names else f"Attr {k+1}"
                    
                    if hd.level_names and k < len(hd.level_names) and product.attributes[k] < len(hd.level_names[k]):
                        row[attr_name] = hd.level_names[k][product.attributes[k]]
                    else:
                        row[attr_name] = f"Level {product.attributes[k] + 1}"
                
                if result.best_line.share_per_product:
                    row['Share %'] = f"{result.best_line.share_per_product[i]:.1f}%"
                row['Margin'] = f"${product.margin:.2f}" if product.margin > 0 else f"${default_margin:.2f}"
                products_data.append(row)
            
            st.dataframe(pd.DataFrame(products_data), hide_index=True, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if result.best_line.share_per_product:
                    shares = result.best_line.share_per_product.copy()
                    sq_share = 100 - sum(shares)
                    labels = [f"Product {i+1}" for i in range(len(shares))]
                    values = shares.copy()
                    if sq_share > 0.1:
                        labels.append("Status Quo")
                        values.append(sq_share)
                    fig = px.pie(values=values, names=labels, title="Market Share Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if result.avg_utility_H and result.avg_utility_W:
                    fig = create_fairness_gauge(result.avg_utility_H, result.avg_utility_W)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Convergence
            if result.convergence_history and len(result.convergence_history) > 1:
                st.subheader("📈 Convergence History")
                fig = px.line(
                    x=list(range(len(result.convergence_history))),
                    y=result.convergence_history,
                    labels={'x': 'Generation', 'y': result.objective_name}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            st.subheader("📥 Download Results")
            
            config = NEGASYSConfig(
                aggregation_rule=aggregation_rule,
                default_margin=default_margin
            )
            results_csv = export_optimization_results_csv(result, hd, config)
            
            st.download_button(
                "📄 Download Optimization Results (CSV)",
                data=results_csv,
                file_name=f"negasys_results_{int(time.time())}.csv",
                mime="text/csv"
            )
        
        # Rule comparison display
        if st.session_state.rule_comparison:
            comparisons = st.session_state.rule_comparison
            
            st.header("🔬 Decision Rule Comparison")
            
            # Comparison table
            comp_df = pd.DataFrame([
                {
                    'Decision Rule': c.rule_name,
                    'Market Share (%)': f"{c.total_share:.1f}",
                    'Profit ($)': f"{c.total_profit:,.2f}",
                    'Avg U_H': f"{c.avg_utility_H:.4f}" if c.avg_utility_H else "N/A",
                    'Avg U_W': f"{c.avg_utility_W:.4f}" if c.avg_utility_W else "N/A",
                    'Dispersion': f"{c.utility_dispersion:.4f}" if c.utility_dispersion else "N/A",
                    'Time (s)': f"{c.elapsed_seconds:.2f}",
                }
                for c in comparisons
            ])
            
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
            # Visualization
            fig = create_rule_comparison_chart(comparisons)
            st.plotly_chart(fig, use_container_width=True)
            
            # Product lines per rule with level names
            with st.expander("📋 Product Lines by Rule"):
                for comp in comparisons:
                    st.write(f"**{comp.rule_name}:**")
                    for p_idx, attrs in enumerate(comp.product_line):
                        share = comp.share_per_product[p_idx] if p_idx < len(comp.share_per_product) else 0
                        
                        # Format with level names
                        if hd.level_names:
                            parts = []
                            for k, lv in enumerate(attrs):
                                if k < len(hd.level_names) and lv < len(hd.level_names[k]):
                                    parts.append(hd.level_names[k][lv])
                                else:
                                    parts.append(f"L{lv+1}")
                            profile_str = " | ".join(parts)
                        else:
                            profile_str = str(attrs)
                        
                        st.write(f"  Product {p_idx+1}: {profile_str} → {share:.1f}%")
            
            # Download comparison
            st.subheader("📥 Download Comparison")
            
            comparison_csv = export_rule_comparison_csv(comparisons, hd)
            
            st.download_button(
                "📄 Download Rule Comparison (CSV)",
                data=comparison_csv,
                file_name=f"negasys_rule_comparison_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Select a data source and load data using Step 1 to begin.")
        
        with st.expander("📚 About NEGASYS", expanded=True):
            st.markdown("""
            ### What is NEGASYS?
            
            NEGASYS extends traditional product line design to model **household dyadic decision-making**.
            Instead of individual consumers, households consist of two partners (H and W) who jointly 
            decide which product to purchase through negotiation.
            
            ### Data Sources
            
            **SmartTVs 4'Us Case** *(Recommended for learning)*
            - Built-in teaching case data - no uploads needed!
            - 100 households, 6 attributes, 5 competitors
            - See the case narrative for full context
            
            **Synthetic Data:**
            - Generate random households with configurable preferences
            - Useful for testing and experimentation
            
            **Real Data (Upload Files):**
            - Upload conjoint utility file (.01b) with individual part-worths
            - Upload competitor file (.01c) with existing products
            - Consecutive consumers are paired into households
            
            ### Key Concepts
            
            1. **Individual Utilities**: Each partner has their own part-worth preferences
            2. **Aggregation Rules**: Nash, Roth, Linear, or Min combining rules
            3. **Status Quo**: Competitors define what households currently own
            4. **Bargaining Power (α)**: Controls relative influence within household
            """)


def show_footer():
    st.divider()
    st.markdown(f"""
    ---
    **NEGASYS v{NEGASYS_VERSION}** | January 2026 | P.V. (Sundar) Balakrishnan
    
    **New in v5.1:**
    - 📺 Built-in SmartTVs 4'Us teaching case (no file uploads needed!)
    
    **Features:**
    - ✅ Real data support (TV format .01b/.01c files)
    - ✅ Automatic household pairing from individual conjoint data
    - ✅ Level names display in product lines
    - ✅ Multi-rule comparison
    
    **References:** Nash (1950), Roth (1979), Balakrishnan & Jacob (1996)
    """)


if __name__ == "__main__":
    main()
    show_footer()
