#!/usr/bin/env python3
"""
Interactive dashboard for XRAI using Streamlit.

This dashboard loads checkpoint data and provides interactive visualizations
for analyzing the evolutionary algorithm's performance.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob
import re

# Set page configuration
st.set_page_config(
    page_title="XRAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load checkpoint data
@st.cache_data(ttl=10)  # Cache for 10 seconds
def load_checkpoint(checkpoint_path):
    """Load checkpoint data from file."""
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None

# Function to find available checkpoints
def find_checkpoints(results_dir="results"):
    """Find all available checkpoints in the results directory."""
    checkpoints = []
    
    # Find all directories in results
    for dir_path in glob(os.path.join(results_dir, "*")):
        if os.path.isdir(dir_path):
            # Look for latest_checkpoint.pkl or any checkpoint files
            latest_path = os.path.join(dir_path, "latest_checkpoint.pkl")
            if os.path.exists(latest_path):
                checkpoints.append((dir_path, latest_path))
            else:
                # Look for other checkpoint files
                checkpoint_files = glob(os.path.join(dir_path, "checkpoint_*.pkl"))
                if checkpoint_files:
                    # Sort by generation number if possible
                    def extract_gen(path):
                        match = re.search(r'gen(\d+)', path)
                        return int(match.group(1)) if match else 0
                    
                    checkpoint_files.sort(key=extract_gen, reverse=True)
                    checkpoints.append((dir_path, checkpoint_files[0]))
    
    return checkpoints

# Function to create chaotic function plot
def plot_chaotic_function(r, x0=0.5, iterations=100):
    """Create a plot of the chaotic function and its iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the chaotic function
    def chaotic_function(x, r):
        result = r * x * (1 - x)
        # Ensure result is within bounds for visualization
        return max(0.0, min(1.0, result))
    
    # Plot the function
    x = np.linspace(0, 1, 1000)
    y = [chaotic_function(xi, r) for xi in x]
    ax.plot(x, y, 'r-', label=f"f(x) = r·x·(1-x), r={r:.2f}")
    
    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], 'k--', label="y=x")
    
    # Generate iterations
    iterations_x = [x0]
    iterations_y = [0]
    
    current_x = x0
    for i in range(iterations):
        # Calculate f(x)
        y_val = chaotic_function(current_x, r)
        
        # Add vertical line to f(x)
        iterations_x.append(current_x)
        iterations_y.append(y_val)
        
        # Add horizontal line to diagonal
        iterations_x.append(current_x)
        iterations_y.append(y_val)
        
        # Move to diagonal
        iterations_x.append(y_val)
        iterations_y.append(y_val)
        
        # Update current_x for next iteration
        current_x = y_val
    
    # Plot the iterations
    ax.plot(iterations_x, iterations_y, 'b-', alpha=0.7, label=f"Iterations (x₀={x0:.2f})")
    
    # Mark the starting point
    ax.plot(x0, 0, 'go', markersize=8, label="Starting Point")
    
    # Set labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Chaotic Function (r={r:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

# Function to create phase space plot
def plot_phase_space(r, chaos_values_history=None):
    """Create a phase space plot of the chaotic function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define the chaotic function
    def chaotic_function(x, r):
        return r * x * (1 - x)
    
    # Plot the function
    x = np.linspace(0, 1, 1000)
    y = [chaotic_function(xi, r) for xi in x]
    ax.plot(x, y, 'r-', alpha=0.5, label=f"f(x) = r·x·(1-x), r={r:.2f}")
    
    # Plot the diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="y=x")
    
    # Plot random samples if available
    if chaos_values_history is not None and len(chaos_values_history) > 0:
        # Generate random x values
        num_points = min(500, len(chaos_values_history))
        random_indices = np.random.choice(len(chaos_values_history), num_points, replace=False)
        random_x = np.random.uniform(0, 1, num_points)
        random_y = np.array([chaos_values_history[i] for i in random_indices])
        
        # Plot scatter points
        ax.scatter(random_x, random_y, s=20, c='blue', alpha=0.6, label="Random Samples")
        
        # Add statistics
        stats_text = (
            f"Chaos Statistics:\n"
            f"Min: {np.min(chaos_values_history):.4f}\n"
            f"Max: {np.max(chaos_values_history):.4f}\n"
            f"Mean: {np.mean(chaos_values_history):.4f}\n"
            f"Std Dev: {np.std(chaos_values_history):.4f}"
        )
        
        # Add text box with statistics
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output f(x)")
    ax.set_title(f"Chaotic Function Phase Space (r={r:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

# Main dashboard
def main():
    st.title("XRAI Evolution Dashboard")
    
    # Sidebar for checkpoint selection
    st.sidebar.header("Checkpoint Selection")
    
    # Find available checkpoints
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        st.sidebar.warning("No checkpoints found in the results directory.")
        checkpoint_path = st.sidebar.text_input("Enter checkpoint path manually:", "")
        if not checkpoint_path:
            st.warning("Please select a checkpoint to continue.")
            return
    else:
        # Create a dropdown for checkpoint selection
        checkpoint_options = {os.path.basename(dir_path): file_path for dir_path, file_path in checkpoints}
        selected_checkpoint = st.sidebar.selectbox(
            "Select checkpoint:",
            options=list(checkpoint_options.keys()),
            format_func=lambda x: f"{x} ({os.path.basename(checkpoint_options[x])})"
        )
        
        checkpoint_path = checkpoint_options[selected_checkpoint]
    
    # Add this near the top of the main function, after checkpoint selection
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Load the selected checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        st.error("Failed to load checkpoint. The file may be corrupted.")
        return

    # Check for required data
    if 'generation' not in checkpoint or 'fitness_history' not in checkpoint:
        st.error("Checkpoint is missing required data.")
        st.json(list(checkpoint.keys()))  # Show what keys are available
        return
    
    # Extract data from checkpoint
    generation = checkpoint.get('generation', 0)
    fitness_history = checkpoint.get('fitness_history', [])
    weight_history = checkpoint.get('weight_history', [])
    predictors = checkpoint.get('predictors', [])
    meta_predictors = checkpoint.get('meta_predictors', [])
    global_best_predictor = checkpoint.get('global_best_predictor', None)
    global_best_fitness = checkpoint.get('global_best_fitness', 0)
    global_best_generation = checkpoint.get('global_best_generation', 0)
    parameters = checkpoint.get('parameters', {})
    chaos_values_history = checkpoint.get('chaos_values_history', [])
    
    # Extract hierarchy information if available
    current_hierarchy_level = checkpoint.get('current_hierarchy_level', None)
    hierarchy_generation_counts = checkpoint.get('hierarchy_generation_counts', None)
    
    # Extract parameters
    r = parameters.get('r', 0)
    population_size = parameters.get('population_size', 0)
    mutation_rate = parameters.get('mutation_rate', 0)
    
    # Create dataframes for plotting
    if fitness_history:
        predictor_fitness, meta_fitness = zip(*fitness_history)
        fitness_df = pd.DataFrame({
            'Generation': np.arange(len(fitness_history)),
            'Predictor Fitness': predictor_fitness,
            'Meta-Predictor Fitness': meta_fitness
        })
    else:
        fitness_df = pd.DataFrame(columns=['Generation', 'Predictor Fitness', 'Meta-Predictor Fitness'])
    
    if weight_history:
        weight_df = pd.DataFrame(weight_history, columns=['a (x²)', 'b (x)', 'c (constant)'])
        weight_df['Generation'] = np.arange(len(weight_history))
    else:
        weight_df = pd.DataFrame(columns=['Generation', 'a (x²)', 'b (x)', 'c (constant)'])
    
    # Display key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Generation", f"{generation:,}")
        if current_hierarchy_level is not None and hierarchy_generation_counts is not None:
            st.caption(f"Hierarchy Level: {current_hierarchy_level + 1}/{len(hierarchy_generation_counts)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Global Best Fitness", f"{global_best_fitness:.6f}")
        st.caption(f"From Generation: {global_best_generation:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Population Size", f"{population_size}")
        st.caption(f"Mutation Rate: {mutation_rate:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Chaos Parameter (r)", f"{r:.4f}")
        if fitness_history:
            current_fitness = fitness_history[-1][0]
            st.caption(f"Current Fitness: {current_fitness:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Fitness Evolution", 
        "Weight Evolution", 
        "Chaos Analysis",
        "Predictor Details",
        "Raw Data"
    ])
    
    # Tab 1: Fitness Evolution
    with tab1:
        st.header("Fitness Evolution")
        
        # Fitness history plot
        if not fitness_df.empty:
            # Use the new downsampling function
            def downsample_for_plotting(df, max_points=2000):
                """Intelligently downsample dataframe for plotting."""
                if len(df) <= max_points:
                    return df.copy()
                
                # For very large datasets, use more aggressive downsampling
                if len(df) > 100000:
                    # Use logarithmic sampling to show more detail in early generations
                    indices = np.unique(np.logspace(0, np.log10(len(df)-1), max_points).astype(int))
                    return df.iloc[indices].copy()
                
                # For moderately large datasets, use linear sampling
                indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
                return df.iloc[indices].copy()

            # Use the function
            plot_df = downsample_for_plotting(fitness_df)
            
            # Plot fitness history
            st.line_chart(
                plot_df.set_index('Generation')[['Predictor Fitness', 'Meta-Predictor Fitness']]
            )
            
            # Calculate and display fitness deviation
            if len(fitness_history) > 0:
                fitness_deviation = np.array([abs(p - m) for p, m in fitness_history])
                deviation_df = pd.DataFrame({
                    'Generation': np.arange(len(fitness_deviation)),
                    'Fitness Deviation': fitness_deviation
                })
                
                # Downsample if needed
                if len(deviation_df) > 10000:
                    indices = np.linspace(0, len(deviation_df) - 1, 5000, dtype=int)
                    deviation_df = deviation_df.iloc[indices].copy()
                
                st.subheader("Fitness Deviation")
                st.line_chart(deviation_df.set_index('Generation'))
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Deviation", f"{np.mean(fitness_deviation):.6f}")
                with col2:
                    st.metric("Max Deviation", f"{np.max(fitness_deviation):.6f}")
                with col3:
                    st.metric("Min Deviation", f"{np.min(fitness_deviation):.6f}")
        else:
            st.warning("No fitness history data available.")
    
    # Tab 2: Weight Evolution
    with tab2:
        st.header("Weight Evolution")
        
        if not weight_df.empty:
            # Downsample for very large datasets
            if len(weight_df) > 10000:
                indices = np.linspace(0, len(weight_df) - 1, 5000, dtype=int)
                plot_df = weight_df.iloc[indices].copy()
            else:
                plot_df = weight_df.copy()
            
            # Plot weight evolution
            st.line_chart(plot_df.set_index('Generation'))
            
            # Display current best weights
            st.subheader("Best Weights")
            col1, col2, col3 = st.columns(3)
            
            if global_best_predictor:
                with col1:
                    st.metric("a (x²)", f"{global_best_predictor.weights[0]:.6f}")
                with col2:
                    st.metric("b (x)", f"{global_best_predictor.weights[1]:.6f}")
                with col3:
                    st.metric("c (constant)", f"{global_best_predictor.weights[2]:.6f}")
                
                # Display the equation
                st.markdown(f"""
                **Global Best Equation:**  
                f(x) = {global_best_predictor.weights[0]:.6f}x² + {global_best_predictor.weights[1]:.6f}x + {global_best_predictor.weights[2]:.6f}
                """)
        else:
            st.warning("No weight history data available.")
    
    # Tab 3: Chaos Analysis
    with tab3:
        st.header("Chaos Analysis")
        
        # Interactive chaos function visualization
        st.subheader("Chaotic Function Visualization")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Allow user to adjust parameters
            custom_r = st.slider("Chaos Parameter (r)", 0.0, 5.0, float(r), 0.01)
            x0 = st.slider("Starting Point (x₀)", 0.0, 1.0, 0.5, 0.01)
            iterations = st.slider("Iterations", 10, 200, 50)
        
        with col2:
            # Create and display the chaotic function plot
            chaos_fig = plot_chaotic_function(custom_r, x0, iterations)
            st.pyplot(chaos_fig)
        
        # Phase space visualization
        st.subheader("Phase Space Visualization")
        phase_fig = plot_phase_space(r, chaos_values_history)
        st.pyplot(phase_fig)
        
        # Display chaos statistics if available
        if chaos_values_history and len(chaos_values_history) > 0:
            st.subheader("Chaos Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min Value", f"{np.min(chaos_values_history):.6f}")
            with col2:
                st.metric("Max Value", f"{np.max(chaos_values_history):.6f}")
            with col3:
                st.metric("Mean Value", f"{np.mean(chaos_values_history):.6f}")
            with col4:
                st.metric("Std Deviation", f"{np.std(chaos_values_history):.6f}")
            
            # Plot histogram of chaos values
            st.subheader("Distribution of Chaos Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(chaos_values_history, bins=50, alpha=0.7)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Chaotic Function Values")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Tab 4: Predictor Details
    with tab4:
        st.header("Predictor Details")
        
        # Display global best predictor details
        if global_best_predictor:
            st.subheader("Global Best Predictor")
            st.markdown(f"""
            - **Fitness:** {global_best_fitness:.6f}
            - **Generation:** {global_best_generation:,}
            - **Equation:** f(x) = {global_best_predictor.weights[0]:.6f}x² + {global_best_predictor.weights[1]:.6f}x + {global_best_predictor.weights[2]:.6f}
            """)
            
            # Plot the predictor's output vs the actual chaotic function
            st.subheader("Predictor vs Actual Function")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Define the chaotic function
            def chaotic_function(x, r):
                return r * x * (1 - x)
            
            # Generate x values
            x_values = np.linspace(0, 1, 1000)
            
            # Calculate actual and predicted values
            actual_values = np.array([chaotic_function(x, r) for x in x_values])
            predicted_values = np.array([global_best_predictor.predict(x) for x in x_values])
            
            # Calculate error
            error = np.abs(actual_values - predicted_values)
            
            # Plot actual and predicted values
            ax.plot(x_values, actual_values, 'r-', label="Actual (Chaotic Function)", alpha=0.7)
            ax.plot(x_values, predicted_values, 'g-', label="Predicted (Best Predictor)", alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel("Input (x)")
            ax.set_ylabel("Output")
            ax.set_title("Predictor vs Actual Chaotic Function")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Plot error
            st.subheader("Prediction Error")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, error, 'b-', label="Absolute Error")
            ax.axhline(y=np.mean(error), color='r', linestyle='--', 
                      label=f"Mean Error: {np.mean(error):.6f}")
            
            ax.set_xlabel("Input (x)")
            ax.set_ylabel("Absolute Error")
            ax.set_title("Prediction Error")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display error statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Error", f"{np.mean(error):.6f}")
            with col2:
                st.metric("Max Error", f"{np.max(error):.6f}")
            with col3:
                st.metric("Min Error", f"{np.min(error):.6f}")
        
        # Display current population statistics if available
        if predictors:
            st.subheader("Current Population")
            
            # Calculate fitness for each predictor
            def chaotic_function(x, r):
                return r * x * (1 - x)
            
            x_test = np.random.uniform(0, 1)
            true_value = chaotic_function(x_test, r)
            
            predictor_fitness = [1 - abs(p.predict(x_test) - true_value) for p in predictors]
            
            # Create a dataframe with predictor information
            predictor_df = pd.DataFrame({
                'Predictor': range(len(predictors)),
                'Fitness': predictor_fitness,
                'Weight a': [p.weights[0] for p in predictors],
                'Weight b': [p.weights[1] for p in predictors],
                'Weight c': [p.weights[2] for p in predictors]
            })
            
            # Sort by fitness
            predictor_df = predictor_df.sort_values('Fitness', ascending=False).reset_index(drop=True)
            
            # Display the top 10 predictors
            st.dataframe(predictor_df.head(10))
            
            # Plot fitness distribution
            st.subheader("Population Fitness Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(predictor_fitness, bins=20, alpha=0.7)
            ax.axvline(x=np.mean(predictor_fitness), color='r', linestyle='--',
                      label=f"Mean Fitness: {np.mean(predictor_fitness):.4f}")
            
            ax.set_xlabel("Fitness")
            ax.set_ylabel("Count")
            ax.set_title("Population Fitness Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
    
    # Tab 5: Raw Data
    with tab5:
        st.header("Raw Data")
        
        # Display checkpoint information
        st.subheader("Checkpoint Information")
        st.markdown(f"""
        - **Checkpoint Path:** {checkpoint_path}
        - **Generation:** {generation:,}
        - **Population Size:** {population_size}
        - **Mutation Rate:** {mutation_rate:.6f}
        - **Chaos Parameter (r):** {r:.6f}
        - **Global Best Fitness:** {global_best_fitness:.6f} (Generation {global_best_generation:,})
        """)
        
        # Display hierarchy information if available
        if current_hierarchy_level is not None and hierarchy_generation_counts is not None:
            st.subheader("Hierarchy Information")
            st.markdown(f"""
            - **Current Level:** {current_hierarchy_level + 1}/{len(hierarchy_generation_counts)}
            - **Generation Distribution:** {hierarchy_generation_counts}
            """)
            
            # Calculate progress
            total_generations = sum(hierarchy_generation_counts)
            completed_levels_gens = sum(hierarchy_generation_counts[:current_hierarchy_level])
            current_level_progress = generation - completed_levels_gens
            current_level_total = hierarchy_generation_counts[current_hierarchy_level]
            
            # Display progress bars
            st.subheader("Progress")
            
            # Overall progress
            overall_progress = generation / total_generations
            st.markdown(f"**Overall Progress:** {generation:,}/{total_generations:,} generations ({overall_progress:.1%})")
            st.progress(overall_progress)
            
            # Current level progress
            level_progress = max(0.0, min(1.0, current_level_progress / current_level_total))
            st.markdown(f"**Current Level Progress:** {current_level_progress:,}/{current_level_total:,} generations ({level_progress:.1%})")
            st.progress(level_progress)
        
        # Raw data exploration
        st.subheader("Raw Data Exploration")
        
        data_option = st.selectbox(
            "Select data to view:",
            ["Fitness History", "Weight History", "Chaos Values", "Parameters"]
        )
        
        if data_option == "Fitness History" and fitness_history:
            st.dataframe(fitness_df)
            st.download_button(
                "Download Fitness History CSV",
                fitness_df.to_csv(index=False).encode('utf-8'),
                "fitness_history.csv",
                "text/csv",
                key='download-fitness-csv'
            )
        
        elif data_option == "Weight History" and weight_history:
            st.dataframe(weight_df)
            st.download_button(
                "Download Weight History CSV",
                weight_df.to_csv(index=False).encode('utf-8'),
                "weight_history.csv",
                "text/csv",
                key='download-weight-csv'
            )
        
        elif data_option == "Chaos Values" and chaos_values_history:
            chaos_df = pd.DataFrame({
                'Index': range(len(chaos_values_history)),
                'Value': chaos_values_history
            })
            st.dataframe(chaos_df)
            st.download_button(
                "Download Chaos Values CSV",
                chaos_df.to_csv(index=False).encode('utf-8'),
                "chaos_values.csv",
                "text/csv",
                key='download-chaos-csv'
            )
        
        elif data_option == "Parameters":
            st.json(parameters)

# Run the dashboard
if __name__ == "__main__":
    main() 
