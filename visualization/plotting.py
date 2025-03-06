"""
Plotting functions for visualization.
"""
import numpy as np
import matplotlib.pyplot as plt

def create_figure_if_closed(fig, figsize=(10, 5)):
    """Create a new figure if the previous one was closed."""
    if not plt.fignum_exists(fig.number):
        new_fig, new_ax = plt.subplots(figsize=figsize)
        return new_fig, new_ax
    return fig, fig.axes[0]

def update_plots(figures_axes, generation, fitness_history, weight_history, current_best_predictor, 
                global_best_predictor, global_best_fitness, global_best_generation, r, 
                mutation_rate_history=None, hierarchy_level=None, total_levels=None, chaos_values_history=None):
    """Update all visualization plots."""
    # Check if visualization is disabled
    if figures_axes is None:
        return None
    
    # Check if we have the new dashboard format or the old separate figures format
    if isinstance(figures_axes[0], tuple):
        # Old format with separate figures
        if len(figures_axes) == 5:
            (fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4), (fig5, ax5) = figures_axes
            
            # ... (original update code for separate figures) ...
            
            return figures_axes
    
    # New dashboard format
    try:
        # Import here to avoid circular imports
        from core import chaotic_function
        
        # Unpack the figures and axes
        main_fig, main_axes, *individual_figs_axes = figures_axes
        
        # Unpack main axes
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = main_axes
        
        # Unpack individual figures and axes for saving
        (fig1, ax1_save), (fig2, ax2_save), (fig3, ax3_save), (fig4, ax4_save), \
        (fig5, ax5_save), (fig6, ax6_save), (fig7, ax7_save), (fig8, ax8_save) = individual_figs_axes
        
        # Check if main figure is still open
        if not plt.fignum_exists(main_fig.number):
            # Recreate the main figure if it was closed
            main_fig = plt.figure(figsize=(18, 12))
            gs = main_fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
            
            ax1 = main_fig.add_subplot(gs[0, 0])
            ax2 = main_fig.add_subplot(gs[0, 1])
            ax3 = main_fig.add_subplot(gs[0, 2])
            ax4 = main_fig.add_subplot(gs[1, 0])
            ax5 = main_fig.add_subplot(gs[1, 1])
            ax6 = main_fig.add_subplot(gs[1, 2])
            ax7 = main_fig.add_subplot(gs[2, 0])
            ax8 = main_fig.add_subplot(gs[2, 1])
            ax9 = main_fig.add_subplot(gs[2, 2])
            ax9.axis('off')
            
            main_axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)
            
            # Set window close event
            main_fig.canvas.mpl_connect('close_event', lambda event: None)
        
        # Unpack fitness history
        predictor_fitness_values, meta_fitness_values = zip(*fitness_history) if fitness_history else ([], [])
        generations = np.arange(len(fitness_history))
        
        # Include hierarchy level information in titles if provided
        hierarchy_info = ""
        if hierarchy_level is not None and total_levels is not None:
            hierarchy_info = f" (Level {hierarchy_level}/{total_levels})"
        
        # Clear all axes
        for ax in main_axes[:-1]:  # Skip the text area
            ax.clear()
        
        # Update fitness plot
        ax1.plot(generations, predictor_fitness_values, label="Predictor Fitness")
        ax1.plot(generations, meta_fitness_values, label="Meta-Predictor Fitness")
        ax1.axhline(y=global_best_fitness, color='r', linestyle='--', label=f"Global Best ({global_best_fitness:.4f})")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(f"Evolution Progress{hierarchy_info}")
        ax1.legend(fontsize='small')
        ax1.grid(True, alpha=0.3)
        
        # Update prediction vs actual plot
        x_values = np.linspace(0, 1, 100)
        actual_values = np.array([chaotic_function(x, r) for x in x_values])
        current_predicted_values = np.array([current_best_predictor.predict(x) for x in x_values])
        global_predicted_values = np.array([global_best_predictor.predict(x) for x in x_values])
        
        ax2.plot(x_values, actual_values, 'r-', label="Target", alpha=0.7)
        ax2.plot(x_values, current_predicted_values, 'b-', label=f"Current Best", alpha=0.7)
        ax2.plot(x_values, global_predicted_values, 'g-', label=f"Global Best", alpha=0.7)
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Output")
        ax2.set_title(f"Predictor Performance{hierarchy_info}")
        ax2.legend(fontsize='small')
        ax2.grid(True, alpha=0.3)
        
        # Update weights plot
        if len(weight_history) > 0:
            weight_history_array = np.array(weight_history)
            ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 0], label="Weight a (x²)")
            ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 1], label="Weight b (x)")
            ax3.plot(np.arange(len(weight_history)), weight_history_array[:, 2], label="Weight c (constant)")
            
            # Add horizontal lines for global best weights
            ax3.axhline(y=global_best_predictor.weights[0], color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=global_best_predictor.weights[1], color='g', linestyle='--', alpha=0.5)
            ax3.axhline(y=global_best_predictor.weights[2], color='b', linestyle='--', alpha=0.5)
            
            ax3.set_xlabel("Generation")
            ax3.set_ylabel("Weight Value")
            ax3.set_title(f"Weight Evolution{hierarchy_info}")
            ax3.legend(fontsize='small')
            ax3.grid(True, alpha=0.3)
        
        # Update fitness deviation plot
        if len(fitness_history) > 0:
            fitness_deviation = np.array([abs(p - m) for p, m in fitness_history])
            
            ax4.plot(generations, fitness_deviation, 'g-', label="Fitness Deviation")
            ax4.set_xlabel("Generation")
            ax4.set_ylabel("Absolute Deviation")
            ax4.set_title(f"Predictor vs Meta-Predictor Deviation{hierarchy_info}")
            
            # Add a moving average for trend visualization
            if len(fitness_deviation) > 10:
                window_size = min(50, len(fitness_deviation) // 5)
                moving_avg = np.convolve(fitness_deviation, np.ones(window_size)/window_size, mode='valid')
                ax4.plot(generations[window_size-1:], moving_avg, 'r-', 
                         label=f"Moving Avg (w={window_size})")
            
            ax4.legend(fontsize='small')
            ax4.grid(True, alpha=0.3)
        
        # Update error comparison plot
        current_error = np.array([abs(chaotic_function(x, r) - current_best_predictor.predict(x)) for x in x_values])
        global_error = np.array([abs(chaotic_function(x, r) - global_best_predictor.predict(x)) for x in x_values])
        
        ax5.plot(x_values, current_error, 'b-', label=f"Current Best Error", alpha=0.7)
        ax5.plot(x_values, global_error, 'g-', label=f"Global Best Error", alpha=0.7)
        
        current_mean_error = np.mean(current_error)
        global_mean_error = np.mean(global_error)
        ax5.axhline(y=current_mean_error, color='b', linestyle='--', alpha=0.5)
        ax5.axhline(y=global_mean_error, color='g', linestyle='--', alpha=0.5)
        
        ax5.set_xlabel("Input (x)")
        ax5.set_ylabel("Absolute Error")
        ax5.set_title(f"Error Comparison{hierarchy_info}")
        ax5.legend(fontsize='small')
        ax5.grid(True, alpha=0.3)
        
        # Update mutation rate plot
        if mutation_rate_history is not None and len(mutation_rate_history) > 0:
            ax6.plot(generations, mutation_rate_history, 'purple', label="Avg Mutation Rate")
            
            if len(mutation_rate_history) > 10:
                window_size = min(50, len(mutation_rate_history) // 5)
                moving_avg = np.convolve(mutation_rate_history, np.ones(window_size)/window_size, mode='valid')
                ax6.plot(generations[window_size-1:], moving_avg, 'orange', 
                         label=f"Moving Avg (w={window_size})")
            
            if len(mutation_rate_history) > 0:
                current_mr = mutation_rate_history[-1]
                ax6.axhline(y=current_mr, color='r', linestyle='--', alpha=0.5)
                
                min_mr = min(mutation_rate_history)
                max_mr = max(mutation_rate_history)
                
                if len(mutation_rate_history) > 0:
                    min_val = max(0, min(mutation_rate_history) * 0.8)
                    max_val = min(0.5, max(mutation_rate_history) * 1.2)
                    ax6.set_ylim(min_val, max_val)
            
            ax6.set_xlabel("Generation")
            ax6.set_ylabel("Mutation Rate")
            ax6.set_title(f"Mutation Rate Evolution{hierarchy_info}")
            ax6.legend(fontsize='small')
            ax6.grid(True, alpha=0.3)
        
        # Update chaos function phase space plot
        if chaos_values_history is not None and len(chaos_values_history) > 0:
            # Generate x values from 0 to 1
            x_values = np.linspace(0, 1, 100)
            
            # Calculate corresponding chaotic function values
            y_values = np.array([chaotic_function(x, r) for x in x_values])
            
            # Plot the chaotic function as a continuous line for reference
            ax7.plot(x_values, y_values, 'r-', alpha=0.3, label="Chaotic Function")
            
            # Generate a set of random x values to show the distribution
            num_points = min(200, len(chaos_values_history))  # Limit the number of points
            random_indices = np.random.choice(len(chaos_values_history), num_points, replace=False) if len(chaos_values_history) > num_points else np.arange(len(chaos_values_history))
            
            # Generate random x values between 0 and 1
            random_x_values = np.random.uniform(0, 1, num_points)
            
            # Get corresponding y values from the chaos history
            random_y_values = np.array([chaos_values_history[i] for i in random_indices])
            
            # Plot random points as scatter plot
            ax7.scatter(random_x_values, random_y_values, s=15, c='blue', alpha=0.6, label="Random Samples")
            
            # Add a diagonal line y=x for reference
            ax7.plot([0, 1], [0, 1], 'g--', alpha=0.5, label="y=x")
            
            # Add annotations for key statistics
            max_val = np.max(chaos_values_history)
            min_val = np.min(chaos_values_history)
            mean_val = np.mean(chaos_values_history)
            std_val = np.std(chaos_values_history)
            
            stats_text = (f"Chaos Parameter r: {r:.4f}\n"
                         f"Min: {min_val:.4f}\n"
                         f"Max: {max_val:.4f}\n"
                         f"Mean: {mean_val:.4f}\n"
                         f"Std Dev: {std_val:.4f}")
            
            ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8))
        
            ax7.set_xlabel("Input (x)")
            ax7.set_ylabel("Chaotic Function Output")
            ax7.set_title(f"Chaos Phase Space (r={r}){hierarchy_info}")
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.legend(fontsize='small', loc='lower right')
            ax7.grid(True, alpha=0.3)
        
        # Update cobweb plot for chaos function
        if chaos_values_history is not None and len(chaos_values_history) > 0:
            # Generate x values from 0 to 1
            x_values = np.linspace(0, 1, 100)
            
            # Calculate corresponding chaotic function values
            y_values = np.array([chaotic_function(x, r) for x in x_values])
            
            # Plot the chaotic function
            ax8.plot(x_values, y_values, 'r-', label=f"f(x), r={r:.2f}")
            
            # Plot the diagonal line y=x
            ax8.plot([0, 1], [0, 1], 'k--', label="y=x")
            
            # Generate a cobweb plot for a random starting point
            x0 = np.random.uniform(0.1, 0.9)  # Random starting point
            iterations = 30  # Number of iterations for the cobweb
            
            # Calculate the cobweb points
            cobweb_x = [x0]
            cobweb_y = [0]
            
            for i in range(iterations):
                # Calculate next y value (on the function)
                y = chaotic_function(cobweb_x[-1], r)
                cobweb_y.append(y)
                cobweb_x.append(cobweb_x[-1])
                
                # Move horizontally to the diagonal
                cobweb_y.append(y)
                cobweb_x.append(y)
            
            # Plot the cobweb
            ax8.plot(cobweb_x, cobweb_y, 'b-', alpha=0.7, label=f"Cobweb (x₀={x0:.2f})")
            
            # Mark the starting point
            ax8.plot(x0, 0, 'go', markersize=6, label="Start")
            
            ax8.set_xlabel("x")
            ax8.set_ylabel("f(x)")
            ax8.set_title(f"Cobweb Plot{hierarchy_info}")
            ax8.set_xlim(0, 1)
            ax8.set_ylim(0, 1)
            ax8.legend(fontsize='small', loc='upper right')
            ax8.grid(True, alpha=0.3)
        
        # Update statistics text area
        ax9.clear()
        ax9.axis('off')
        
        # Compile statistics text
        stats_text = f"Generation: {generation}\n\n"
        
        if hierarchy_level is not None and total_levels is not None:
            stats_text += f"Hierarchy Level: {hierarchy_level}/{total_levels}\n\n"
        
        stats_text += f"Parameters:\n"
        stats_text += f"- Chaos parameter (r): {r:.4f}\n"
        
        if hasattr(current_best_predictor, 'mutation_rate'):
            stats_text += f"- Current best mutation rate: {current_best_predictor.mutation_rate:.4f}\n"
        
        stats_text += f"\nCurrent Best (Gen {generation}):\n"
        stats_text += f"- Fitness: {predictor_fitness_values[-1]:.4f}\n"
        stats_text += f"- Equation: f(x) = {current_best_predictor.weights[0]:.4f}x² + "
        stats_text += f"{current_best_predictor.weights[1]:.4f}x + {current_best_predictor.weights[2]:.4f}\n"
        
        stats_text += f"\nGlobal Best (Gen {global_best_generation}):\n"
        stats_text += f"- Fitness: {global_best_fitness:.4f}\n"
        stats_text += f"- Equation: f(x) = {global_best_predictor.weights[0]:.4f}x² + "
        stats_text += f"{global_best_predictor.weights[1]:.4f}x + {global_best_predictor.weights[2]:.4f}\n"
        
        if chaos_values_history is not None and len(chaos_values_history) > 0:
            stats_text += f"\nChaos Statistics:\n"
            stats_text += f"- Min: {np.min(chaos_values_history):.4f}\n"
            stats_text += f"- Max: {np.max(chaos_values_history):.4f}\n"
            stats_text += f"- Mean: {np.mean(chaos_values_history):.4f}\n"
            stats_text += f"- Std Dev: {np.std(chaos_values_history):.4f}\n"
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Update the main figure
        main_fig.suptitle(f"XRAI Evolution Dashboard - Generation {generation}{hierarchy_info}", fontsize=16)
        
        # Adjust the spacing manually instead of using tight_layout
        main_fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.9, wspace=0.3, hspace=0.4)
        
        main_fig.canvas.draw_idle()
        
        # Also update individual figures for saving
        update_individual_figures(
            individual_figs_axes, generation, fitness_history, weight_history,
            current_best_predictor, global_best_predictor, global_best_fitness, 
            global_best_generation, r, current_error, global_error, 
            current_mean_error, global_mean_error, mutation_rate_history,
            chaos_values_history, hierarchy_info, chaotic_function
        )
        
        # Return the updated figures and axes
        return (main_fig, main_axes, *individual_figs_axes)
    except Exception as e:
        print(f"Error updating plots: {e}")
        return figures_axes  # Return unchanged to avoid further errors

def update_individual_figures(individual_figs_axes, generation, fitness_history, weight_history,
                             current_best_predictor, global_best_predictor, global_best_fitness, 
                             global_best_generation, r, current_error, global_error, 
                             current_mean_error, global_mean_error, mutation_rate_history,
                             chaos_values_history, hierarchy_info, chaotic_function):
    """Update individual figures for saving."""
    # Unpack individual figures and axes
    (fig1, ax1_save), (fig2, ax2_save), (fig3, ax3_save), (fig4, ax4_save), \
    (fig5, ax5_save), (fig6, ax6_save), (fig7, ax7_save), (fig8, ax8_save) = individual_figs_axes
    
    # Unpack fitness history
    predictor_fitness_values, meta_fitness_values = zip(*fitness_history) if fitness_history else ([], [])
    generations = np.arange(len(fitness_history))
    
    # Generate x values for plots
    x_values = np.linspace(0, 1, 100)
    actual_values = np.array([chaotic_function(x, r) for x in x_values])
    current_predicted_values = np.array([current_best_predictor.predict(x) for x in x_values])
    global_predicted_values = np.array([global_best_predictor.predict(x) for x in x_values])
    
    # Fitness history
    ax1_save.clear()
    ax1_save.plot(generations, predictor_fitness_values, label="Predictor Fitness")
    ax1_save.plot(generations, meta_fitness_values, label="Meta-Predictor Fitness")
    ax1_save.axhline(y=global_best_fitness, color='r', linestyle='--', label=f"Global Best ({global_best_fitness:.4f})")
    ax1_save.set_xlabel("Generation")
    ax1_save.set_ylabel("Fitness")
    ax1_save.set_title(f"Evolution Progress - Generation {generation}{hierarchy_info}")
    ax1_save.legend()
    ax1_save.grid(True, alpha=0.3)
    fig1.tight_layout()
    
    # Prediction vs actual
    ax2_save.clear()
    ax2_save.plot(x_values, actual_values, 'r-', label="Target (Chaotic Function)", alpha=0.7)
    ax2_save.plot(x_values, current_predicted_values, 'b-', label=f"Current Best (Gen {generation})", alpha=0.7)
    ax2_save.plot(x_values, global_predicted_values, 'g-', label=f"Global Best (Gen {global_best_generation})", alpha=0.7)
    ax2_save.set_xlabel("Input (x)")
    ax2_save.set_ylabel("Output")
    ax2_save.set_title(f"Predictor Performance: Current vs Global Best{hierarchy_info}")
    ax2_save.legend()
    ax2_save.grid(True, alpha=0.3)
    fig2.tight_layout()
    
    # Weight evolution
    ax3_save.clear()
    if len(weight_history) > 0:
        weight_history_array = np.array(weight_history)
        ax3_save.plot(np.arange(len(weight_history)), weight_history_array[:, 0], label="Weight a (x²)")
        ax3_save.plot(np.arange(len(weight_history)), weight_history_array[:, 1], label="Weight b (x)")
        ax3_save.plot(np.arange(len(weight_history)), weight_history_array[:, 2], label="Weight c (constant)")
        
        ax3_save.axhline(y=global_best_predictor.weights[0], color='r', linestyle='--', alpha=0.5, 
                       label=f"Global Best a={global_best_predictor.weights[0]:.4f}")
        ax3_save.axhline(y=global_best_predictor.weights[1], color='g', linestyle='--', alpha=0.5,
                       label=f"Global Best b={global_best_predictor.weights[1]:.4f}")
        ax3_save.axhline(y=global_best_predictor.weights[2], color='b', linestyle='--', alpha=0.5,
                       label=f"Global Best c={global_best_predictor.weights[2]:.4f}")
    
    ax3_save.set_xlabel("Generation")
    ax3_save.set_ylabel("Weight Value")
    ax3_save.set_title(f"Evolution of Best Predictor Weights - Generation {generation}{hierarchy_info}")
    ax3_save.legend()
    ax3_save.grid(True, alpha=0.3)
    fig3.tight_layout()
    
    # Fitness deviation
    ax4_save.clear()
    if len(fitness_history) > 0:
        fitness_deviation = np.array([abs(p - m) for p, m in fitness_history])
        ax4_save.plot(generations, fitness_deviation, 'g-', label="Fitness Deviation")
        
        if len(fitness_deviation) > 10:
            window_size = min(50, len(fitness_deviation) // 5)
            moving_avg = np.convolve(fitness_deviation, np.ones(window_size)/window_size, mode='valid')
            ax4_save.plot(generations[window_size-1:], moving_avg, 'r-', 
                         label=f"Moving Average (window={window_size})")
    
    ax4_save.set_xlabel("Generation")
    ax4_save.set_ylabel("Absolute Deviation")
    ax4_save.set_title(f"Predictor vs Meta-Predictor Fitness Deviation - Generation {generation}{hierarchy_info}")
    ax4_save.legend()
    ax4_save.grid(True, alpha=0.3)
    fig4.tight_layout()
    
    # Error comparison
    ax5_save.clear()
    ax5_save.plot(x_values, current_error, 'b-', label=f"Current Best Error (Gen {generation})", alpha=0.7)
    ax5_save.plot(x_values, global_error, 'g-', label=f"Global Best Error (Gen {global_best_generation})", alpha=0.7)
    
    ax5_save.axhline(y=current_mean_error, color='b', linestyle='--', alpha=0.5,
                   label=f"Current Mean Error: {current_mean_error:.4f}")
    ax5_save.axhline(y=global_mean_error, color='g', linestyle='--', alpha=0.5,
                   label=f"Global Mean Error: {global_mean_error:.4f}")
    
    ax5_save.set_xlabel("Input (x)")
    ax5_save.set_ylabel("Absolute Error")
    ax5_save.set_title(f"Error Comparison: Current vs Global Best{hierarchy_info}")
    ax5_save.legend()
    ax5_save.grid(True, alpha=0.3)
    fig5.tight_layout()
    
    # Mutation rate
    ax6_save.clear()
    if mutation_rate_history is not None and len(mutation_rate_history) > 0:
        ax6_save.plot(generations, mutation_rate_history, 'purple', label="Average Mutation Rate")
        
        if len(mutation_rate_history) > 10:
            window_size = min(50, len(mutation_rate_history) // 5)
            moving_avg = np.convolve(mutation_rate_history, np.ones(window_size)/window_size, mode='valid')
            ax6_save.plot(generations[window_size-1:], moving_avg, 'orange', 
                         label=f"Moving Average (window={window_size})")
        
        if len(mutation_rate_history) > 0:
            current_mr = mutation_rate_history[-1]
            ax6_save.axhline(y=current_mr, color='r', linestyle='--', alpha=0.5,
                           label=f"Current Rate: {current_mr:.4f}")
            
            min_mr = min(mutation_rate_history)
            max_mr = max(mutation_rate_history)
            min_idx = mutation_rate_history.index(min_mr)
            max_idx = mutation_rate_history.index(max_mr)
            
            ax6_save.annotate(f"Min: {min_mr:.4f}", 
                            xy=(min_idx, min_mr),
                            xytext=(min_idx, min_mr - 0.05),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=9)
            
            ax6_save.annotate(f"Max: {max_mr:.4f}", 
                            xy=(max_idx, max_mr),
                            xytext=(max_idx, max_mr + 0.05),
                            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                            fontsize=9)
            
            if len(mutation_rate_history) > 0:
                min_val = max(0, min(mutation_rate_history) * 0.8)
                max_val = min(0.5, max(mutation_rate_history) * 1.2)
                ax6_save.set_ylim(min_val, max_val)
    
    ax6_save.set_xlabel("Generation")
    ax6_save.set_ylabel("Mutation Rate")
    ax6_save.set_title(f"Mutation Rate Evolution - Generation {generation}{hierarchy_info}")
    ax6_save.legend()
    ax6_save.grid(True, alpha=0.3)
    fig6.tight_layout()
