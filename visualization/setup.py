"""
Setup functions for visualization.
"""
import matplotlib.pyplot as plt

def setup_visualization():
    """Set up and return the visualization figures and axes."""
    # Enable interactive mode for matplotlib
    plt.ion()
    
    try:
        # Create a single figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Create a grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Create axes for each subplot
        ax1 = fig.add_subplot(gs[0, 0])  # Fitness history
        ax2 = fig.add_subplot(gs[0, 1])  # Prediction vs actual
        ax3 = fig.add_subplot(gs[0, 2])  # Weight evolution
        ax4 = fig.add_subplot(gs[1, 0])  # Fitness deviation
        ax5 = fig.add_subplot(gs[1, 1])  # Error comparison
        ax6 = fig.add_subplot(gs[1, 2])  # Mutation rate evolution
        ax7 = fig.add_subplot(gs[2, 0])  # Chaos function phase space
        ax8 = fig.add_subplot(gs[2, 1])  # Chaos function cobweb plot
        
        # Add a text area for statistics in the bottom right
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')  # Turn off axis for text display
        
        # Set window close event to handle gracefully
        fig.canvas.mpl_connect('close_event', lambda event: None)
        
        # Store individual figures for saving purposes
        fig1 = plt.figure(figsize=(10, 5))
        ax1_save = fig1.add_subplot(111)
        
        fig2 = plt.figure(figsize=(10, 5))
        ax2_save = fig2.add_subplot(111)
        
        fig3 = plt.figure(figsize=(10, 5))
        ax3_save = fig3.add_subplot(111)
        
        fig4 = plt.figure(figsize=(10, 5))
        ax4_save = fig4.add_subplot(111)
        
        fig5 = plt.figure(figsize=(10, 5))
        ax5_save = fig5.add_subplot(111)
        
        fig6 = plt.figure(figsize=(10, 5))
        ax6_save = fig6.add_subplot(111)
        
        fig7 = plt.figure(figsize=(10, 5))
        ax7_save = fig7.add_subplot(111)
        
        fig8 = plt.figure(figsize=(10, 5))
        ax8_save = fig8.add_subplot(111)
        
        # Hide all individual figures (they're just for saving)
        for fig_save in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]:
            fig_save.canvas.manager.set_window_title('Hidden Figure')
            try:
                # Try to minimize or hide the window if possible
                fig_save.canvas.manager.window.withdraw()
            except:
                pass
        
        # Return the main figure and all axes, plus individual figures for saving
        return (fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9), 
                (fig1, ax1_save), (fig2, ax2_save), (fig3, ax3_save), 
                (fig4, ax4_save), (fig5, ax5_save), (fig6, ax6_save), 
                (fig7, ax7_save), (fig8, ax8_save))
    
    except Exception as e:
        print(f"Error setting up visualization: {e}")
        print("Falling back to basic visualization...")
        
        # Create basic figures for fallback
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 5))  # Fitness history
            fig2, ax2 = plt.subplots(figsize=(10, 5))  # Prediction vs actual
            fig3, ax3 = plt.subplots(figsize=(10, 5))  # Weight evolution
            fig4, ax4 = plt.subplots(figsize=(10, 5))  # Fitness deviation
            fig5, ax5 = plt.subplots(figsize=(10, 5))  # Error comparison
            
            # Set window close events to handle gracefully
            for fig in [fig1, fig2, fig3, fig4, fig5]:
                fig.canvas.mpl_connect('close_event', lambda event: None)
            
            return (fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4), (fig5, ax5)
        except Exception as e2:
            print(f"Error setting up fallback visualization: {e2}")
            print("Running without visualization...")
            return None 
