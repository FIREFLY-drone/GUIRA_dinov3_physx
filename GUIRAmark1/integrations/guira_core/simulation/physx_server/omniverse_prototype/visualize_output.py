#!/usr/bin/env python3
"""
Visualize PhysX prototype output GeoJSON.

Creates a matplotlib animation showing fire perimeter evolution.
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Error: numpy required. Install with: pip install numpy matplotlib")
    sys.exit(1)


def visualize_geojson(geojson_path: str, output_gif: str = None):
    """Visualize fire perimeter evolution from GeoJSON.
    
    Args:
        geojson_path: Path to GeoJSON file
        output_gif: Optional path to save animation as GIF
    """
    print(f"Loading GeoJSON: {geojson_path}")
    
    with open(geojson_path) as f:
        data = json.load(f)
    
    features = data['features']
    print(f"Loaded {len(features)} timesteps")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all perimeters
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, len(features)))
    
    for i, feature in enumerate(features):
        coords = feature['geometry']['coordinates'][0]
        x = [p[0] for p in coords]
        y = [p[1] for p in coords]
        
        time_sec = feature['properties']['time_seconds']
        num_cells = feature['properties']['num_cells']
        
        ax.plot(x, y, color=colors[i], linewidth=2, 
                label=f"t={time_sec:.0f}s ({num_cells} cells)")
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Fire Perimeter Evolution (PhysX Prototype)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend (only show subset to avoid clutter)
    handles, labels = ax.get_legend_handles_labels()
    step = max(1, len(handles) // 5)
    ax.legend(handles[::step], labels[::step], loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    if output_gif:
        print(f"Saving static visualization: {output_gif.replace('.gif', '.png')}")
        plt.savefig(output_gif.replace('.gif', '.png'), dpi=150)
    else:
        plt.savefig('fire_perimeter_evolution.png', dpi=150)
        print("Saved: fire_perimeter_evolution.png")
    
    # Create animation
    print("\nCreating animation...")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 10))
    
    def animate(frame):
        ax_anim.clear()
        
        # Plot all previous perimeters (faded)
        for i in range(frame + 1):
            feature = features[i]
            coords = feature['geometry']['coordinates'][0]
            x = [p[0] for p in coords]
            y = [p[1] for p in coords]
            
            alpha = 0.3 if i < frame else 1.0
            linewidth = 1 if i < frame else 3
            color = colors[i]
            
            ax_anim.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
            
            if i == frame:
                # Fill current perimeter
                ax_anim.fill(x, y, color=color, alpha=0.2)
        
        # Add title with current timestep
        feature = features[frame]
        time_sec = feature['properties']['time_seconds']
        num_cells = feature['properties']['num_cells']
        ax_anim.set_title(
            f'Fire Spread - t={time_sec:.1f}s ({num_cells} cells burning)',
            fontsize=14, fontweight='bold'
        )
        
        ax_anim.set_xlabel('X (meters)', fontsize=12)
        ax_anim.set_ylabel('Y (meters)', fontsize=12)
        ax_anim.grid(True, alpha=0.3)
        ax_anim.set_aspect('equal')
        
        # Set fixed axis limits
        ax_anim.set_xlim(-120, 120)
        ax_anim.set_ylim(-120, 120)
        
        return []
    
    anim = animation.FuncAnimation(
        fig_anim, animate, frames=len(features), 
        interval=500, blit=True, repeat=True
    )
    
    if output_gif:
        print(f"Saving animation: {output_gif}")
        anim.save(output_gif, writer='pillow', fps=2)
        print(f"✓ Animation saved: {output_gif}")
    else:
        output_path = 'fire_spread_animation.gif'
        print(f"Saving animation: {output_path}")
        anim.save(output_path, writer='pillow', fps=2)
        print(f"✓ Animation saved: {output_path}")
    
    plt.close('all')
    
    print("\n✓ Visualization complete!")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_output.py <geojson_path> [output_gif]")
        print("\nExample:")
        print("  python visualize_output.py samples/physx/prototype_output.geojson samples/physx/fire_evolution.gif")
        sys.exit(1)
    
    geojson_path = sys.argv[1]
    output_gif = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(geojson_path).exists():
        print(f"Error: GeoJSON file not found: {geojson_path}")
        sys.exit(1)
    
    try:
        visualize_geojson(geojson_path, output_gif)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
