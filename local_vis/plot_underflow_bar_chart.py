#!/usr/bin/env python3
"""
Underflow Bar Chart Plotter

Reads underflow analysis JSON files and creates bar charts by layer and pass type.
Usage: python plot_underflow_bar_chart.py --json-file hifp8_underflow.json
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_json_data(json_path: Path) -> dict:
    """Load JSON data from file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_underflow_bar_chart(json_data: dict, output_path: Path = None):
    """
    Create bar chart showing underflow percentages by layer and pass type.
    
    Args:
        json_data: Dictionary containing underflow analysis results
        output_path: Optional path to save the plot
    """
    # Extract data
    elem_format = json_data.get('elem_format', 'unknown')
    results = json_data.get('results', [])
    
    if not results:
        print("No data to plot")
        return
    
    # Organize data by layer and pass type
    layer_data = {}
    for result in results:
        layer = result['layer']
        pass_type = result['pass_type']
        
        if layer not in layer_data:
            layer_data[layer] = {'forward': None, 'backward': None}
        
        layer_data[layer][pass_type] = result['underflow_percentage']
    
    # Get sorted layers
    layers = sorted(layer_data.keys())
    
    # Prepare data for plotting
    forward_values = []
    backward_values = []
    for layer in layers:
        forward_values.append(layer_data[layer]['forward'] if layer_data[layer]['forward'] is not None else 0)
        backward_values.append(layer_data[layer]['backward'] if layer_data[layer]['backward'] is not None else 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(layers))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, forward_values, width, label='Forward Pass', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, backward_values, width, label='Backward Pass', 
                   color='#A23B72', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Underflow Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{elem_format.upper()} Underflow Analysis by Layer and Pass Type', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Bar chart saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot underflow bar chart from JSON data')
    parser.add_argument('--json-file', required=True, help='Path to JSON file (e.g., hifp8_underflow.json)')
    parser.add_argument('--output', default=None, help='Output image path (default: same as JSON with .png extension)')
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Load data
    json_data = load_json_data(json_path)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.with_suffix('.png')
    
    # Create plot
    plot_underflow_bar_chart(json_data, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())

