import pygame
import sys
import numpy as np
import argparse
import threading
import subprocess
import time
from environment import GridWorld

def run_agent(agent_type, death_scenario=False, cell_size=25):
    """
    Run a specific agent type in a separate process.
    
    Args:
        agent_type: 'rule_based' or 'model_based'
        death_scenario: Whether to run the death scenario
        cell_size: Size of grid cells in pixels
    """
    cmd = [
        "python", "main.py",
        "--mode", "ai",
        "--agent-type", agent_type,
        "--cell-size", str(cell_size)
    ]
    
    if death_scenario:
        cmd.append("--death")
    
    # Run the command as a separate process
    subprocess.Popen(cmd)

def main():
    """Run both agents side by side for comparison."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare rule-based and model-based agents')
    parser.add_argument('--death', action='store_true', 
                        help='Run death scenario')
    parser.add_argument('--cell-size', type=int, default=20,
                        help='Size of grid cells in pixels (smaller for side-by-side)')
    args = parser.parse_args()
    
    print("Starting agent comparison...")
    print("Rule-based agent will appear in one window, model-based in another.")
    print("Press Esc in either window to close it.")
    print("Close both windows to end the comparison.")
    
    # Start the rule-based agent
    run_agent('rule_based', args.death, args.cell_size)
    
    # Wait a moment to prevent potential conflicts with window creation
    time.sleep(1)
    
    # Start the model-based agent
    run_agent('model_based', args.death, args.cell_size)
    
    print("Both agents are running. Compare their behavior.")

if __name__ == "__main__":
    main() 