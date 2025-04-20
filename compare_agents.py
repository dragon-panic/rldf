import pygame
import sys
import numpy as np
import argparse
import threading
import subprocess
import time
import os
import sys
from environment import GridWorld

def run_agent(agent_type, death_scenario=False, cell_size=25):
    """
    Run a specific agent type in a separate process.
    
    Args:
        agent_type: 'rule_based' or 'model_based'
        death_scenario: Whether to run the death scenario
        cell_size: Size of grid cells in pixels
    """
    # Use the same Python interpreter that's running this script
    python_executable = sys.executable
    
    cmd = [
        python_executable, "main.py",
        "--mode", "ai",
        "--agent-type", agent_type,
        "--cell-size", str(cell_size)
    ]
    
    if death_scenario:
        cmd.append("--death")
    
    try:
        # Run the command as a separate process, inherit the environment
        process = subprocess.Popen(cmd, env=os.environ.copy(), 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 universal_newlines=True)
        
        # Print the agent type for identification
        print(f"Started {agent_type} agent (PID: {process.pid})")
        
        # Return the process so we can monitor it
        return process
    except Exception as e:
        print(f"Error starting {agent_type} agent: {e}")
        return None

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
    print(f"Using Python interpreter: {sys.executable}")
    print("Rule-based agent will appear in one window, model-based in another.")
    print("Press Esc in either window to close it.")
    print("Close both windows to end the comparison.")
    
    # Start the rule-based agent
    rb_process = run_agent('rule_based', args.death, args.cell_size)
    
    # Wait a moment to prevent potential conflicts with window creation
    time.sleep(2)
    
    # Start the model-based agent
    mb_process = run_agent('model_based', args.death, args.cell_size)
    
    print("Both agents are running. Compare their behavior.")
    
    # Wait for processes to complete
    try:
        # Monitor both processes, but don't block the main thread
        while True:
            if rb_process and rb_process.poll() is not None:
                print("Rule-based agent has exited.")
                # Check for any error output
                stderr = rb_process.stderr.read()
                if stderr:
                    print(f"Rule-based agent error: {stderr}")
                rb_process = None
            
            if mb_process and mb_process.poll() is not None:
                print("Model-based agent has exited.")
                # Check for any error output
                stderr = mb_process.stderr.read()
                if stderr:
                    print(f"Model-based agent error: {stderr}")
                mb_process = None
            
            # If both processes have exited, break the loop
            if rb_process is None and mb_process is None:
                print("Both agents have exited. Comparison complete.")
                break
            
            # Sleep to prevent high CPU usage
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C to gracefully terminate the processes
        print("\nInterrupted by user. Terminating processes...")
        if rb_process:
            rb_process.terminate()
        if mb_process:
            mb_process.terminate()

if __name__ == "__main__":
    main() 