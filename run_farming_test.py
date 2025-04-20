#!/usr/bin/env python
import argparse
import logging
import sys
from test_improved_farming import test_improved_farming, TRACE_LEVEL

def main():
    """Run the improved farming test with configurable logging level."""
    parser = argparse.ArgumentParser(description="Run the improved farming test")
    parser.add_argument(
        "--farming-log-level", 
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=300,
        help="Number of steps to run (default: 300)"
    )
    parser.add_argument(
        "--no-visualize", 
        action="store_true",
        help="Disable visualization entirely"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run with mock visualization (no windows) but still log visualization events"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots at the end (implied by --visualize)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.farming_log_level == "TRACE":
        log_level = TRACE_LEVEL
    else:
        log_level = getattr(logging, args.farming_log_level)
    
    # If using headless mode, we need to set up the mocks
    if args.headless:
        # Make sure we can find the mock modules
        sys.path.insert(0, '.')
        # Mock modules for headless testing
        sys.modules['pygame'] = __import__('mock_pygame').pygame
        # Force non-interactive matplotlib
        import matplotlib
        matplotlib.use('Agg')
    
    # Run the test with the specified parameters
    visualize = not args.no_visualize
    
    # Add show-plots flag to sys.argv if needed
    if args.show_plots and '--show-plots' not in sys.argv:
        sys.argv.append('--show-plots')
    
    test_improved_farming(
        steps=args.steps,
        visualize=visualize,
        log_level=log_level
    )

if __name__ == "__main__":
    main() 