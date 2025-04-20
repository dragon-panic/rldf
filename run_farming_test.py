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
        help="Disable visualization"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.farming_log_level == "TRACE":
        log_level = TRACE_LEVEL
    else:
        log_level = getattr(logging, args.farming_log_level)
    
    # Run the test with the specified parameters
    test_improved_farming(
        steps=args.steps,
        visualize=not args.no_visualize,
        log_level=log_level
    )

if __name__ == "__main__":
    main() 