import subprocess
import os
import sys

def run_test(test_path):
    """Run a test and check for warnings."""
    print(f"Running test: {test_path}")
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v"],
        capture_output=True,
        text=True
    )
    
    stdout = result.stdout
    stderr = result.stderr
    
    print("\nOutput:")
    print(stdout)
    
    if "PytestReturnNotNoneWarning" in stdout or "PytestReturnNotNoneWarning" in stderr:
        print("\nWarning found! Test still needs to be fixed.")
        return False
    else:
        print("\nNo warnings found! Test is fixed.")
        return True

def main():
    """Run all tests that previously had warnings."""
    tests = [
        "test_resources.py::test_water_proximity",
        "test_resources.py::test_fertility_regeneration",
        "test_resources.py::test_plant_growth",
        "test_rule_based_agent.py::test_farming_capability",
        "test_rule_based_agent.py::test_agent_death",
        "test_agent.py::test_agent_initialization",
        "test_agent.py::test_basic_movement",
        "test_agent.py::test_action_effects"
    ]
    
    all_fixed = True
    
    for test in tests:
        if not run_test(test):
            all_fixed = False
        print("\n" + "-" * 80 + "\n")
    
    if all_fixed:
        print("All tests are fixed! No more warnings.")
    else:
        print("Some tests still have warnings. Please check the output above.")

if __name__ == "__main__":
    main() 