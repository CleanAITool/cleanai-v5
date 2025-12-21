"""
Test Scenario TS1 - Master Script: Run All Experiments
======================================================
This script runs all three test scenario scripts in sequence.

Usage:
    python TS1_run_all.py [--skip-prepare] [--skip-coverage] [--skip-wanda]
    
Options:
    --skip-prepare    Skip model preparation (Script 1)
    --skip-coverage   Skip coverage pruning (Script 2)
    --skip-wanda      Skip WANDA pruning (Script 3)
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_script(script_name: str, description: str) -> bool:
    """
    Run a script and return success status.
    
    Args:
        script_name: Name of the script to run
        description: Description of the script
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*100)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("="*100 + "\n")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            cwd=Path(__file__).parent
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {script_name} completed successfully in {elapsed/60:.2f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {script_name} failed after {elapsed/60:.2f} minutes")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {script_name} interrupted by user")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run all TS1 test scenario scripts')
    parser.add_argument('--skip-prepare', action='store_true', help='Skip model preparation')
    parser.add_argument('--skip-coverage', action='store_true', help='Skip coverage pruning')
    parser.add_argument('--skip-wanda', action='store_true', help='Skip WANDA pruning')
    args = parser.parse_args()
    
    print("\n" + "#"*100)
    print("# TEST SCENARIO TS1 - MASTER SCRIPT")
    print("#"*100)
    print("\nThis will run all test scenario scripts in sequence:")
    print("  1. TS1_01_prepare_model.py    - Model preparation & fine-tuning")
    print("  2. TS1_02_coverage_pruning.py - Neuron Coverage pruning")
    print("  3. TS1_03_wanda_pruning.py    - WANDA pruning")
    
    if args.skip_prepare:
        print("\n⚠ Skipping model preparation (--skip-prepare)")
    if args.skip_coverage:
        print("\n⚠ Skipping coverage pruning (--skip-coverage)")
    if args.skip_wanda:
        print("\n⚠ Skipping WANDA pruning (--skip-wanda)")
    
    # Confirm execution
    response = input("\nContinue? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return
    
    total_start = time.time()
    results = {}
    
    # Script 1: Model Preparation
    if not args.skip_prepare:
        success = run_script(
            "TS1_01_prepare_model.py",
            "Download pretrained model, adapt to CIFAR-10, and fine-tune"
        )
        results['prepare'] = success
        
        if not success:
            print("\n✗ Model preparation failed. Cannot continue.")
            return
    else:
        print("\n⚠ Skipping model preparation")
        results['prepare'] = 'skipped'
    
    # Script 2: Coverage Pruning
    if not args.skip_coverage:
        success = run_script(
            "TS1_02_coverage_pruning.py",
            "Apply Neuron Coverage pruning and fine-tune"
        )
        results['coverage'] = success
    else:
        print("\n⚠ Skipping coverage pruning")
        results['coverage'] = 'skipped'
    
    # Script 3: WANDA Pruning
    if not args.skip_wanda:
        success = run_script(
            "TS1_03_wanda_pruning.py",
            "Apply WANDA pruning and fine-tune"
        )
        results['wanda'] = success
    else:
        print("\n⚠ Skipping WANDA pruning")
        results['wanda'] = 'skipped'
    
    # Print summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*100)
    print("EXECUTION SUMMARY")
    print("="*100)
    print(f"\n{'Script':<40} {'Status':<20}")
    print("-" * 60)
    
    for name, status in results.items():
        if status == 'skipped':
            status_str = "⚠ SKIPPED"
        elif status:
            status_str = "✓ SUCCESS"
        else:
            status_str = "✗ FAILED"
        
        print(f"{name.capitalize():<40} {status_str:<20}")
    
    print("-" * 60)
    print(f"\nTotal execution time: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours)")
    
    # Check if all succeeded
    all_success = all(v is True or v == 'skipped' for v in results.values())
    
    if all_success:
        print("\n✓ All scripts completed successfully!")
        print("\nGenerated outputs:")
        print("  - Checkpoints: C:\\source\\checkpoints\\TS1\\")
        print("  - Checkpoints: C:\\source\\checkpoints\\TS1_Coverage_ResNet18_CIFAR10\\")
        print("  - Checkpoints: C:\\source\\checkpoints\\TS1_Wanda_ResNet18_CIFAR10\\")
        print("  - Reports: See reports/ subdirectories in checkpoint folders")
    else:
        print("\n✗ Some scripts failed. Please check the logs above.")
    
    print("\n" + "#"*100)
    print("# MASTER SCRIPT COMPLETED")
    print("#"*100 + "\n")


if __name__ == '__main__':
    main()
