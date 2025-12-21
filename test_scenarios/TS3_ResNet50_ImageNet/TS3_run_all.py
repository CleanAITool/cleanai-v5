"""
Test Scenario TS3 - Run All Scripts
Executes all test scenario scripts in sequence.
"""

import os
import sys
import subprocess
import time

# Configuration
CONFIG = {
    'test_scenario': 'TS3',
    'script_dir': os.path.dirname(__file__),
    'scripts': [
        'TS3_01_prepare_model.py',
        'TS3_02_coverage_pruning.py',
        'TS3_03_wanda_pruning.py',
        'TS3_compare_results.py'
    ]
}

def run_script(script_name):
    """Run a single script"""
    script_path = os.path.join(CONFIG['script_dir'], script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False
    
    print("\n" + "="*100)
    print(f"RUNNING: {script_name}")
    print("="*100 + "\n")
    
    start_time = time.time()
    
    try:
        # Run script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=CONFIG['script_dir'],
            check=True
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*100)
        print(f"✓ {script_name} completed successfully")
        print(f"  Elapsed time: {elapsed_time:.2f} seconds")
        print("="*100)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*100)
        print(f"✗ {script_name} failed with error code {e.returncode}")
        print(f"  Elapsed time: {elapsed_time:.2f} seconds")
        print("="*100)
        
        return False
    except KeyboardInterrupt:
        print("\n\n" + "="*100)
        print("Execution interrupted by user")
        print("="*100)
        return False
    except Exception as e:
        print("\n" + "="*100)
        print(f"✗ Unexpected error running {script_name}: {e}")
        print("="*100)
        return False

def main():
    """Main execution function"""
    print("\n" + "="*100)
    print(f"TEST SCENARIO {CONFIG['test_scenario']}: RUN ALL SCRIPTS")
    print("="*100)
    print(f"\nTotal scripts to run: {len(CONFIG['scripts'])}")
    print("\nScripts:")
    for i, script in enumerate(CONFIG['scripts'], 1):
        print(f"  {i}. {script}")
    print("\n" + "="*100)
    
    # Confirmation
    response = input("\nProceed with running all scripts? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Execution cancelled by user.")
        return
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Run each script
    for i, script in enumerate(CONFIG['scripts'], 1):
        print(f"\n\n{'='*100}")
        print(f"SCRIPT {i}/{len(CONFIG['scripts'])}: {script}")
        print('='*100)
        
        success = run_script(script)
        results.append((script, success))
        
        if not success:
            print(f"\n{'='*100}")
            print(f"STOPPING EXECUTION: {script} failed")
            print('='*100)
            
            response = input("\nContinue with remaining scripts? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                break
    
    # Print summary
    total_elapsed_time = time.time() - total_start_time
    
    print("\n\n" + "="*100)
    print("EXECUTION SUMMARY")
    print("="*100)
    
    print(f"\nTotal elapsed time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)")
    print(f"\nScripts executed: {len(results)}/{len(CONFIG['scripts'])}")
    
    print("\nResults:")
    for script, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {script}")
    
    # Overall status
    all_success = all(success for _, success in results)
    
    if all_success and len(results) == len(CONFIG['scripts']):
        print("\n" + "="*100)
        print("✓ ALL SCRIPTS COMPLETED SUCCESSFULLY")
        print("="*100 + "\n")
    else:
        print("\n" + "="*100)
        print("✗ SOME SCRIPTS FAILED OR WERE NOT EXECUTED")
        print("="*100 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
