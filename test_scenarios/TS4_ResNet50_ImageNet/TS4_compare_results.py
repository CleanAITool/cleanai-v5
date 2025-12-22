"""
Test Scenario TS4 - Compare Results
Compares results from all pruning methods (Neuron Coverage, Wanda, Magnitude, and Taylor).
"""

import os
import sys
import json
from tabulate import tabulate

# Configuration
CONFIG = {
    'test_scenario': 'TS4',
    'results_dir': os.path.dirname(__file__),
    'results_file': 'TS4_Results.json'
}

def load_results():
    """Load results from JSON file"""
    results_file = os.path.join(CONFIG['results_dir'], CONFIG['results_file'])
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found: {results_file}")
        print("Please run the test scenarios first:")
        print("  1. TS4_01_prepare_model.py")
        print("  2. TS4_02_coverage_pruning.py")
        print("  3. TS4_03_wanda_pruning.py")
        print("  4. TS4_04_magnitude_pruning.py")
        print("  5. TS4_05_taylor_pruning.py")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def print_summary_table(results):
    """Print comprehensive comparison table"""
    print("\n" + "="*110)
    print("TEST SCENARIO TS4: COMPREHENSIVE COMPARISON")
    print("="*110)
    
    # Check which results are available
    has_prepare = 'prepare_model' in results
    has_coverage = 'coverage_pruning' in results
    has_wanda = 'wanda_pruning' in results
    has_magnitude = 'magnitude_pruning' in results
    has_taylor = 'taylor_pruning' in results
    
    if not any([has_prepare, has_coverage, has_wanda, has_magnitude, has_taylor]):
        print("No results available. Please run the test scenarios first.")
        return
    
    # Build comparison table
    table_data = [["Metric", "Original (Fine-tuned)"]]
    
    if has_coverage:
        table_data[0].append("Neuron Coverage")
    if has_wanda:
        table_data[0].append("Wanda")
    if has_magnitude:
        table_data[0].append("Magnitude (TP)")
    if has_taylor:
        table_data[0].append("Taylor (TP)")
    
    # Accuracy
    accuracy_row = ["Accuracy (%)"]
    if has_prepare:
        accuracy_row.append(f"{results['prepare_model']['finetuned']['accuracy']:.2f}")
    elif has_coverage:
        accuracy_row.append(f"{results['coverage_pruning']['original']['accuracy']:.2f}")
    elif has_wanda:
        accuracy_row.append(f"{results['wanda_pruning']['original']['accuracy']:.2f}")
    
    if has_coverage:
        accuracy_row.append(f"{results['coverage_pruning']['final']['accuracy']:.2f}")
    if has_wanda:
        accuracy_row.append(f"{results['wanda_pruning']['final']['accuracy']:.2f}")
    if has_magnitude:
        accuracy_row.append(f"{results['magnitude_pruning']['final']['accuracy']:.2f}")
    if has_taylor:
        accuracy_row.append(f"{results['taylor_pruning']['final']['accuracy']:.2f}")
    
    table_data.append(accuracy_row)
    
    # Size
    size_row = ["Size (MB)"]
    if has_prepare:
        size_row.append(f"{results['prepare_model']['finetuned']['size_mb']:.2f}")
    elif has_coverage:
        size_row.append(f"{results['coverage_pruning']['original']['size_mb']:.2f}")
    elif has_wanda:
        size_row.append(f"{results['wanda_pruning']['original']['size_mb']:.2f}")
    
    if has_coverage:
        size_row.append(f"{results['coverage_pruning']['final']['size_mb']:.2f}")
    if has_wanda:
        size_row.append(f"{results['wanda_pruning']['final']['size_mb']:.2f}")
    if has_magnitude:
        size_row.append(f"{results['magnitude_pruning']['final']['size_mb']:.2f}")
    if has_taylor:
        size_row.append(f"{results['taylor_pruning']['final']['size_mb']:.2f}")
    
    table_data.append(size_row)
    
    # Inference Time
    inference_row = ["Inference Time (ms)"]
    if has_prepare:
        inference_row.append(f"{results['prepare_model']['finetuned']['inference_time_ms']:.4f}")
    elif has_coverage:
        inference_row.append(f"{results['coverage_pruning']['original']['inference_time_ms']:.4f}")
    elif has_wanda:
        inference_row.append(f"{results['wanda_pruning']['original']['inference_time_ms']:.4f}")
    
    if has_coverage:
        inference_row.append(f"{results['coverage_pruning']['final']['inference_time_ms']:.4f}")
    if has_wanda:
        inference_row.append(f"{results['wanda_pruning']['final']['inference_time_ms']:.4f}")
    if has_magnitude:
        inference_row.append(f"{results['magnitude_pruning']['final']['inference_time_ms']:.4f}")
    if has_taylor:
        inference_row.append(f"{results['taylor_pruning']['final']['inference_time_ms']:.4f}")
    
    table_data.append(inference_row)
    
    # FLOPs
    has_flops = False
    if has_prepare and results['prepare_model']['finetuned']['flops'] > 0:
        has_flops = True
    elif has_coverage and results['coverage_pruning']['original']['flops'] > 0:
        has_flops = True
    elif has_wanda and results['wanda_pruning']['original']['flops'] > 0:
        has_flops = True
    
    if has_flops:
        flops_row = ["FLOPs (GFLOPs)"]
        if has_prepare:
            flops_row.append(f"{results['prepare_model']['finetuned']['flops']/1e9:.2f}")
        elif has_coverage:
            flops_row.append(f"{results['coverage_pruning']['original']['flops']/1e9:.2f}")
        elif has_wanda:
            flops_row.append(f"{results['wanda_pruning']['original']['flops']/1e9:.2f}")
        
        if has_coverage:
            flops_row.append(f"{results['coverage_pruning']['final']['flops']/1e9:.2f}")
        if has_wanda:
            flops_row.append(f"{results['wanda_pruning']['final']['flops']/1e9:.2f}")
        if has_magnitude:
            flops_row.append(f"{results['magnitude_pruning']['final']['flops']/1e9:.2f}")
        if has_taylor:
            flops_row.append(f"{results['taylor_pruning']['final']['flops']/1e9:.2f}")
        
        table_data.append(flops_row)
    
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    
    # Print percentage changes
    print("\n" + "="*120)
    print("PERCENTAGE CHANGES (compared to original)")
    print("="*120)
    
    change_data = [["Metric"]]
    if has_coverage:
        change_data[0].append("Neuron Coverage")
    # Get original values
    if has_prepare:
        orig_acc = results['prepare_model']['finetuned']['accuracy']
        orig_size = results['prepare_model']['finetuned']['size_mb']
        orig_time = results['prepare_model']['finetuned']['inference_time_ms']
        orig_flops = results['prepare_model']['finetuned']['flops']
    elif has_coverage:
        orig_acc = results['coverage_pruning']['original']['accuracy']
        orig_size = results['coverage_pruning']['original']['size_mb']
        orig_time = results['coverage_pruning']['original']['inference_time_ms']
        orig_flops = results['coverage_pruning']['original']['flops']
    elif has_wanda:
        orig_acc = results['wanda_pruning']['original']['accuracy']
        orig_size = results['wanda_pruning']['original']['size_mb']
        orig_time = results['wanda_pruning']['original']['inference_time_ms']
        orig_flops = results['wanda_pruning']['original']['flops']
    
    # Accuracy change
    acc_change_row = ["Accuracy Change (%)"]
    if has_coverage:
        cov_acc = results['coverage_pruning']['final']['accuracy']
        acc_change_row.append(f"{cov_acc - orig_acc:+.2f} ({(cov_acc/orig_acc - 1)*100:+.2f}%)")
    if has_wanda:
        wan_acc = results['wanda_pruning']['final']['accuracy']
        acc_change_row.append(f"{wan_acc - orig_acc:+.2f} ({(wan_acc/orig_acc - 1)*100:+.2f}%)")
    if has_magnitude:
        mag_acc = results['magnitude_pruning']['final']['accuracy']
        acc_change_row.append(f"{mag_acc - orig_acc:+.2f} ({(mag_acc/orig_acc - 1)*100:+.2f}%)")
    if has_taylor:
        tay_acc = results['taylor_pruning']['final']['accuracy']
        acc_change_row.append(f"{tay_acc - orig_acc:+.2f} ({(tay_acc/orig_acc - 1)*100:+.2f}%)")
    change_data.append(acc_change_row)
    
    # Size reduction
    size_change_row = ["Size Reduction (%)"]
    if has_coverage:
        cov_size = results['coverage_pruning']['final']['size_mb']
        size_change_row.append(f"{cov_size - orig_size:+.2f} MB ({(cov_size/orig_size - 1)*100:+.2f}%)")
    if has_wanda:
        wan_size = results['wanda_pruning']['final']['size_mb']
        size_change_row.append(f"{wan_size - orig_size:+.2f} MB ({(wan_size/orig_size - 1)*100:+.2f}%)")
    if has_magnitude:
        mag_size = results['magnitude_pruning']['final']['size_mb']
        size_change_row.append(f"{mag_size - orig_size:+.2f} MB ({(mag_size/orig_size - 1)*100:+.2f}%)")
    if has_taylor:
        tay_size = results['taylor_pruning']['final']['size_mb']
        size_change_row.append(f"{tay_size - orig_size:+.2f} MB ({(tay_size/orig_size - 1)*100:+.2f}%)")
    change_data.append(size_change_row)
    if has_wanda:
        wan_acc = results['wanda_pruning']['final']['accuracy']
        acc_change_row.append(f"{wan_acc - orig_acc:+.2f} ({(wan_acc/orig_acc - 1)*100:+.2f}%)")
    change_data.append(acc_change_row)
    
    # Size reduction
    size_change_row = ["Size Reduction (%)"]
    if has_magnitude:
        mag_time = results['magnitude_pruning']['final']['inference_time_ms']
        time_change_row.append(f"{mag_time - orig_time:+.4f} ms ({(mag_time/orig_time - 1)*100:+.2f}%)")
    if has_coverage:
        cov_size = results['coverage_pruning']['final']['size_mb']
        size_change_row.append(f"{cov_size - orig_size:+.2f} MB ({(cov_size/orig_size - 1)*100:+.2f}%)")
    if has_wanda:
        wan_size = results['wanda_pruning']['final']['size_mb']
        size_change_row.append(f"{wan_size - orig_size:+.2f} MB ({(wan_size/orig_size - 1)*100:+.2f}%)")
    change_data.append(size_change_row)
    
    # Inif has_magnitude:
            mag_flops = results['magnitude_pruning']['final']['flops']
            flops_change_row.append(f"{(mag_flops - orig_flops)/1e9:+.2f} G ({(mag_flops/orig_flops - 1)*100:+.2f}%)")
        change_data.append(flops_change_row)
    
    print(tabulate(change_data, headers="firstrow", tablefmt="grid"))
    
    # Print summary insights
    print("\n" + "="*110)
    print("SUMMARY INSIGHTS")
    print("="*110)
    
    methods_available = []
    if has_coverage:
        methods_available.append(('coverage', 'Neuron Coverage', results['coverage_pruning']['final']['accuracy']))
    if has_wanda:
        methods_available.append(('wanda', 'Wanda', results['wanda_pruning']['final']['accuracy']))
    if has_magnitude:
        methods_available.append(('magnitude', 'Magnitude', results['magnitude_pruning']['final']['accuracy']))
    if has_taylor:
        methods_available.append(('taylor', 'Taylor', results['taylor_pruning']['final']['accuracy']))
    
    if len(methods_available) >= 2
    # FLOPs reduction
    if has_flops and orig_flops > 0:
        flops_change_row = ["FLOPs Reduction (%)"]
        if has_coverage:
            cov_flops = results['coverage_pruning']['final']['flops']
            flops_change_row.append(f"{(cov_flops - orig_flops)/1e9:+.2f} G ({(cov_flops/orig_flops - 1)*100:+.2f}%)")
        if has_wanda:
            wan_flops = results['wanda_pruning']['final']['flops']
            flops_change_row.append(f"{(wan_flops - orig_flops)/1e9:+.2f} G ({(wan_flops/orig_flops - 1)*100:+.2f}%)")
        change_data.append(flops_change_row)
    
    print(tabulate(change_data, headers="firstrow", tablefmt="grid"))
    
    # Print summary insights
    print("\n" + "="*100)
    print("SUMMARY INSIGHTS")
    print("="*100)
    
    if has_coverage and has_wanda:
        cov_acc = results['coverage_pruning']['final']['accuracy']
        wan_acc = results['wanda_pruning']['final']['accuracy']
        cov_size = results['coverage_pruning']['final']['size_mb']
        wan_size = results['wanda_pruning']['final']['size_mb']
        
        print(f"\nAccuracy Comparison:")
        if cov_acc > wan_acc:
            print(f"  ✓ Neuron Coverage achieved better accuracy: {cov_acc:.2f}% vs {wan_acc:.2f}%")
            print(f"    Difference: {cov_acc - wan_acc:.2f}%")
        elif wan_acc > cov_acc:
            print(f"  ✓ Wanda achieved better accuracy: {wan_acc:.2f}% vs {cov_acc:.2f}%")
            print(f"    Difference: {wan_acc - cov_acc:.2f}%")
        else:
            print(f"  ✓ Both methods achieved equal accuracy: {cov_acc:.2f}%")
        
        print(f"\nModel Size Comparison:")
        if cov_size < 10)
    print("DETAILED RESULTS")
    print("="*110)
    
    if 'prepare_model' in results:
        print("\n--- Model Preparation ---")
        prep = results['prepare_model']
        print(f"Pretrained Accuracy: {prep['pretrained']['accuracy']:.2f}%")
        print(f"Fine-tuned Accuracy: {prep['finetuned']['accuracy']:.2f}%")
        print(f"Improvement: {prep['finetuned']['accuracy'] - prep['pretrained']['accuracy']:+.2f}%")
        print(f"Fine-tuning Epochs: {prep['finetuned']['epochs']}")
    
    if 'coverage_pruning' in results:
        print("\n--- Neuron Coverage Pruning ---")
        cov = results['coverage_pruning']
        print(f"Original Accuracy: {cov['original']['accuracy']:.2f}%")
        print(f"After Pruning: {cov['pruned']['accuracy']:.2f}%")
        print(f"After Fine-tuning: {cov['final']['accuracy']:.2f}%")
        print(f"Pruning Ratio: {cov['pruning_config']['ratio']*100}%")
        print(f"Fine-tuning Epochs: {cov['final']['fine_tune_epochs']}")
    
    if 'wanda_pruning' in results:
        print("\n--- Wanda Pruning ---")
        wan = results['wanda_pruning']
        print(f"Original Accuracy: {wan['original']['accuracy']:.2f}%")
        print(f"After Pruning: {wan['pruned']['accuracy']:.2f}%")
        print(f"After Fine-tuning: {wan['final']['accuracy']:.2f}%")
        print(f"Pruning Ratio: {wan['pruning_config']['ratio']*100}%")
        print(f"Fine-tuning Epochs: {wan['final']['fine_tune_epochs']}")
    
    if 'magnitude_pruning' in results:
        print("\n--- Magnitude-Based Pruning (Torch-Pruning Baseline) ---")
        mag = results['magnitude_pruning']
        print(f"Original Accuracy: {mag['original']['accuracy']:.2f}%")
        print(f"After 10)
    print(f"TEST SCENARIO {CONFIG['test_scenario']}: RESULTS COMPARISON")
    print("="*110)
    
    # Load results
    results = load_results()
    
    if results is None:
        return
    
    # Print summary comparison table
    print_summary_table(results)
    
    # Print detailed individual results
    print_individual_results(results)
    
    print("\n✓ Results comparison completed")
    print("="*11sults['coverage_pruning']
        print(f"Original Accuracy: {cov['original']['accuracy']:.2f}%")
        print(f"After Pruning: {cov['pruned']['accuracy']:.2f}%")
        print(f"After Fine-tuning: {cov['final']['accuracy']:.2f}%")
        print(f"Pruning Ratio: {cov['pruning_config']['ratio']*100}%")
        print(f"Fine-tuning Epochs: {cov['final']['fine_tune_epochs']}")
    
    if 'wanda_pruning' in results:
        print("\n--- Wanda Pruning ---")
        wan = results['wanda_pruning']
        print(f"Original Accuracy: {wan['original']['accuracy']:.2f}%")
        print(f"After Pruning: {wan['pruned']['accuracy']:.2f}%")
        print(f"After Fine-tuning: {wan['final']['accuracy']:.2f}%")
        print(f"Pruning Ratio: {wan['pruning_config']['ratio']*100}%")
        print(f"Fine-tuning Epochs: {wan['final']['fine_tune_epochs']}")
    
    print("\n" + "="*100)

def main():
    """Main execution function"""
    print("\n" + "="*120)
    print(f"TEST SCENARIO {CONFIG['test_scenario']}: RESULTS COMPARISON")
    print("="*120)
    
    # Load results
    results = load_results()
    
    if results is None:
        return
    
    # Print summary comparison table
    print_summary_table(results)
    
    # Print detailed individual results
    print_individual_results(results)
    
    print("\n✓ Results comparison completed")
    print("="*120 + "\n")

if __name__ == "__main__":
    main()

