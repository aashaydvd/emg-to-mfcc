import os
import json
import torch
import numpy as np
from scripts.evaluation import evaluate_models
from scripts.embeddings import run_analysis

def main():
    # Setup configuration
    root_data_dir = "CSL-EMG_Processed"
    output_base_dir = "results_server"
    
    # Ensure the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    speakers = ["Spk1", "Spk2", 
                "Spk3", "Spk4", "Spk5", "Spk6", "Spk7", "Spk8"
                ]

    # Dictionary to aggregate results for all speakers
    all_results = {}

    for spk in speakers:
        print(f"--- Processing {spk} ---")
        
        # 1. Capture the returned metrics from evaluate_models
        metrics = evaluate_models(
            baseline_path=f"models/{spk}_baseline_model.pth", 
            invariant_path=f"models/{spk}_invariant_model.pth"
        )
        
        # 2. Store the metrics in the dictionary under the speaker's key
        all_results[spk] = metrics
        
        # 3. Run embedding visualizations
        run_analysis(speaker=spk)

    # 4. Save the aggregated evaluation data to a JSON file
    summary_path = os.path.join(output_base_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nSuccessfully saved all evaluation results to: {summary_path}")

if __name__ == "__main__":
    main()
