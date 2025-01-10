# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:13:04 2025

@author: azamb
"""

import pandas as pd
import os


# Specify the relative folder paths for redacted files
openai_output_folder = 'corrected_results/OpenAI_redacted_files/'
llama_output_folder = 'corrected_results/Llama_redacted_files/'

all_metrics = []
# Process files for both OpenAI and Llama
for model_type, output_folder in [('OpenAI', openai_output_folder), ('Fireworks', llama_output_folder), ('Llama', llama_output_folder)]:
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist. Skipping {model_type} files.")
        continue

    all_csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]

    for file in all_csv_files:
        file_path = os.path.join(output_folder, file)
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')

        # Extract original file name and prompt number
        prompt_number_part = file.split('_')[-2]
        prompt_number = int(prompt_number_part.replace('prompt', ''))
        original_file_name = file.split("_prompt")[0] + ".csv"
        
        total_true_positives=sum(df['true_positives'])
        total_false_positives=sum(df['false_positives'])
        total_true_negatives=sum(df['true_negatives'])
        total_false_negatives=sum(df['false_negatives'])
        total_words=total_true_positives+total_false_positives+total_true_negatives+total_false_negatives

        # Avoid division by zero
        if (total_true_positives + total_false_positives) == 0:
            precision = 0
        else:
            precision = total_true_positives / (total_true_positives + total_false_positives)

        if (total_true_positives + total_false_negatives) == 0:
            recall = 0
        else:
            recall = total_true_positives / (total_true_positives + total_false_negatives)

        if total_words == 0:
            accuracy = 0
            observed_agreement = 0
            expected_agreement = 0
            kappa = 0
        else:
            accuracy = (total_true_positives + total_true_negatives) / total_words
            observed_agreement = (total_true_positives + total_true_negatives) / total_words
            expected_agreement = (
                ((total_true_positives + total_false_positives) * (total_true_positives + total_false_negatives) +
                 (total_false_positives + total_true_negatives) * (total_false_negatives + total_true_negatives))
                / (total_words ** 2)
            )
            if (1 - expected_agreement) == 0:
                kappa = 0
            else:
                kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

        print(f"Updated Metrics for {original_file_name}, Prompt {prompt_number}, Model {model_type}: "
              f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Kappa: {kappa:.3f}")


        # Store metrics in a list
        all_metrics.append({
            'Model': model_type,
            'Prompt': prompt_number,
            'File': original_file_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Kappa': kappa
        })





