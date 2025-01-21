# -*- coding: utf-8 -*-

import pandas as pd
import os
import string
from collections import Counter

#TODO Specify as AZ if want to use alternative counting method for tp, tn, fp, fn
countstrategy='AZ'

# Specify the relative folder paths for redacted files
openai_output_folder = 'results/OpenAI_redacted_files/'
llama_output_folder = 'results/Llama_redacted_files/'
fireworks_output_folder = 'results/Fireworks_redacted_files/'
human_redacted_folder = 'human_redacted_files/'

# Ensure directory for correcting results exists
if not os.path.exists('corrected_results'):
    os.makedirs('corrected_results')

def count_redacted(text):
    return text.count("[REDACTED]")

def get_true_false_positives_negatives(human_redacted, model_redacted):
    fp=0
    fn=0
    tp=0
    tn=0
    human_redacted = [item.lower() for item in human_redacted]
    model_redacted = [item.lower() for item in model_redacted]
    for i in range(len(human_redacted)):
        wordS1=human_redacted[i]
        if wordS1!='redacted':
            #If word is still there after redaction for both, it was a true negative
            if wordS1 in model_redacted:
                index=model_redacted.index(wordS1)
                tn+=1
                if index<3:
                    del model_redacted[:index+1]
            #If word was not redacted in the ground truth but eliminated by the llm, it is a false positive
            else:
                if len(model_redacted)!=0:
                    wordS2=model_redacted[0]
                    #Check cases when there was no space in the middle of 2 words
                    if wordS2 in wordS1:
                        tn+=1
                        index=model_redacted.index(wordS2)
                        del model_redacted[:index+1]
                    elif wordS1 in wordS2:
                        tn+=1
                        model_redacted[0]=wordS2.replace(wordS1, "")
                    else:
                        fp+=1
        else:
            if len(model_redacted)!=0:
                wordS2=model_redacted[0]
                #If next word in the llm is also redacted it is a true positive
                if wordS2=='redacted':
                    tp+=1
                #If next word was not redacted, it is a false negative
                else:
                    fn+=1
    
    return tp, tn, fp, fn
    
def clean_text(text):
    # Checks if text is a string, converts to string if not
    if not isinstance(text, str):
        text = str(text)

    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = text.split()
    # Remove articles and non-alphanumeric words
    words = [word for word in words if word.isalnum() and word.lower() not in ['a', 'an', 'the']]
    return words

def add_word_lists_to_df(df, model_type):
    df['word_list_human_redacted'] = df['post_text_human_redacted'].apply(clean_text)
    df['word_list_original'] = df['post_text_original'].apply(clean_text)
    if model_type == 'OpenAI':
        df['word_list_model_redacted'] = df['post_text_OpenAI_redacted'].apply(clean_text)
        df = df.apply(adjust_model_output, axis=1, args=(model_type,))
    elif model_type == 'Fireworks':
        df['word_list_model_redacted'] = df['post_text_Fireworks_redacted'].apply(clean_text)
        df = df.apply(adjust_model_output, axis=1, args=(model_type,))
    elif model_type == 'Llama':
        df['word_list_model_redacted'] = df['post_text_Llama_redacted'].apply(clean_text)
        df = df.apply(adjust_model_output, axis=1, args=(model_type,))
    else:
        raise ValueError("Unknown model_type. Should be 'OpenAI' or 'Llama'")
    return df

def adjust_model_output(row, model_type):
    """
    Adjusts the model output if it contains extra lines or prefixes that are not part of the redacted text.
    Specifically designed to handle outputs from OpenAI and Llama models.
    """
    if model_type == 'OpenAI':
        model_output_column = 'post_text_OpenAI_redacted'
        word_list_column = 'word_list_model_redacted'
    elif model_type == 'Llama':
        model_output_column = 'post_text_Llama_redacted'
        word_list_column = 'word_list_model_redacted'
    else:
        return row  # If model_type is not recognized, return the row as is

    # Check if the first word in the human redacted text and model output do not match
    if row['word_list_human_redacted'] and row[word_list_column]:
        if row['word_list_human_redacted'][0] != row[word_list_column][0]:
            # For Llama, sometimes there's an extra line or preamble that needs to be removed
            # We can attempt to find the index where the actual redacted text starts
            lines = row[model_output_column].split('\n')
            # Attempt to find the line that matches the start of the human redacted text
            for i, line in enumerate(lines):
                cleaned_line = clean_text(line)
                if cleaned_line and cleaned_line[0] == row['word_list_human_redacted'][0]:
                    # Reconstruct the output from this line onwards
                    new_output = '\n'.join(lines[i:]).strip()
                    row[model_output_column] = new_output
                    row[word_list_column] = clean_text(new_output)
                    break
            else:
                # If no matching line is found, assume the last line is the redacted text
                new_output = lines[-1].strip()
                row[model_output_column] = new_output
                row[word_list_column] = clean_text(new_output)
    return row

def calculate_metrics(df):
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_words = 0

    tp_counts = []
    fp_counts = []
    fn_counts = []
    tn_counts = []

    for _, row in df.iterrows():
        row_tp = 0
        row_fp = 0
        row_fn = 0
        row_tn = 0

        original_words = row['word_list_original']
        human_words = row['word_list_human_redacted']
        model_words = row['word_list_model_redacted']
            
        row_tp, row_tn, row_fp, row_fn = get_true_false_positives_negatives(human_words, model_words)
        
        total_true_positives+=row_tp
        total_false_positives+=row_fp
        total_false_negatives+=row_fn
        total_true_negatives+=row_tn
        total_words+=len(human_words)
            
        tp_counts.append(row_tp)
        fp_counts.append(row_fp)
        fn_counts.append(row_fn)
        tn_counts.append(row_tn)
        df.at[_, 'true_positives'] = row_tp
        df.at[_, 'true_negatives'] = row_tn
        df.at[_, 'false_positives'] = row_fp
        df.at[_, 'false_negatives'] = row_fn


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

    return accuracy, precision, recall, kappa, tp_counts, tn_counts, fp_counts, fn_counts

prompt_df = pd.read_csv('prompts.csv', encoding_errors='ignore')

# Initialize empty lists to store metrics
all_metrics = []

# Process files for both OpenAI and Llama
for model_type, output_folder in [('OpenAI', openai_output_folder), ('Fireworks', fireworks_output_folder), ('Llama', llama_output_folder)]:
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} does not exist. Skipping {model_type} files.")
        continue

    all_csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]

    for file in all_csv_files:
        file_path = os.path.join(output_folder, file)
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore')
        df = df.replace({'�': ' '}, regex=True)
        df = df.replace({'&nbsp;': ' '}, regex=True)
        df = df.replace({'&amp;': '&'}, regex=True)
        df = df.replace({'&quot;': '"'}, regex=True)
        df = df.replace({'&#39;': ' '}, regex=True)
        df = df.replace({'Ì¢Â‰': ' '}, regex=True)
        df = df.replace({'Ì¢Â': ' '}, regex=True)
        df = df.replace({'Ì¢': ' '}, regex=True)
        df = df.replace({'Ûª': ' '}, regex=True)
        df = df.replace({'http': ' '}, regex=True)
        
        # Extract original file name and prompt number
        prompt_number_part = file.split('_')[-2]
        prompt_number = int(prompt_number_part.replace('prompt', ''))
        original_file_name = file.split("_prompt")[0] + ".csv"

        human_file_path = os.path.join(human_redacted_folder, original_file_name)
        if not os.path.exists(human_file_path):
            print(f"Human redacted file not found for {original_file_name}, skipping...")
            continue

        # Read human redacted file
        df_human = pd.read_csv(human_file_path, encoding='utf-8', encoding_errors='ignore')
        df = df.replace({'�': ' '}, regex=True)
        df = df.replace({'&nbsp;': ' '}, regex=True)
        df = df.replace({'&amp;': '&'}, regex=True)
        df = df.replace({'&quot;': '"'}, regex=True)
        df = df.replace({'&#39;': ' '}, regex=True)
        df = df.replace({'Ì¢Â‰': ' '}, regex=True)
        df = df.replace({'Ì¢Â': ' '}, regex=True)
        df = df.replace({'Ì¢': ' '}, regex=True)
        df = df.replace({'Ûª': ' '}, regex=True)
        df_human['id'] = df_human.index
        if 'post_text_human_redacted' not in df.columns:
            df = pd.merge(df, df_human[['id', 'post_text_human_redacted']], on='id')

        # Add word lists as new columns and adjust model outputs if necessary
        df = add_word_lists_to_df(df, model_type=model_type)

        # Calculate metrics and get counts
        accuracy, precision, recall, kappa, tp_counts, tn_counts, fp_counts, fn_counts = calculate_metrics(df)

        # Add counts to DataFrame
        df['true_positives'] = tp_counts
        df['true_negatives'] = tn_counts
        df['false_positives'] = fp_counts
        df['false_negatives'] = fn_counts

        print(f"Updated Metrics for {original_file_name}, Prompt {prompt_number}, Model {model_type}: "
              f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, Kappa: {kappa:.3f}")

        # Save updated DataFrame
        df.to_csv(file_path, index=False)
        
        # Save updated DataFrame
        df.to_csv('corrected_'+file_path, index=False)

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

# Create a DataFrame with all metrics
all_metrics_df = pd.DataFrame(all_metrics)


# Save all metrics to a CSV file
all_metrics_df.to_csv('results/metrics_combined.csv', index=False)
