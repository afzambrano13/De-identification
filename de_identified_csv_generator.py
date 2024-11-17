import os
import pandas as pd

# Import both OpenAI and Fireworks libraries
from openai import OpenAI
import requests
import json

# User choice for the language model
print("Choose the language model to use:")
print("1. OpenAI GPT-4")
print("2. Fireworks LLaMA")
model_choice = input("Enter 1 or 2: ")

# Set up API keys and clients based on the choice
if model_choice == '1':
    # OpenAI GPT-4 setup
    openai_api_key = input("Please enter your OpenAI API key: ").strip()
    client = OpenAI(api_key=openai_api_key)
    model_name = "gpt-4o"
    api_provider = 'OpenAI'
elif model_choice == '2':
    model_name = "llama3.1"
    api_provider = 'Llama'
    url = "http://localhost:11434/api/chat"
    
    #Method to exchange messages with Llama
    def llama3(prompt):
        data = {
            "model": model_name,
            "messages": [
                {
                  "role": "user",
                  "content": prompt
                }
            ],
            "stream": False
        }
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, json=data)
        return(response.json()['message']['content'])
    
    #Method to correct extra messages in Llama repplies
    def extract_from_first_word(OriginalText, Llamaresponse):
        # Get the first word of the first string
        first_word = OriginalText.split()[0]
    
        # Find the position of the first word in the second string
        start_index = Llamaresponse.find(first_word)
    
        # If the first word is found, return the substring starting from it
        if start_index != -1:
            return Llamaresponse[start_index:]
        
        # If the first word is not found, return a message
        return Llamaresponse
else:
    print("Invalid choice. Please run the script again and select 1 or 2.")
    exit()

# Specify the relative folder paths for original files and output
original_folder = 'original_files/'
output_folder = f'results/{api_provider}_redacted_files/'

# Reads prompts from a CSV file
if api_provider == 'OpenAI':
    try:
        prompt_df = pd.read_csv('Prompts.csv', encoding='utf-8')
    except Exception as e:
        print("Error reading prompts.csv:", str(e))
        exit()
elif api_provider == 'Llama':
    try:
        prompt_df = pd.read_csv('PromptsLlama.csv', encoding='utf-8')
    except Exception as e:
        print("Error reading prompts.csv:", str(e))
        exit()
# Creates the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Constructs full path using the current working directory
original_folder_path = os.path.join(os.getcwd(), original_folder)

def api_call(post_text, prompt):
    try:
        message_content = f'{prompt}\n{post_text}'
        if api_provider == 'OpenAI':
            client = OpenAI(api_key=openai_api_key )
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message_content}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        elif api_provider == 'Llama':
            response = llama3(message_content)
            return response
        else:
            return "Invalid API Provider"
    except Exception as e:
        print(f"API call error for post_text: {post_text[:50]}...", str(e))
        return "API_ERROR"

try:
    original_files = os.listdir(original_folder_path)
except Exception as e:
    print(f"Error reading from directory {original_folder_path}:", str(e))
    exit()

for original_file in original_files:
    if original_file.endswith('.csv'):
        try:
            original_df = pd.read_csv(
                os.path.join(original_folder_path, original_file),
                encoding='utf-8',
                encoding_errors='replace'
            )
            # Iterates for each prompt
            for index, row in prompt_df.iterrows():
                prompt = row['prompt']
                # Make a copy of the original DataFrame to avoid overwriting previous results
                temp_df = original_df.copy()
                for row_idx in temp_df.index:
                    post_text=temp_df.loc[row_idx, 'post_text_original']
                    post_text_response = api_call(post_text, prompt)
                    if api_provider == 'Llama':
                        post_text_response = extract_from_first_word(post_text,post_text_response)
                    temp_df.loc[row_idx, f'post_text_{api_provider}_redacted'] = post_text_response
                    
                    print(f"Updated row {row_idx+1} for prompt {index+1} in dataframe.")

                # Saves the CSV for the current prompt
                csv_filename = f'{original_file.replace(".csv", "")}_prompt{index+1}_{api_provider.lower()}.csv'
                output_file_path = os.path.join(output_folder, csv_filename)
                temp_df.to_csv(output_file_path, index=False)
                print(f"Saved file {output_file_path} for prompt {index+1}")

        except Exception as e:
            print(f"Error processing file {original_file}:", str(e))
