# De-identification-using-LLMs

## Objective

This research project focuses on utilizing Large Language Models (LLMs) to remove personally identifiable information (PII) from forum posts. The project explores the effectiveness of both OpenAI's GPT-4o model and Meta's LLama3.1 model (run locally) in de-identifying sensitive data.

## Project Structure

### Data

- **original_files**: Contains the original data with PII.
- **human_redacted_files**: Contains the files that have been de-identified by humans.
- **results**:
  - **OpenAI_redacted_files**: Files generated by the OpenAI GPT-4o model.
  - **Fireworks_redacted_files**: Files generated by LLama3.3 70B model (through Fireworks API).
  - **Llama_redacted_files**: Files generated by the LLama3.1 8B model (local).
- **prompts.csv**: A CSV file containing a list of prompts used during the de-identification process and for evaluation.

### Code Files

- **de_identified_csv_generator.py**: This script processes the original files and generates de-identified CSV files using either the OpenAI GPT-4 model or the LLaMA model accessed via Fireworks, based on user selection.
- **de_identified_csv_evaluator.py**: This script evaluates the reliability of the de-identification process by comparing the model outputs with human-redacted files. It calculates metrics such as accuracy, precision, recall, and Cohen's kappa.

## Prerequisites

Before getting started, ensure you have the following:

- **Python Environment**: Python 3.6 or higher is recommended.
- **OpenAI API Key**: Obtain an API key from OpenAI to access the GPT-4o model.
- **Fireworks API Key**: Obtain an API key from Fireworks to access the Llama 3.3 70B model.
- **Llama 3.1 Model**: Obtain LLama 3.1 8B model via [Ollama](https://ollama.com/download).
- **Python Packages**: Install the required packages using pip:

  ```bash
  pip install openai pandas
  ```

- **Input Data**:
  - **original_files**: Place your original CSV files containing PII in this folder.
  - **human_redacted_files**: Place the corresponding human-redacted CSV files in this folder.

## Downloading and Installing Llama 3.1 8B
**Step 1:** Download Ollama
On Mac or Windows, go to the Ollama download page [here](https://ollama.com/download) and select your platform to download it, then double click the downloaded file to install Ollama.

**Step 2:** Download and test run Llama 3.1
On a terminal or console, run ```ollama pull llama3.1``` to download the Llama 3.1 8b chat model in the 4-bit quantized format with a size of about 4.7 GB. For better results, you can download Llama 3.1 70b, ```ollama pull llama3.1:70b```, but you will need around 128 of RAM to run it locally. You can check all the available models on the [Ollama web page](https://ollama.com/search). If you decide to use a different model from Llama 3.1 8b you need to change the model name in the `de_identified_csv_generator.py` script.

## Getting Started

Here's how to initiate the project:

**Step 1:** Organize Data
Place your original files with PII in the original_files folder and human-redacted files in the human_redacted_files folder. **We have artificially created some sample files in the folder for your reference.**

**Step 2:** Run the `de_identified_csv_generator.py`
Execute the de_identified_csv_generator script, providing the necessary input files and the output folder (which will be automatically created):
This script will process the files, remove PII, and generate an OpenAI de-identified CSV file within an OpenAI_redacted_files folder inside the results folder.

**Step 3:** Run the `de_identified_csv_evaluator.py`
To evaluate the accuracy of the de-identification process, run the De-identified CSV Evaluator script:
This script will analyze the de-identified CSV files in the results folder and update the metrics csv with accuracy, precision, recall, and kappa values. 

**Step 4:** Check discrepancies between human and LLM-based redaction 
To evaluate discrepancies between both human-based and LLM-based redactions, open the `corrected_results` folder and check if the disagreements are due to human or LLM mistakes. Some discrepancies can also be due to inconsistencies in the formatting of both humans or LLMs when providing the redacted version of the files (e.g., "Professor Jordan" de-identified as Professor [REDACTED] by humans and Prof. [REDACTED] by the LLM). Correct the True/False Positives/Negatives on each row of the file if necessary.

**Step 5:** Run the `performance_metrics.py`
After checking the discrepancies and correcting the number of True/False Positives/Negatives, run the  `performance_metrics.py` script to calculate performance metrics again.

