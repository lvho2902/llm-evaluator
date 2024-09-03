import os
import csv
import json
import pandas as pd
from datetime import datetime
from evaluation import evaluator

def load_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading text file: {e}")
        return None

def load_csv_file(filepath):
    try:
        df = pd.read_csv(filepath)
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

def save_results(results, config):
    try:
        # Ensure the directory exists
        directory = 'evaluation_results'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create a filename with configuration parameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_parts = [
            f"questions{config['number_of_question']}",
            f"chunksize{config['chunk_size']}",
            f"overlap{config['chunk_overlap']}",
            f"split{config['split_method']}",
            f"retriever{config['retriever_type']}",
            f"embedding{config['embedding_provider']}",
            f"model{config['model']}",
            f"evaluator{config['evaluator_model']}",
            f"neighbors{config['num_neighbors']}",
            timestamp
        ]
        csv_filename = "_".join(filename_parts) + ".csv"

        # Complete path including directory
        file_path = os.path.join(directory, csv_filename)

        # Initialize the CSV file
        fieldnames = ['question', 'expected', 'actual', 'consistency score', 
                      'consistency results']
        
        # Dynamically add fieldnames for all deepeval items
        dynamic_fieldnames = set()  # Use a set to avoid duplicates
        for result_with_id in results:
            deepeval = result_with_id['data'].get('deepeval', {})
            for eval_name in deepeval:
                dynamic_fieldnames.add(f"{eval_name} score")
                dynamic_fieldnames.add(f"{eval_name} reason")
        
        # Add the dynamic field names to the list of all fieldnames
        fieldnames.extend(sorted(dynamic_fieldnames))  # Sort for consistent order
        
        # Write to the CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result_with_id in results:
                result_dict = result_with_id["data"]
                row = {
                    'question': result_dict['question'],
                    'expected': result_dict['expected'],
                    'actual': result_dict['actual'],
                    'consistency score': result_dict['consistency']['score'],
                    'consistency results': result_dict['consistency']['results']
                }
                
                # Add dynamic deepeval items
                deepeval = result_dict.get('deepeval', {})
                for eval_name, eval_data in deepeval.items():
                    row[f"{eval_name} score"] = eval_data['score']
                    row[f"{eval_name} reason"] = eval_data['reason']
                
                writer.writerow(row)
        print(f"Results saved to {csv_filename}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":

    # Load text data from the file
    text = load_text_file('docs/istqb/ISTQB_CTFL_Syllabus-v4.0.txt')

    # Load test dataset from the CSV file
    test_dataset = load_csv_file('docs/istqb/ISTQB_CTFL_Syllabus-v4.0.csv')

    # Configuration parameters
    config = {
        'number_of_question': 10,
        'chunk_size': 1500,
        'chunk_overlap': 200,
        'split_method': "RecursiveTextSplitter",
        'retriever_type': "similarity-search",
        'embedding_provider': "Ollama",
        'model': "mistral",
        'evaluator_model': "llama3",
        'num_neighbors': 3
    }

    # # Run the evaluator and collect results
    # try:
    #     results = evaluator.run_evaluator(
    #         None,
    #         text,
    #         test_dataset,
    #         config['number_of_question'],
    #         config['chunk_size'],
    #         config['chunk_overlap'],
    #         config['split_method'],
    #         config['retriever_type'],
    #         config['embedding_provider'],
    #         config['model'],
    #         config['evaluator_model'],
    #         config['num_neighbors']
    #     )

    #     all_results = []
    #     if not results:
    #         print("Evaluator returned no results.")
        
    #     for i, result_json in enumerate(results):
    #         if not result_json.strip():  # Check if result_json is empty or whitespace
    #             print(f"Result {i} is empty or whitespace")
    #             continue  # Skip empty result_json
            
    #         print(f"Result {i}: {result_json}")
    #         try:
    #             result = json.loads(result_json)
    #             all_results.append(result)
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON in result {i}: {e}")
    
    #     # Save all results to a new file with configuration details in the filename
    #     save_results(all_results, config)


    # except Exception as e:
    #     print(f"Error during evaluation: {e}")
    import logging
    from text_utils import QA_CHAIN_PROMPT
    from evaluation import helpers
    from model.ollama_model import OllamaModel

    logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    llm = evaluator.make_llm("mistral", logger)
    splits = evaluator.split_texts(text, config['chunk_size'], config['chunk_overlap'], config['split_method'], logger)
    retriever = evaluator.make_retriever(splits, config['retriever_type'], config['embedding_provider'], config['num_neighbors'], logger)
    chain = evaluator.make_chain(llm, retriever, QA_CHAIN_PROMPT, "question", logger)
    

    predictions = []
    retrieved_docs = []
    for i in range(0, config['number_of_question']):
        prediction = chain(test_dataset[i])
        predictions.append(prediction)
        doc = evaluator.make_retrieved_docs(retriever, test_dataset[i])
        retrieved_docs.append(doc)
    
    for prediction in predictions:
        print(predictions)
    
    test_cases = helpers.create_test_cases(predictions, retrieved_docs)

    custom_model = OllamaModel("llama3")
    metrics = helpers.create_metrics(custom_model)
    test_results = helpers.evaluate(test_cases, metrics, run_async=True, use_cache=False, verbose_mode=True)
        

