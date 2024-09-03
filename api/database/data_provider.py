import sqlite3
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProvider:
    
    @staticmethod
    def create_tables():
        try:
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ExperimentSummary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        number_of_question INTEGER,
                        chunk_size INTEGER,
                        chunk_overlap INTEGER,
                        split_method TEXT,
                        retriever_type TEXT,
                        embedding_provider TEXT,
                        model TEXT,
                        evaluator_model TEXT,
                        num_neighbors INTEGER
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS Evaluations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_summary_id INTEGER,
                        data TEXT,
                        FOREIGN KEY (experiment_summary_id) REFERENCES ExperimentSummary(id)
                    )
                ''')
                # Adding indexes for performance optimization
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_experiment_summary_id ON Evaluations (experiment_summary_id)')
                conn.commit()
                logging.info('Tables created successfully.')
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")

    @staticmethod
    def save_experiment_summary(number_of_question, chunk_size, chunk_overlap, split_method, retriever_type, embedding_provider, model, evaluator_model, num_neighbors):
        try:
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ExperimentSummary (number_of_question, chunk_size, chunk_overlap, split_method, retriever_type, embedding_provider, model, evaluator_model, num_neighbors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    number_of_question, chunk_size, chunk_overlap, split_method, retriever_type, embedding_provider, model, evaluator_model, num_neighbors
                ))
                conn.commit()
                inserted_id = cursor.lastrowid
                logging.info(f"Experiment summary saved with ID: {inserted_id}")
                return inserted_id
        except sqlite3.Error as e:
            logging.error(f"Error saving experiment summary: {e}")
            return None

    @staticmethod
    def get_experiment_summary_by_id(id):
        if not isinstance(id, int):
            logging.error("Invalid ID type. Must be an integer.")
            return None

        try:
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM ExperimentSummary WHERE id = ?', (id,))
                result = cursor.fetchone()
                if result:
                    logging.info(f"Experiment summary retrieved: {result}")
                    return result
                else:
                    logging.info(f"No experiment summary found with ID: {id}")
                    return None
        except sqlite3.Error as e:
            logging.error(f"Error retrieving experiment summary by ID: {e}")
            return None

    @staticmethod
    def save_evaluation_result(experiment_summary_id, result_dict):
        if not isinstance(experiment_summary_id, int) or not isinstance(result_dict, dict):
            logging.error("Invalid input types for saving evaluation result.")
            return None
        
        try:
            data_json = json.dumps(result_dict)
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO Evaluations (experiment_summary_id, data) VALUES (?, ?)', (experiment_summary_id, data_json))
                conn.commit()
                inserted_id = cursor.lastrowid
                logging.info(f"Evaluation result saved with ID: {inserted_id}")
                return inserted_id
        except sqlite3.Error as e:
            logging.error(f"Error saving evaluation result: {e}")
            return None

    @staticmethod
    def get_all_experiments():
        try:
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM ExperimentSummary')
                rows = cursor.fetchall()
                if rows:
                    experiments = [
                        {
                            "id": row[0],
                            "number_of_question": row[1],
                            "chunk_size": row[2],
                            "chunk_overlap": row[3],
                            "split_method": row[4],
                            "retriever_type": row[5],
                            "embedding_provider": row[6],
                            "model": row[7],
                            "evaluator_model": row[8],
                            "num_neighbors": row[9]
                        }
                        for row in rows
                    ]
                    logging.info(f"Retrieved {len(experiments)} experiment summaries.")
                    return experiments
                else:
                    logging.info("No experiments found.")
                    return []
        except sqlite3.Error as e:
            logging.error(f"Error retrieving all experiments: {e}")
            return []

    @staticmethod
    def get_evaluations_by_experiment_id(experiment_summary_id):
        if not isinstance(experiment_summary_id, int):
            logging.error("Invalid experiment_summary_id type. Must be an integer.")
            return None
        
        try:
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, data FROM Evaluations WHERE experiment_summary_id = ?', (experiment_summary_id,))
                rows = cursor.fetchall()
                evaluations = [
                    {
                        "experiment_id": row[0],
                        "data": json.loads(row[1])
                    }
                    for row in rows
                ]
                logging.info(f"Retrieved {len(evaluations)} evaluations for experiment ID: {experiment_summary_id}")
                return evaluations
        except sqlite3.Error as e:
            logging.error(f"Error retrieving evaluations by experiment ID: {e}")
            return []

    @staticmethod
    def get_evaluations():
        try:
            with sqlite3.connect('evaluation_results.db') as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, data FROM Evaluations')
                rows = cursor.fetchall()
                evaluations = [
                    {
                        "experiment_id": row[0],
                        "data": json.loads(row[1])
                    }
                    for row in rows
                ]
                logging.info(f"Retrieved {len(evaluations)} total evaluations.")
                return evaluations
        except sqlite3.Error as e:
            logging.error(f"Error retrieving all evaluations: {e}")
            return []
