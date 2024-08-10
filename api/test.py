# from statistics import mean
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_score import rouge_scorer

# def calculate_bleu(reference, candidate):
#     """
#     Calculates the BLEU score between a reference and a candidate sentence.
    
#     :param reference: The ground truth sentence (string).
#     :param candidate: The generated sentence by the model (string).
#     :return: BLEU score (float).
#     """
#     reference_tokens = []
#     reference_tokens.append(reference.split())  # Tokenize reference
#     candidate_tokens = candidate.split()  # Tokenize candidate
#     smoothie = SmoothingFunction().method4
#     bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
#     return bleu_score

# def calculate_rouge(reference, candidate):
#     """
#     Calculates the ROUGE-1, ROUGE-2, and ROUGE-L scores between a reference and a candidate sentence.
    
#     :param reference: The ground truth sentence (string).
#     :param candidate: The generated sentence by the model (string).
#     :return: Dictionary containing ROUGE scores (rouge1, rouge2, rougeL).
#     """
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     rouge_scores = scorer.score(reference, candidate)
#     return rouge_scores

# def evaluate_bleu_rouge(predictions, logger):
#     """
#     Evaluates BLEU and ROUGE scores from a list of dictionaries.
    
#     :param predictions: List of dictionaries containing 'question', 'answer' (reference), and 'result' (generated sentence).
#     :param logger: Logger for logging information.
#     :return: Tuple containing:
#              - Average BLEU score (float).
#              - Overall average of the ROUGE-1, ROUGE-2, and ROUGE-L F-measure scores (float).
#     """
#     logger.info("Evaluating BLEU and ROUGE scores from predictions...")

#     bleu_scores = []
#     rouge_scores = []

#     for item in predictions:
#         reference = item['answer']
#         candidate = item['result']
        
#         # Calculate BLEU score
#         bleu_score = calculate_bleu(reference, candidate)
#         print(bleu_score)
#         bleu_scores.append(bleu_score)
        
#         # Calculate ROUGE scores
#         rouge_score = calculate_rouge(reference, candidate)
#         rouge_scores.append(rouge_score)

#     # Calculate average BLEU score
#     avg_bleu_score = mean(bleu_scores) if bleu_scores else 0

#     # Calculate average F-measure for each ROUGE metric
#     avg_rouge1 = mean([score['rouge1'].fmeasure for score in rouge_scores]) if rouge_scores else 0
#     avg_rouge2 = mean([score['rouge2'].fmeasure for score in rouge_scores]) if rouge_scores else 0
#     avg_rougeL = mean([score['rougeL'].fmeasure for score in rouge_scores]) if rouge_scores else 0

#     # Calculate the overall average ROUGE score
#     avg_rouge = mean([avg_rouge1, avg_rouge2, avg_rougeL]) if rouge_scores else 0

#     return avg_bleu_score, avg_rouge

# # Example usage:
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     predictions = [
#         {
#             'question': 'What is the primary purpose of the ALOM plugin?',
#             'answer': 'The ALOM plugin integrates OmniSwitch management into the Milestone VMS, allowing users to control port and switch functions directly from the Milestone interface. This eliminates the need to switch between applications.',
#             'result': 'The primary purpose of the ALOM plugin is to integrate OmniSwitch management functions into the Milestone VMS, allowing users to perform tasks such as rebooting cameras and managing port power allocations without having to connect to a separate user interface.'
#         }
#     ]

#     avg_bleu_score, avg_rouge = evaluate_bleu_rouge(predictions, logger)

#     print(f"Average BLEU Score: {avg_bleu_score:.4f}")
#     print(f"Overall Average ROUGE F-measure: {avg_rouge:.4f}")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_rouge(reference, candidate):
    """
    Calculates the ROUGE-1, ROUGE-2, and ROUGE-L scores between a reference and a candidate sentence.
    
    :param reference: The ground truth sentence (string).
    :param candidate: The generated sentence by the model (string).
    :return: Array containing ROUGE scores (rouge1, rouge2).
    """
    # Initialize CountVectorizer for unigram (ROUGE-1) and bigram (ROUGE-2) calculations
    vectorizer = CountVectorizer(ngram_range=(1, 2))

    # Fit and transform the texts
    vectors = vectorizer.fit_transform([reference, candidate])

    # Compute cosine similarity (as a proxy for ROUGE score)
    cosine_sim = cosine_similarity(vectors)

    # The diagonal contains the similarity of each text with itself, and the off-diagonal contains the similarity between the texts
    rouge_scores = cosine_sim[0, 1]
    return rouge_scores

# Reference and candidate texts
reference_text = "The ALOM plugin integrates OmniSwitch management into the Milestone VMS, allowing users to control port and switch functions directly from the Milestone interface. This eliminates the need to switch between applications."
candidate_text = "The primary purpose of the ALOM plugin is to integrate OmniSwitch management functions into the Milestone VMS, allowing users to perform tasks such as rebooting cameras and managing port power allocations without having to connect to a separate user interface."

print(calculate_rouge(reference_text, candidate_text))
