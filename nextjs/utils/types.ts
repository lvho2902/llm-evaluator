import { UseFormReturn } from "react-hook-form";

export type FormValues = {
  files: any[];
  number_of_question: number;
  chunk_size: number;
  chunk_overlap: number;
  split_method: string;
  embedding_provider: string;
  retriever_type: string;
  model: string;
  evaluator_model: string;
  num_neighbors: number;
};

export type Form = UseFormReturn<FormValues>;

export type Result = {
  experiment_summary_id: number;
  question: string;
  expected: string;
  actual: string;
  consistency: Consistency;
  deepeval: DeepEval;
};

export type Consistency = {
  questions: string;
  answers: string;
  results: string;
  score: number;
}

export type DeepEval = {
  [key: string]: {
    score: number;
    reason: string;
  };
};

export type QAPair = {
  question: string;
  answer: string;
};

export type Experiment = {
  number_of_question: number;
  chunk_size: number;
  chunk_overlap: number;
  split_method: string;
  retriever_type: string;
  embedding_provider: string;
  model: string;
  evaluator_model: string;
  num_neighbors: number;
  performance: number;
  id: number;
};
