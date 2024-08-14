import { ScrollArea, Table } from "@mantine/core";
import { Experiment } from "../utils/types";

const ExperimentSummaryTable = ({
  experiments,
}: {
  experiments: Experiment[];
}) => {
  return (
    <ScrollArea scrollbarSize={0}>
      <Table withBorder withColumnBorders striped highlightOnHover fontSize="10.8px">
        <thead>
          <tr>
            <th>#</th>
            <th># of Questions</th>
            <th>Chunk Size</th>
            <th>Overlap</th>
            <th>Split Method</th>
            <th>Retriever</th>
            <th>Embedding Algorithm</th>
            <th>Model</th>
            <th>Evaluator</th>
            <th>Grading Prompt Style</th>
            <th># of Chunks Retrieved</th>
            <th>Avg Retrieval Relevancy Score</th>
            <th>Avg Answer Similarity Score</th>
            <th>Avg BLEU</th>
            <th>Avg ROUGE</th>
            <th>Avg METEOR</th>
            <th>Avg Latency (s)</th>
          </tr>
        </thead>
        <tbody>
          {experiments?.map((result: Experiment, index: number) => (
            <tr key={index}>
              <td>{result.id}</td>
              <td>{result?.evalQuestionsCount}</td>
              <td>{result?.chunkSize}</td>
              <td>{result?.overlap}</td>
              <td>{result?.splitMethod}</td>
              <td>{result?.retriever}</td>
              <td>{result?.embeddingAlgorithm}</td>
              <td>{result?.model}</td>
              <td>{result?.grader}</td>
              <td>{result?.gradingPrompt}</td>
              <td>{result?.numNeighbors}</td>
              <td>{result?.avgRelevancyScore}</td>
              <td>{result?.avgAnswerScore}</td>
              <td>{result?.avgBleuScore.toFixed(3)}</td>
              <td>{result?.avgRougeScore.toFixed(3)}</td>
              <td>{result?.avgMeteorScores.toFixed(3)}</td>
              <td>{result?.avgLatency.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </ScrollArea>
  );
};
export default ExperimentSummaryTable;
