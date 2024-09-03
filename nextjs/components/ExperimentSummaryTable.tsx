import { ScrollArea, Table } from "@mantine/core";
import { Experiment } from "../utils/types";

interface ExperimentSummaryTableProps {
  experiments: Experiment[];
  onRowClick?: (experiment: Experiment) => void; // Add this prop
}

const ExperimentSummaryTable: React.FC<ExperimentSummaryTableProps> = ({ experiments, onRowClick }) => {
  const handleRowClick = (experiment: Experiment) => {
    if (onRowClick) {
      onRowClick(experiment);
    }
  };
  return (
    <ScrollArea scrollbarSize={0}>
      <Table withBorder withColumnBorders striped highlightOnHover>
        <thead>
          <tr>
            <th>ID</th>
            <th># of Questions</th>
            <th>Chunk Size</th>
            <th>Overlap</th>
            <th>Split Method</th>
            <th>Retriever</th>
            <th>Embedding Provider</th>
            <th>Model</th>
            <th>Evaluator</th>
            <th># of Chunks Retrieved</th>
          </tr>
        </thead>
        <tbody>
          {experiments?.map((experiment: Experiment, index: number) => (
            <tr key={experiment.id} onClick={() => handleRowClick(experiment)}>
              <td>{experiment.id}</td>
              <td>{experiment?.number_of_question}</td>
              <td>{experiment?.chunk_size}</td>
              <td>{experiment?.chunk_overlap}</td>
              <td>{experiment?.split_method}</td>
              <td>{experiment?.retriever_type}</td>
              <td>{experiment?.embedding_provider}</td>
              <td>{experiment?.model}</td>
              <td>{experiment?.evaluator_model}</td>
              <td>{experiment?.num_neighbors}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </ScrollArea>
  );
};
export default ExperimentSummaryTable;
