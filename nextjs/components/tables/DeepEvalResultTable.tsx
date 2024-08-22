import { ScrollArea, Table, Text } from "@mantine/core";
import { Result } from "../../utils/types";

const DeepEvalResultTable = ({
  results,
  isFastGradingPrompt,
}: {
  results: Result[];
  isFastGradingPrompt: boolean;
}) => {
  // Extract unique DeepEval keys from the results
  const deepEvalKeys = Array.from(
    new Set(results.flatMap((result) => Object.keys(result.deepeval)))
  );

  return (
    <ScrollArea scrollbarSize={0}>
      <Table withBorder withColumnBorders striped highlightOnHover>
        <thead>
          <tr>
            <th>#</th>
            {deepEvalKeys.flatMap((key) => [
              <th colSpan={2} key={`${key}-header`}>{key}</th>
            ])}
          </tr>
          <tr>
            <th></th>
            {deepEvalKeys.flatMap((key) => [
              <th key={`${key}-score`}>Score</th>,
              <th key={`${key}-reason`}>Reason</th>
            ])}
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => (
            <tr key={index}>
              <td>{index + 1}</td>
              {deepEvalKeys.flatMap((key) => {
                const evaluation = result.deepeval[key];
                return [
                  <td key={`${key}-score`}>
                    {evaluation ? <Text>{evaluation.score}</Text> : <Text>-</Text>}
                  </td>,
                  <td key={`${key}-reason`}>
                    {evaluation ? <Text size="sm" color="dimmed">{evaluation.reason}</Text> : <Text>-</Text>}
                  </td>
                ];
              })}
            </tr>
          ))}
        </tbody>
      </Table>
    </ScrollArea>
  );
};

export default DeepEvalResultTable;
