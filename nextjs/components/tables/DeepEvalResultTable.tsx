import { ScrollArea, Spoiler, Table, Text } from "@mantine/core";
import { Result } from "../../utils/types";

const DeepEvalResultTable = ({ results }: { results: Result[] }) => {
  // Extract unique DeepEval keys from the results
  const deepEvalKeys = Array.from(
    new Set(results.flatMap((result) => Object.keys(result?.deepeval || {})))
  );

  return (
    <ScrollArea scrollbarSize={0}>
      {results.length > 0 ? (
        <Table withBorder withColumnBorders striped highlightOnHover>
          <thead>
            <tr>
              <th rowSpan={2}>Question</th>
              <th rowSpan={2}>Expected Answer</th>
              <th rowSpan={2}>Actual Answer</th>
              {deepEvalKeys.map((key) => (
                <th colSpan={2} key={`${key}-header`}>
                  {key}
                </th>
              ))}
            </tr>
            <tr>
              {deepEvalKeys.map((key) => [
                <th key={`${key}-score`}>Score</th>,
                <th key={`${key}-reason`}>Reason</th>,
              ])}
            </tr>
          </thead>
          <tbody>
            {results.map((result, index) => (
              <tr key={index}>
                <td style={{ whiteSpace: "pre-wrap" }}>
                  <Spoiler
                      maxHeight={150}
                      hideLabel={<Text weight="bold" color="blue">Show less</Text>}
                      showLabel={<Text weight="bold" color="blue">Show more</Text>}>
                      {result?.question}
                    </Spoiler>
                </td>
                <td style={{ whiteSpace: "pre-wrap" }}>
                  <Spoiler
                      maxHeight={150}
                      hideLabel={<Text weight="bold" color="blue">Show less</Text>}
                      showLabel={<Text weight="bold" color="blue">Show more</Text>}>
                      {result?.expected}
                    </Spoiler>
                </td>
                <td style={{ whiteSpace: "pre-wrap" }}>
                  <Spoiler
                      maxHeight={150}
                      hideLabel={<Text weight="bold" color="blue">Show less</Text>}
                      showLabel={<Text weight="bold" color="blue">Show more</Text>}>
                      {result?.actual}
                    </Spoiler>
                </td>
                {/* Other DeepEval keys */}
                {deepEvalKeys.map((key) => {
                  if(result.deepeval)
                    {
                      const evaluation = result.deepeval[key];
                  return (
                    <>
                      <td key={`${key}-score`}>
                        {evaluation ? (<Text>{evaluation.score?.toFixed(2)}</Text>) : (<Text>-</Text>)}
                      </td>
                      <td key={`${key}-reason`} style={{ whiteSpace: "pre-wrap" }}>
                        <Spoiler
                        maxHeight={150}
                        hideLabel={<Text weight="bold" color="blue">Show less</Text>}
                        showLabel={<Text weight="bold" color="blue">Show more</Text>}>
                        {evaluation ? (
                          <Text size="sm" color="dimmed">{evaluation.reason}</Text>) : (<Text>-</Text>)}
                        </Spoiler>
                      </td>
                    </>
                  );
                }})}
              </tr>
            ))}
          </tbody>
        </Table>
      ) : (
        <Text align="center" color="dimmed" size="lg">
          No results available.
        </Text>
      )}
    </ScrollArea>
  );
};

export default DeepEvalResultTable;
