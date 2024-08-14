import { ScrollArea, Spoiler, Table, Text } from "@mantine/core";
import { Result } from "../../utils/types";
import renderPassFail from "../../utils/renderPassFail";

const ExperimentResultsTable = ({
  results,
  isFastGradingPrompt,
}: {
  results: any[];
  isFastGradingPrompt: boolean;
}) => {
  return (
    <ScrollArea scrollbarSize={0}>
      <Table withBorder withColumnBorders striped highlightOnHover fontSize="10.8px">
        <thead>
          <tr>
            <th>Question</th>
            <th>Expected Answer</th>
            <th>Observed Answer</th>
            <th>Retrieval Relevancy Score</th>
            <th>Answer Similarity Score</th>
            <th>BLEU</th>
            <th>ROUGE</th>
            <th>METEOR</th>
            <th>Latency (s)</th>
            <th>Self Check Statements</th>
            <th>Expected Self Check </th>
            <th>Actual Self Check</th>
            <th>Grade Self Check</th>
          </tr>
        </thead>
        <tbody>
          {results?.map((result: Result, index: number) => {
            return (
              <tr key={index}>
                <td>{result?.question}</td>
                <td>{result?.answer}</td>
                <td>{result?.result}</td>
                <td style={{ whiteSpace: "pre-wrap" }}>
                  {isFastGradingPrompt ? (
                    renderPassFail(result.retrievalScore)
                  ) : (
                    <Spoiler
                      maxHeight={150}
                      hideLabel={
                        <Text weight="bold" color="blue">
                          Show less
                        </Text>
                      }
                      showLabel={
                        <Text weight="bold" color="blue">
                          Show more
                        </Text>
                      }
                    >
                      {result?.retrievalScore.justification}
                    </Spoiler>
                  )}
                </td>
                <td style={{ whiteSpace: "pre-wrap" }}>
                  {isFastGradingPrompt ? (
                    renderPassFail(result?.answerScore)
                  ) : (
                    <Spoiler
                      maxHeight={150}
                      hideLabel={
                        <Text weight="bold" color="blue">
                          Show less
                        </Text>
                      }
                      showLabel={
                        <Text weight="bold" color="blue">
                          Show more
                        </Text>
                      }
                    >
                      {result?.answerScore.justification}
                    </Spoiler>
                  )}
                </td>
                <td>{result?.avgBleuScore.toFixed(3)}</td>
                <td>{result?.avgRougeScore.toFixed(3)}</td>
                <td>{result?.avgMeteorScores.toFixed(3)}</td>
                <td>{result?.latency?.toFixed(3)}</td>
                <td style={{ whiteSpace: "pre-wrap" }}>
                  <Spoiler
                      maxHeight={150}
                      hideLabel={
                        <Text weight="bold" color="blue">
                          Show less
                        </Text>
                      }
                      showLabel={
                        <Text weight="bold" color="blue">
                          Show more
                        </Text>
                      }
                    >
                      {result?.selfCheckResult?.questions}
                    </Spoiler>
                </td>
                <td>{result?.selfCheckResult?.expected}</td>
                <td>{result?.selfCheckResult?.actual}</td>
                <td>{result?.selfCheckResult?.grade}</td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </ScrollArea>
  );
};
export default ExperimentResultsTable;
