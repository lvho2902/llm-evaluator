import { ScrollArea, Spoiler, Table, Text } from "@mantine/core";
import { Result } from "../../utils/types";
import renderPassFail from "../../utils/renderPassFail";

const ConsistencyResultTable = ({
  results,
  isFastGradingPrompt,
}: {
  results: any[];
  isFastGradingPrompt: boolean;
}) => {
  return (
    <ScrollArea scrollbarSize={0}>
      <Table withBorder withColumnBorders striped highlightOnHover>
        <thead>
          <tr>
            <th>Question</th>
            <th>Variant Questions</th>
            <th>Model Answers</th>
            <th>Grade Result</th>
          </tr>
        </thead>
        <tbody>
          {results?.map((result: Result, index: number) => {
            return (
              <tr key={index}>
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
                      {result?.question}
                    </Spoiler>
                </td>
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
                      {result?.consistencyResults?.questions}
                    </Spoiler>
                </td>
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
                      {result?.consistencyResults?.answers}
                    </Spoiler>
                </td>
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
                      {result?.consistencyResults?.results}
                    </Spoiler>
                </td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </ScrollArea>
  );
};
export default ConsistencyResultTable;
