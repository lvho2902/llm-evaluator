import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {Group, Text, useMantineTheme, Alert, Table, Button, Title, Flex, Stack, Spoiler, Progress, Card, ScrollArea, createStyles} from "@mantine/core";
import { IconUpload, IconX, IconAlertCircle } from "@tabler/icons-react";
import { Dropzone, MIME_TYPES } from "@mantine/dropzone";
import { Experiment, Form, QAPair, Result } from "../utils/types";
import { notifications } from "@mantine/notifications";
import { API_URL, IS_DEV } from "../utils/variables";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { Parser } from "@json2csv/plainjs";
import { IconFile } from "@tabler/icons-react";
import { ResponsiveScatterPlot } from "@nivo/scatterplot";
import { isEmpty, isNil, orderBy, set } from "lodash";
import TestFileUploadZone from "./TestFileUploadZone";
import LogRocket from "logrocket";
import ConsistencyResultTable from "./tables/ConsistencyResultTable";
import DeepEvalResultTable from "./tables/DeepEvalResultTable";
import ExperimentSummaryTable from "./ExperimentSummaryTable";

import { Pagination } from '@mantine/core';

const MAX_FILE_SIZE_MB = 50;

enum DropZoneErrorCode {
  FileTooLarge = "file-too-large",
  FileInvalidType = "file-invalid-type",
}

const useStyles = createStyles((theme) => ({
  disabled: {
    backgroundColor:
      theme.colorScheme === "dark"
        ? theme.colors.dark[6]
        : theme.colors.gray[0],
    borderColor:
      theme.colorScheme === "dark"
        ? theme.colors.dark[5]
        : theme.colors.gray[2],
    cursor: "not-allowed",

    "& *": {
      color:
        theme.colorScheme === "dark"
          ? theme.colors.dark[3]
          : theme.colors.gray[5],
    },
  },
}));

const Playground = ({ form }: { form: Form }) => {
  const { setValue, watch, getValues, handleSubmit } = form;
  const watch_files = watch("files");
  const theme = useMantineTheme();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [test_dataset, setTestDataset] = useState<QAPair[]>([]);
  const [number_of_question, setNumberOfQuestion] = useState(-1);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selected_experiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [did_upload_test_dataset, setDidUploadTestDataset] = useState(false);
  const [should_show_progress, setShouldShowProgress] = useState(false);
  const consistency_results_spoiler_ref = useRef<HTMLButtonElement>(null);
  const deepeval_results_spoiler_ref = useRef<HTMLButtonElement>(null);
  const summary_spoiler_ref = useRef<HTMLButtonElement>(null);
  const test_dataset_spoiler_ref = useRef<HTMLButtonElement>(null);
  const [test_files_dropzone_disabled, setTestFilesDropzoneDisabled] = useState(true);
  const [file_upload_disabled, setFileUploadDisabled] = useState(false);
  const { classes } = useStyles();

  const [experimentRun, setExperimentRun] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 5;
  const [resultsPage, setResultsPage] = useState(1);
  const [resultsPerPage, setResultsPerPage] = useState(3); // Adjust items per page as needed

  const initialProgress = {value: 15, color: "purple", label: "Processing Files"};
  const finishedProgress = {value: 100, color: "green", label: "Completed"};

  const fetchExperiments = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/experiments`);
      const data: Experiment[] = await response.json();
      setExperiments(data);
      if (experimentRun) {
        const totalPages = Math.ceil(data.length / itemsPerPage);
        setCurrentPage(totalPages);
        setExperimentRun(false); // Reset the flag after navigating
      }
    } catch (error) {
      console.error("Failed to fetch experiments:", error);
    }
  }, [experimentRun, itemsPerPage]);

  const paginatedConsistencyResults = useMemo(() => {
    const startIndex = (resultsPage - 1) * resultsPerPage;
    const endIndex = startIndex + resultsPerPage;
    console.log(resultsPage, startIndex, endIndex)
    return results.slice(startIndex, endIndex);
  }, [results, resultsPage, resultsPerPage]);

  const fetchResults = useCallback(async (experiment: Experiment) => {
    try {
      const response = await fetch(`${API_URL}/evaluations/${experiment?.id}`);
      const data = await response.json();
      const results = data?.map((item: { data: Result }) => item.data);
      setResults(results)
    } catch (error) {
      console.error(`Failed to fetch results for experiment ${experiment?.id}:`, error);
    }
  }, []);

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  useEffect(() => {
    if (selected_experiment !== null) {
      fetchResults(selected_experiment);
    }
  }, [selected_experiment, fetchResults]);

  const experimentProgress = useMemo(() => {
    if (results.length === 0) {return [initialProgress];}
    const res = 15 + Math.floor((results?.length / number_of_question) * 85);
    console.log("rusults len", results?.length)
    console.log("number of questions", number_of_question)
    console.log("res", res)
    if (res == 100) {return [finishedProgress];}
    return [initialProgress, {value: res, color: "blue", label: "Generating Evals & Grading"}];
  }, [results, number_of_question]);

  const submit = handleSubmit(async (data) => {
    setShouldShowProgress(true);
    setLoading(true);
    setResults([]);
    // const resetExpts = data.number_of_question !== number_of_question || did_upload_test_dataset;
    // if (resetExpts) {setExperiments([]);}
    setDidUploadTestDataset(false);
    const formData = new FormData();
    data.files.forEach((file) => {formData.append("files", file);});
    formData.append("test_dataset", JSON.stringify(test_dataset));
    formData.append("number_of_question", data.number_of_question.toString());
    formData.append("chunk_size", data.chunk_size.toString());
    formData.append("chunk_overlap", data.chunk_overlap.toString());
    formData.append("split_method", data.split_method);
    formData.append("retriever_type", data.retriever_type);
    formData.append("embedding_provider", data.embedding_provider);
    formData.append("model", data.model);
    formData.append("evaluator_model", data.evaluator_model);
    formData.append("num_neighbors", data.num_neighbors.toString());

    if (!IS_DEV) {
      LogRocket.track("PlaygroundSubmission", {
        file_size: data.files.map((file) => file.size),
        file_type: data.files.map((file) => file.type),
        number_of_question: data.number_of_question,
        chunk_overlap: data.chunk_overlap,
        split_method: data.split_method,
        retriever_type: data.retriever_type,
        embedding_provider: data.embedding_provider,
        model: data.model,
        evaluator_model: data.evaluator_model,
        num_neighbors: data.num_neighbors,
        uploaded_test_dataset: !!test_dataset.length,
      });
    }

    setNumberOfQuestion(data.number_of_question);
    const controller = new AbortController();
    let localResults = [];
    let rowCount = 0;
    try {
      await fetchEventSource(API_URL + "/evaluator-stream", 
        { method: "POST", body: formData, headers: { Accept: "text/event-stream", Connection: "keep-alive" }, openWhenHidden: true, signal: controller.signal,
        onmessage(ev) {
          try {
            const row: Result = JSON.parse(ev?.data)?.data;
            setResults((results) => [...results, row]);
            localResults = [...localResults, row];
            rowCount += 1;
            if (rowCount > test_dataset.length) {
              setTestDataset((testDataset) => [
                ...testDataset,
                {question: row.question, answer: row.expected},
              ]);
            }
            if (rowCount === data.number_of_question) {controller.abort();}
          } catch (e) {
            console.warn("Error parsing data", e);
          }
        },
        onclose() {
          console.log("Connection closed by the server");
          setLoading(false);
          if (!rowCount) throw new Error("No results were returned from the server.");
        },
        onerror(err) {
          console.log("There was an error from server", err);
          throw new Error(err);
        },
      });
    } catch (e) {
      notifications.show({ title: "Error", message: "There was an error from the server.", color: "red"});
      setShouldShowProgress(false);
      setLoading(false);
      return;
    }
    setLoading(false);
    setExperimentRun(true);
    fetchExperiments();
  });
  const runExperimentButtonLabel = experiments?.length
    ? "Re-run experiment"
    : "Run Experiment";
  const download = useCallback(
    (data: any[], filename: string) => {
      const parser = new Parser();
      const csv = parser.parse(data);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", filename + ".csv");
      link.style.visibility = "hidden";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    },
    [results]
  );
  const indexOfLastExperiment = currentPage * itemsPerPage;
  const indexOfFirstExperiment = indexOfLastExperiment - itemsPerPage;
  const paginatedExperiments = experiments.slice(indexOfFirstExperiment, indexOfLastExperiment);
  const alertStyle = { backgroundColor: `rgba(193,194,197,0.38)` };
  return (
    <Stack>
      <Alert icon={<IconAlertCircle size="1rem" />} title="Instructions" style={alertStyle}>
        Upload a file (up to 50 MB) and choose the parameters for your QA chain. This evaluator will generate a test dataset of QA pairs
        and grade the performance of the QA chain. You can experiment with different
        parameters and evaluate the performance.
      </Alert>
      <Flex direction="row" gap="md">
        <Dropzone
          disabled={file_upload_disabled}
          className={file_upload_disabled ? classes.disabled : null}
          onDrop={(files) => {
            setValue("files", [...(getValues("files") ?? []), ...files]);
            setShouldShowProgress(false);
            setTestFilesDropzoneDisabled(false);
            setFileUploadDisabled(true);
          }}
          maxFiles={1}
          multiple={false}
          maxSize={MAX_FILE_SIZE_MB * 1024 ** 2} // 50 MB
          accept={[
            MIME_TYPES.pdf,
            MIME_TYPES.docx,
            MIME_TYPES.doc,
            "text/plain",
          ]}
          onReject={(files) => {
            const errorCode = files?.[0]?.errors?.[0]?.code;
            let message = files?.[0]?.errors?.[0]?.message;
            switch (errorCode) {
              case DropZoneErrorCode.FileTooLarge:
                message = `File size too large. Max file size is ${MAX_FILE_SIZE_MB} MB.`;
                break;
              case DropZoneErrorCode.FileInvalidType:
                message = "File type not supported";
                break;
              default:
                break;
            }
            notifications.show({title: "Error", message, color: "red"});
          }}
          style={{ width: "100%" }}
        >
          <Stack align="center">
            <Dropzone.Accept>
              <IconUpload
                size="3.2rem"
                stroke={1.5}
                color={theme.colors[theme.primaryColor][theme.colorScheme === "dark" ? 4 : 6]}
              />
            </Dropzone.Accept>
            <Dropzone.Reject>
              <IconX
                size="3.2rem"
                stroke={1.5}
                color={theme.colors.red[theme.colorScheme === "dark" ? 4 : 6]}
              />
            </Dropzone.Reject>
            <Dropzone.Idle>
              <IconFile size="3.2rem" stroke={1.5} />
            </Dropzone.Idle>
            <div>
              <Text size="xl" inline align="center">
                Upload Text for QA Evaluation
              </Text>
              <Text size="sm" color="dimmed" mt={7} align="center">
                {"Attach a file (.txt, .pdf, .doc, .docx)"}
              </Text>
            </div>
          </Stack>
        </Dropzone>
        <TestFileUploadZone
          disabled={test_files_dropzone_disabled}
          setTestDataset={setTestDataset}
          setDidUploadTestDataset={setDidUploadTestDataset}
        />
      </Flex>
      {!!watch_files?.length && (
        <>
          <Table>
            <thead>
              <tr>
                <th>File Name</th>
                <th>Size (MB)</th>
              </tr>
            </thead>
            <tbody>
              {watch_files?.map((file, id) => (
                <tr key={id}>
                  <td>{file?.name}</td>
                  <td>{(file?.size / 1024 ** 2).toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </Table>
          {!!test_dataset.length && (
            <Card>
              <Spoiler
                maxHeight={0}
                showLabel="Show available test dataset"
                hideLabel={null}
                transitionDuration={500}
                controlRef={test_dataset_spoiler_ref}
              >
                <Stack>
                  <Group position="apart">
                    <Title order={3}>Test Dataset</Title>
                    <Group>
                      <Button
                        style={{ marginBottom: "18px" }}
                        type="button"
                        variant="secondary"
                        onClick={() => download(test_dataset, "test_dataset")}>
                        Download
                      </Button>
                      <Button
                        style={{ marginBottom: "18px" }}
                        type="button"
                        variant="subtle"
                        onClick={() => {
                          setTestDataset([]);
                          notifications.show({title: "Success", message: "The test dataset has been cleared.", color: "green"});
                        }}
                      >
                        Reset
                      </Button>
                      <Button
                        style={{ marginBottom: "18px" }}
                        type="button"
                        variant="subtle"
                        onClick={() => {if (test_dataset_spoiler_ref.current) test_dataset_spoiler_ref.current.click();}}>
                        Hide
                      </Button>
                    </Group>
                  </Group>
                </Stack>
                <Table withBorder withColumnBorders striped highlightOnHover>
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th>Answer</th>
                    </tr>
                  </thead>
                  <tbody>
                    {test_dataset?.map((result: QAPair, index: number) => (
                      <tr key={index}>
                        <td>{result?.question}</td>
                        <td>{result?.answer}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Spoiler>
            </Card>
          )}
          <Flex direction="row" gap="md">
            <Button
              style={{ marginBottom: "18px" }}
              type="submit"
              onClick={submit}
              disabled={loading}>
              {runExperimentButtonLabel}
            </Button>
          </Flex>
        </>
      )}
      {should_show_progress && (
        <Progress
          size="xl"
          radius="xl"
          sections={experimentProgress}
          color={loading ? "blue" : "green"}
        />
      )}
        <Card>
          <Spoiler
            maxHeight={0}
            showLabel="Show summary"
            hideLabel={null}
            transitionDuration={500}
            initialState={true}
            controlRef={summary_spoiler_ref}
          >
            <Stack>
              <Group position="apart">
                <Title order={3}>Summary</Title>
                <Group>
                  <Button
                    style={{ marginBottom: "18px" }}
                    type="button"
                    variant="secondary"
                    onClick={() => download(experiments, "summary")}
                  >
                    Download
                  </Button>
                  <Button
                    style={{ marginBottom: "18px" }}
                    type="button"
                    variant="subtle"
                    onClick={() => {if (summary_spoiler_ref.current) summary_spoiler_ref.current.click();}}>
                    Hide
                  </Button>
                </Group>
              </Group>
            </Stack>
            <ExperimentSummaryTable
              experiments={paginatedExperiments}
              onRowClick={(experiment) => 
                {
                  setSelectedExperiment(experiment)
                  setNumberOfQuestion(experiment.number_of_question)
                  setShouldShowProgress(false)
                }
              }
            />
            <Flex justify="center" mt="md">
              <Pagination
                value={currentPage}
                onChange={setCurrentPage}
                total={Math.ceil(experiments.length / itemsPerPage)}
              />
            </Flex>
          </Spoiler>
        </Card>
        <Card>
          <Spoiler
            maxHeight={0}
            showLabel="Show consistency results"
            hideLabel={null}
            transitionDuration={500}
            initialState={true}
            controlRef={consistency_results_spoiler_ref}
          >
            <Stack>
              <Group position="apart">
                <Title order={3}>Consistency Results</Title>
                <br />
                <br />
                <Group>
                  <Button
                    style={{ marginBottom: "18px" }}
                    type="button"
                    variant="subtle"
                    onClick={() => download(results.map(result => result.consistency), "consistency_results")}
                  >
                    Download
                  </Button>
                  <Button
                    style={{ marginBottom: "18px" }}
                    type="button"
                    variant="subtle"
                    onClick={() => {if (consistency_results_spoiler_ref.current) consistency_results_spoiler_ref.current.click();}}>
                    Hide
                  </Button>
                </Group>
              </Group>
            </Stack>
            <ConsistencyResultTable results={paginatedConsistencyResults}/>
            <Flex justify="center" mt="md">
              <Pagination
                value={resultsPage}
                onChange={setResultsPerPage}
                total={Math.ceil(results.length / resultsPerPage)}
              />
            </Flex>
          </Spoiler>
        </Card>
        <Card>
          <Spoiler
            maxHeight={0}
            showLabel="Show deep eval results"
            hideLabel={null}
            transitionDuration={500}
            initialState={true}
            controlRef={deepeval_results_spoiler_ref}
          >
            <Stack>
              <Group position="apart">
                <Title order={3}>Deep Eval Results</Title>
                <br />
                <br />
                <Group>
                  <Button
                    style={{ marginBottom: "18px" }}
                    type="button"
                    variant="subtle"
                    onClick={() => download(results.map(result => result?.deepeval), "deep_eval_results")}
                  >
                    Download
                  </Button>
                  <Button
                    style={{ marginBottom: "18px" }}
                    type="button"
                    variant="subtle"
                    onClick={() => {if (deepeval_results_spoiler_ref.current) deepeval_results_spoiler_ref.current.click();}}>
                    Hide
                  </Button>
                </Group>
              </Group>
            </Stack>
            <DeepEvalResultTable results={paginatedConsistencyResults}/>
            <Flex justify="center" mt="md">
              <Pagination
                value={resultsPage}
                onChange={setResultsPerPage}
                total={Math.ceil(results.length / resultsPerPage)}
              />
            </Flex>
          </Spoiler>
        </Card>
    </Stack>
  );
};
export default Playground;
