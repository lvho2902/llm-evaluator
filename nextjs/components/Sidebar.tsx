import { ScrollArea, Select, Slider, Stack, Text } from "@mantine/core";
import React from "react";
import { Form } from "../utils/types";
import { Controller, useForm } from "react-hook-form";

const Sidebar = ({ form }: { form: Form }) => {
  const { control, setValue } = form;

  return (
    <>
      <ScrollArea scrollbarSize={0}>
        <Text fz="xl">Parameters</Text>
        <Stack
          spacing="30px"
          style={{
            overflowX: "hidden",
            height: "100%",
            paddingRight: "15px",
            paddingLeft: "5px",
            paddingTop: "15px",
          }}
        >
          <div>
            <Text fz="md">Number of questions</Text>
            <Controller
              name="number_of_question"
              control={control}
              render={({ field }) => (
                <Slider
                  {...field}
                  labelTransition="skew-down"
                  marks={[
                    { value: 1, label: "1" },
                    { value: 5, label: "5" },
                    { value: 10, label: "10" },
                    { value: 15, label: "15" },
                    { value: 20, label: "20" },
                  ]}
                  max={20}
                  min={1}
                  step={1}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Chunk size</Text>
            <Controller
              name="chunk_size"
              control={control}
              render={({ field }) => (
                <Slider
                  {...field}
                  labelTransition="skew-down"
                  marks={[
                    { value: 500, label: "500" },
                    { value: 1000, label: "1000" },
                    { value: 1500, label: "1500" },
                    { value: 2000, label: "2000" },
                    { value: 2500, label: "2500" },
                    { value: 3000, label: "3000" },
                  ]}
                  max={3000}
                  min={500}
                  step={100}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Chunk overlap</Text>
            <Controller
              name="chunk_overlap"
              control={control}
              render={({ field }) => (
                <Slider
                  {...field}
                  labelTransition="skew-down"
                  marks={[
                    { value: 0, label: "0" },
                    { value: 50, label: "50" },
                    { value: 100, label: "100" },
                    { value: 150, label: "150" },
                  ]}
                  max={150}
                  min={0}
                  step={10}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Model</Text>
            <Controller
              name="model"
              control={control}
              render={({ field }) => (
                <Select
                  {...field}
                  data={[
                    { label: "Mistral 7B", value: "mistral" },
                    { label: "Llama 3 8B", value: "llama3" },
                    { label: "Llama 3.1 8B", value: "llama3.1" },
                    { label: "Llama 2 13B", value: "llama2:13b" },
                    { label: "GPT 3.5 Turbo", value: "gpt-3.5-turbo" },
                    { label: "GPT 4", value: "gpt-4" },
                    { label: "Eden GPT 3.5 Turbo Instruct", value: "eden-gpt-3.5-turbo-instruct" },
                  ]}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Evaluator</Text>
            <Controller
              name="evaluator_model"
              control={control}
              render={({ field }) => (
                <Select
                  {...field}
                  data={[
                    { label: "Mistral 7B", value: "mistral" },
                    { label: "Llama 3 8B", value: "llama3" },
                    { label: "Llama 3.1 8B", value: "llama3.1" },
                    { label: "Llama 2 13B", value: "llama2:13b" },
                    { label: "Eden GPT 3.5 Turbo Instruct", value: "eden-gpt-3.5-turbo-instruct" },
                    { label: "OpenAI", value: "openai" },
                  ]}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Split method</Text>
            <Controller
              name="split_method"
              control={control}
              render={({ field }) => (
                <Select
                  {...field}
                  data={[
                    {
                      label: "CharacterTextSplitter",
                      value: "CharacterTextSplitter",
                    },
                    {
                      label: "RecursiveTextSplitter",
                      value: "RecursiveTextSplitter",
                    },
                  ]}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Embedding provider</Text>
            <Controller
              name="embedding_provider"
              control={control}
              render={({ field }) => (
                <Select
                  {...field}
                  data={[
                    {
                      label: "Ollama",
                      value: "Ollama",
                    },
                    {
                      label: "OpenAI",
                      value: "OpenAI",
                    }
                  ]}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Retriever</Text>
            <Controller
              name="retriever_type"
              control={control}
              render={({ field }) => (
                <Select
                  {...field}
                  //onChange={(value) => {
                    // field.onChange(value);
                    // if (value === "Anthropic-100k") {
                    //   setValue("model", "anthropic");
                    //   setValue("splitMethod", "");
                    //   setValue("embeddingProvider", ""); 
                    // }
                  // }}
                  data={[
                    {
                      label: "Similarity Search",
                      value: "similarity-search",
                    },
                    {
                      label: "SVM",
                      value: "SVM",
                    },
                    {
                      label: "TF-IDF",
                      value: "TF-IDF",
                    }
                  ]}
                />
              )}
            />
          </div>
          <div>
            <Text fz="md">Number of chunks to retrieve</Text>
            <Controller
              name="num_neighbors"
              control={control}
              render={({ field }) => (
                <Slider
                  {...field}
                  labelTransition="skew-down"
                  marks={[
                    { value: 3, label: "3" },
                    { value: 4, label: "4" },
                    { value: 5, label: "5" },
                  ]}
                  max={5}
                  min={3}
                  step={1}
                />
              )}
            />
          </div>
        </Stack>
      </ScrollArea>
    </>
  );
};
export default Sidebar;
