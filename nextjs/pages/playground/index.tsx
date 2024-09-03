import { AppShell, Navbar } from "@mantine/core";
import React from "react";
import { useForm } from "react-hook-form";
import HeaderEvaluator, { MenuItem } from "../../components/HeaderEvaluator";
import Sidebar from "../../components/Sidebar";
import { FormValues } from "../../utils/types";
import Playground from "../../components/Playground";

const PlaygroundPage = () => {
  const form = useForm<FormValues>({
    defaultValues: {
      files: [],
      number_of_question: 1,
      chunk_size: 2000,
      chunk_overlap: 150,
      split_method: "RecursiveTextSplitter",
      embedding_provider: "Ollama",
      retriever_type: "similarity-search",
      model: "mistral",
      evaluator_model: "llama3.1",
      num_neighbors: 3
    },
  });

  return (
    <AppShell
      navbarOffsetBreakpoint="sm"
      navbar={
        <Navbar p="md" hiddenBreakpoint="sm" width={{ sm: 200, lg: 300 }}>
          <Sidebar form={form} />
          <br />
        </Navbar>
      }
      header={<HeaderEvaluator activeTab={MenuItem.Playground} />}
      styles={(theme) => ({
        main: {
          backgroundColor:
            theme.colorScheme === "dark"
              ? theme.colors.dark[8]
              : theme.colors.gray[0],
        },
      })}
    >
      <Playground form={form} />
    </AppShell>
  );
};
export default PlaygroundPage;
