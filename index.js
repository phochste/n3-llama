import { Ollama } from "@langchain/community/llms/ollama";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "langchain/embeddings/tensorflow";
import { RetrievalQAChain } from "langchain/chains";

// Read from web
const loader = new CheerioWebBaseLoader("https://lib.ugent.be/en/catalog/rug01:000763774?i=0&q=liber+floridus");
const data = await loader.load();

// Split the text into 500 character chunks. And overlap each chunk by 20 characters
const textSplitter = new RecursiveCharacterTextSplitter({
 chunkSize: 500,
 chunkOverlap: 20
});
const splitDocs = await textSplitter.splitDocuments(data);

// Then use the TensorFlow Embedding to store these chunks in the datastore
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new TensorFlowEmbeddings());

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama3",
});

const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
const result = await chain.call({query: "Who was the author of the liber floridus and in what year it was published?"});
console.log(result.text)

