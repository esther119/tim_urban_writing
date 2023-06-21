// %%
const fs = require('fs');

let data = fs.readFileSync('sample.txt', 'utf8').replace(/\n/g, '');

// %%
const openai_api_key = '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';

// Helper to read local files
const os = require('os');

// Vector Support
// const { FAISS } = require('langchain/vectorstores');
const { FaissStore } = require('langchain/vectorstores/faiss');
const { OpenAIEmbeddings } = require('langchain/embeddings/openai');

// Model and chain
const { ChatOpenAI } = require('langchain/chat_models');

// Text splitters
const { CharacterTextSplitter } = require('langchain/text_splitter');
const { TextLoader } = require('langchain/document_loaders');

const llm = new ChatOpenAI({ model_name: 'gpt-3.5-turbo-16k-0613', openai_api_key });

// %%
const embeddings = new OpenAIEmbeddings({ disallowed_special: [], openai_api_key });

// %%
const { LlamaCppEmbeddings } = require('langchain/embeddings');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
// const { FAISS } = require('langchain/vectorstores');
// const { TextLoader } = require('langchain/document_loaders');

// %%

const loader = new TextLoader('./sample.txt');
const documents = loader.load();
const text_splitter = new RecursiveCharacterTextSplitter({
    chunk_size: 1000,
    chunk_overlap: 50,
    length_function: (str) => str.length
});

const _docs = text_splitter.split_documents(documents); // list of lists of strings

// const db = FAISS.from_documents(docs, embeddings);
const db = FaissStore.fromDocuments(
    docs,
    new OpenAIEmbeddings()
  );

db.save_local("faiss_index");

// %%
// Now let's test it out
const query = "What is first principles?";
const search_docs = db.similarity_search(query);
let store = '';
for (const doc of search_docs) {
    console.log(doc.page_content);
    store += doc.page_content;
}

// %%
// Response style 1
const { RetrievalQA } = require('langchain/chains');

const retriever = db.as_retriever();

// Conversational QA retrieval chain
const qa = RetrievalQA.from_chain_type({ llm: llm, chain_type: "stuff", retriever: retriever });


// template = '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.'''
// template = '''Use the following pieces of context to answer the question at the end. If you don't know the answer, just clarify that you don't know, but still generate new answers based on the tone of the text.'''
let template = '';
let user_input = 'Write a post about apocolypse.';
let final_input = template + store + user_input;
let response = qa.run(final_input);
console.log(`AI: ${response}`);

// while (true) {
//     user_input = input("You: ")
//     if user_input.lower() == "exit":
//         break

//     response = qa.run(user_input)
//     print(f"AI: {response}")
// }

// %%
// Response style 2
const { load_qa_chain } = require('langchain/chains/question_answering');

const chain = load_qa_chain(llm, { chain_type: "refine" });

// %%
response = chain({ "input_documents": search_docs, "question": user_input, "language": "English", "existing_answer": store }, { return_only_outputs: true });


// %%
const pprint = require('pprint');
pprint(response);


// %%