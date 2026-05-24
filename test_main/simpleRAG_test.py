import ollama
import os


"""
After ollma installed, open a terminal and run the following command to download the required models:

> ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
> ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF

* bge-base-en-v1.5 is a general-purpose English text embedding model developed by BAAI (Beijing Academy of Artificial Intelligence).
As part of the BGE series focused on efficient retrieval-augmented generation and embedding, its primary function is to
1) convert text (such as sentences, queries, or documents) into 768-dimensional dense vector embeddings that is
2) suitable for semantic retrieval, clustering, classification, and related natural language processing tasks.

* Llama-3.2-1B-Instruct-GGUF is a quantized, instruction-tuned language model with ~1.24 billion parameters, designed for efficient
local inference in the GGUF format.
1) Meta Llama 3.2, instruction-tuned for dialogue, Q&A, and task-following use cases.
2) Comes in multiple quantized variants ranging from around 0.66GB to 2.48GB depending on quantization method (F16, Q8, Q6, Q4, Q3, IQ3_M, etc.), allowing a trade-off between speed/memory and output quality.
3) Supported languages: English, German, French, Spanish, Italian, Portuguese, Hindi, and Thai
4) Context window: Supports up to 128,000 tokens for handling long text inputs.
5) Typical use: Chatbots, assistants, text generation, content creation, retrieval, summarization, and research tasks.
"""

### get the home directory from the environment variable
home_dir = os.environ.get('HOME')

### Load addtional dataset of facts about cats
fact_dataset = []
with open(f"{home_dir}/projects/python_algo_production/data/experiment/cat-facts.txt", 'r') as file:
  fact_dataset = file.readlines()
  # print(f'Loaded {len(fact_dataset)} entries')

### Implement the retrieval system: 1) embedding model of additional facts,  and 2) a LLM model/chatbot that uses the user's query to get facts and then generate a response.
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

### Create the VECTOR_DB where each element will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
# Each element in the embedding vector captures an abstract feature derived from the text; collectively, these
# features describe the position of the chunk in semantic space.
# Similar texts produce similar embeddings (vectors close together), while dissimilar texts are far apart.
VECTOR_DB = []
# use loaded embedding model to convert each chunk of text into an embedding vector: chunk of text is a string, embedding is a list of floats
def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(fact_dataset):
  add_chunk_to_database(chunk)
  # print(f'Added chunk {i+1}/{len(fact_dataset)} to the database')

# Define a function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

# Define a function to retrieve the most relevant chunks based on a query
# This function takes a query, converts it into an embedding, and then return the top N most similar chunks from the VECTOR_DB based on cosine similarity.
def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort in place by similarity in descending order, because higher similarity means more relevant chunks
  # key=lambda x: x means “use the second element in each tuple (the similarity score) as the sort key”.
  similarities.sort(key=lambda x: x[1], reverse=True)  # Sort by the second item x[1] in each tuple (similarity score), highest first
  # finally, return the top N most relevant chunks
  return similarities[:top_n]


### Let's start the Chatbot with just user query without any additional knowledge
input_query = input('Ask me a question: ')

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': ""},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

## print the response from the chatbot in real-time
print('Chatbot response without RAG:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
print("\n")

### Retrieve relevant knowledge chunks from the VECTOR_DB
retrieved_knowledge = retrieve(input_query)
print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

### Prepare the instruction prompt for the chatbot based on the retrieved knowledge
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
print(f"Instruction Prompt based on query and retriveal knowledge: \n {instruction_prompt}")

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

### print the response from the chatbot in real-time
print('Chatbot response with RAG:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
print("\n")
