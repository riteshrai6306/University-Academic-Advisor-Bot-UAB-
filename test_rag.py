from agents.rag_agent import run_rag_agent

# Test question
question = "What are the admission requirements for Computer Science Engineering at LPU?"
result = run_rag_agent(question)

print("Question:", question)
print("Answer:", result['answer'])
print("Number of chunks retrieved:", len(result['chunks']))
print("Chunks:")
for i, chunk in enumerate(result['chunks']):
    print(f"Chunk {i+1}: {chunk[:200]}...")  # First 200 chars