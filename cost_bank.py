from langchain_google_genai import ChatGoogleGenerativeAI
import tiktoken
import json
import time
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TokenAnalysis:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_time: float
    completion_time: float
    total_time: float
    queue_time: float
    response: str

class DocumentSummaryAgentBank:
    def __init__(self, model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else 2048,
        )
        # Initialize the tokenizer
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def analyze_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def generate_summary(self, knowledge_graph_json: dict) -> TokenAnalysis:
        # Initialize timing variables
        start_time = time.time()
        queue_start = time.time()

        knowledge_graph_str = json.dumps(knowledge_graph_json, indent=2)

        system_prompt = """
        You are a highly skilled assistant specializing in extracting structured information from a JSON-based Knowledge Graph (KG) and summarizing it.

        The input will be a JSON knowledge graph that represents a document. You need to:
        1. Identify the **type** of document (e.g., KYC, Loan, Passbook, etc.).
        2. Provide a **bulleted list** of the key points extracted from the JSON knowledge graph.
        3. Write a **summary** of the document.

        Your output should contain the following:
        - **Document Type**: The type of document.
        - **Key Points (Bulleted List)**: A list of important information extracted from the knowledge graph.
        - **Summary**: A concise summary of the document.
        
        Do not include any extra commentary or explanations in the output. Keep the structure clean and relevant to the document.
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Knowledge Graph:\n\n{knowledge_graph_str}"),
        ]

        # Calculate prompt tokens
        prompt_tokens = self.analyze_tokens(system_prompt + knowledge_graph_str)
        
        # Record queue time and start completion
        queue_time = time.time() - queue_start
        completion_start = time.time()
        
        # Get response
        ai_msg = self.llm.invoke(messages)
        response = ai_msg.content

        # Calculate completion tokens and time
        completion_tokens = self.analyze_tokens(response)
        completion_time = time.time() - completion_start

        # Calculate total metrics
        total_tokens = prompt_tokens + completion_tokens
        total_time = time.time() - start_time
        prompt_time = completion_start - start_time - queue_time

        return TokenAnalysis(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_time=prompt_time,
            completion_time=completion_time,
            total_time=total_time,
            queue_time=queue_time,
            response=response
        )

if __name__ == "__main__":
    doc_summary_agent = DocumentSummaryAgentBank()

    knowledge_graph_example = {
        "graph_type": "KYC",
        "entities": [
            {"name": "John Doe", "type": "Person", "attributes": {"DOB": "1985-05-12", "Gender": "Male", "Nationality": "Canadian"}},
            {"name": "John's Passport", "type": "Document", "attributes": {"Passport Number": "AB1234567", "Issue Date": "2020-01-15", "Expiry Date": "2030-01-15"}},
            {"name": "Address", "type": "Location", "attributes": {"Street": "123 Maple Street", "City": "Toronto", "Country": "Canada"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "John's Passport", "relationship": "Holds"},
            {"from": "John Doe", "to": "Address", "relationship": "Lives At"}
        ]
    }

    analysis = doc_summary_agent.generate_summary(knowledge_graph_example)
    
    # Print detailed analysis
    print("\n=== Token Analysis ===")
    print(f"Prompt Tokens: {analysis.prompt_tokens}")
    print(f"Completion Tokens: {analysis.completion_tokens}")
    print(f"Total Tokens: {analysis.total_tokens}")
    
    print("\n=== Time Analysis (seconds) ===")
    print(f"Queue Time: {analysis.queue_time:.3f}")
    print(f"Prompt Processing Time: {analysis.prompt_time:.3f}")
    print(f"Completion Time: {analysis.completion_time:.3f}")
    print(f"Total Time: {analysis.total_time:.3f}")
    
    print("\n=== Response ===")
    print(f"Summary:\n{analysis.response}")
    
    # Optional: Calculate estimated cost (adjust rates as needed)
    INPUT_COST_PER_1K = 0.00025  # Example rate for input tokens
    OUTPUT_COST_PER_1K = 0.0005  # Example rate for output tokens
    
    input_cost = (analysis.prompt_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (analysis.completion_tokens / 1000) * OUTPUT_COST_PER_1K
    total_cost = input_cost + output_cost
    
    print("\n=== Cost Analysis (USD) ===")
    print(f"Input Cost: ${input_cost:.6f}")
    print(f"Output Cost: ${output_cost:.6f}")
    print(f"Total Cost: ${total_cost:.6f}") 