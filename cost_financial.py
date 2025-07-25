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

class FinancialSummaryAgent:
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

    def generate_summary(self, financial_knowledge_graph_json: dict) -> TokenAnalysis:
        # Initialize timing variables
        start_time = time.time()
        queue_start = time.time()

        financial_knowledge_graph_str = json.dumps(financial_knowledge_graph_json, indent=2)

        system_prompt = """
        You are a highly skilled assistant that specializes in understanding and summarizing financial documents based on structured knowledge graphs. 
        Your task involves two main parts:

        1. **Identifying the type of financial document**: 
            - Based on the provided entities and relationships, determine whether the document is a Loan Agreement, Investment Agreement, Non-Disclosure Agreement (NDA), 
              Purchase Agreement, Employment Contract, Partnership Agreement, Lease Agreement, Credit Agreement, Divorce Settlement Agreement, M&A Agreement, 
              Warranty Agreement, Supply Agreement, Guarantee Agreement, Stock Option Agreement, Debt Settlement Agreement, or another type of financial document. 
            - Use **Chain of Thought (COT) reasoning** to analyze the entities, their attributes, and relationships in the knowledge graph to infer the document type. 
            - **Do not rely on any explicit type information** in the knowledge graph (e.g., "graph_type" attribute). Instead, reason step-by-step about the document type based on its structure.

        2. **Extracting key points**:
            - After identifying the document type, extract important details and present them in a **bulleted list**.
            - This should include key entities, their attributes, and relationships that are critical to understanding the financial document. 
            - Use the entities and relationships in the knowledge graph to build a clear and concise list of essential points.

        Your response should be structured as follows:
        - **Document Type**: The type of financial document.
        - **Key Points (Bulleted List)**: A list of the most important information extracted from the knowledge graph.
        - **Summary**: A concise summary of the document's contents, based on the extracted key points.
        
        ** Don't put your COT thoughts in the response. Just give the key points and summary. **
        """

        messages = [
            ("system", system_prompt),
            ("human", f"Financial Knowledge Graph:\n\n{financial_knowledge_graph_str}"),
        ]

        # Calculate prompt tokens
        prompt_tokens = self.analyze_tokens(system_prompt + financial_knowledge_graph_str)
        
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
    financial_summary_agent = FinancialSummaryAgent()

    # Test with loan agreement
    loan_agreement_knowledge_graph = {
        "entities": [
            {"name": "John Doe", "type": "Borrower", "attributes": {"Loan Amount": "$500,000", "Interest Rate": "5%", "Loan Term": "10 years"}},
            {"name": "XYZ Bank", "type": "Lender", "attributes": {"Loan Amount": "$500,000", "Interest Rate": "5%", "Loan Term": "10 years", "Repayment Schedule": "Monthly"}},
            {"name": "Collateral", "type": "Loan Clause", "attributes": {"Collateral": "Property in Downtown"}},
        ],
        "relationships": [
            {"from": "John Doe", "to": "XYZ Bank", "relationship": "Borrows from"},
            {"from": "XYZ Bank", "to": "Loan Clause", "relationship": "Secured by"},
        ]
    }

    analysis = financial_summary_agent.generate_summary(loan_agreement_knowledge_graph)
    
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