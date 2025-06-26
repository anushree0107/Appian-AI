import re
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import tiktoken
import time
from dataclasses import dataclass
from typing import Dict, Any
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache


set_llm_cache(InMemoryCache())

def clean_json_string(input_string):
    cleaned_string = re.sub(r"```json|```", '', input_string)
    return cleaned_string.strip()

def load_as_json(input_string):
    cleaned_string = clean_json_string(input_string)
    return json.loads(cleaned_string.strip())

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

class KnowledgeGraphAgent:
    def __init__(self, model="mixtral-8x7b-32768", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Replace with the appropriate model for tokenization

    def analyze_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def generate_knowledge_graph(self, text: str, graph_type: str) -> TokenAnalysis:
        # Start timers
        start_time = time.time()
        queue_start = time.time()

        system_prompt = f"""
        You are a highly skilled assistant specializing in extracting structured information from unstructured text and organizing it into a JSON-based Knowledge Graph (KG). Your task is to analyze the provided unstructured text and identify key entities, attributes, and relationships within the content. You will then generate a JSON representation of the information, ensuring it includes the most relevant data.

        The input will also specify the type of Knowledge Graph to generate, which is provided as `graph_type`. The graph type can be one of the following:

        1. **Bank Application KG**: Personal details, identity proof, contact information, employment details, and card selection.
        2. **Identity Document KG**: Personal details and document details.
        3. **Financial Statement KG**: Financial summary, transactions, and key metrics.
        4. **Receipt or Invoice KG**: Store details, purchase details, and payment information.

        Based on the provided graph type, you should extract and organize the relevant information accordingly.

        Your output should be a structured JSON object that represents the information extracted from the unstructured text, with the following attributes:
        - `graph_type`: The type of knowledge graph (e.g., "Bank Application KG").
        - `attributes`: Relevant attributes and relationships between entities, categorized based on the `graph_type`.
        - `relevant_information`: The relevant key-value pairs extracted from the text in dictionary format.
        """
        messages = [
            ("system", system_prompt),
            ("human", f"Graph Type: {graph_type}\n\nText: {text}"),
        ]

        # Analyze tokens for the prompt
        prompt_tokens = self.analyze_tokens(system_prompt + f"Graph Type: {graph_type}\n\nText: {text}")
        
        # Record queue time and start the completion process
        queue_time = time.time() - queue_start
        completion_start = time.time()
        
        # Get response from the LLM
        ai_msg = self.llm.invoke(messages)
        response = ai_msg.content

        # Analyze completion tokens
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
    kg_agent = KnowledgeGraphAgent()

    bank_application_text = """
    Karina Richards has applied for a Union Bank of India credit card, as indicated by the provided application details. While the application includes personal information like her name, birthdate (September 9, 1970), and PAN number (I1570), several fields are incomplete...
    """
    
    analysis = kg_agent.generate_knowledge_graph(bank_application_text, graph_type="Bank Application KG")
    
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
    print(f"Knowledge Graph:\n{analysis.response}")
    
    # Cost estimation (adjust rates as needed)
    INPUT_COST_PER_1K = 0.00025  # Example input token rate
    OUTPUT_COST_PER_1K = 0.0005  # Example output token rate
    
    input_cost = (analysis.prompt_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (analysis.completion_tokens / 1000) * OUTPUT_COST_PER_1K
    total_cost = input_cost + output_cost

    print("\n=== Cost Analysis (USD) ===")
    print(f"Input Cost: ${input_cost:.6f}")
    print(f"Output Cost: ${output_cost:.6f}")
    print(f"Total Cost: ${total_cost:.6f}")
