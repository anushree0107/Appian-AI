from langchain_google_genai import ChatGoogleGenerativeAI
import tiktoken
import os
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

class DocumentCategoryAgentGemini:
    def __init__(self, model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        """
        Initialize the Document Category Agent with the Mistral AI model and configuration parameters.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else 2048,
        )
        # Initialize the tokenizer
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or another appropriate encoding

    def analyze_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    

    def categorize_document(self, input_text: str) -> TokenAnalysis:
        """
        Classify the given input text into predefined categories and return the token count.
        Returns a tuple of (category, token_count)
        """
        # Initialize timing variables at the start of the method
        start_time = time.time()
        queue_start = time.time()

        system_prompt = """
        You are an expert document classifier assistant. Your task is to analyze the given text and determine its category based on its content.

        The possible categories are:
        - **Bank**: Texts that are related to banking operations, such as applications for accounts, credit card details, or account numbers.
        - **Finance**: Texts related to financial documents like income statements, pay stubs, tax returns, or other monetary transactions.
        - **Receipt**: Texts that resemble transaction records, purchase receipts, or payment acknowledgments.
        - **Identity**: Texts containing personal identification information, such as driver's licenses, passports, or government-issued ID numbers.

        ### Guidelines:
        1. Analyze the overall context, structure, and keywords in the text.
        2. Return the **most relevant category**. If the text does not fit any category, respond with **'Uncategorized'**.

        ### Few-Shot Examples:

        **Example 1:**
        Input: 
        Application for a new checking account with Union Bank of India. 
        Account Number: 1234567890
        Home Branch: Mumbai
        IFSC Code: UBIN123456
        Output: bank

        **Example 2:**
        Input: 
        Gross Monthly Income: $4,500
        Deductions: $1,200
        Net Pay: $3,300
        Income Statement for January 2024
        Output: finance

        **Example 3:**
        Input: 
        Receipt No: 1456
        Date: 2024-08-01
        Amount Paid: $200
        Payment Method: Credit Card
        Description: Purchase of electronics from Tech Store
        Output: receipt

        **Example 4:**
        Input: 
        Driver's License
        Name: John Doe
        License Number: D12345678
        Date of Birth: 1980-01-01
        Issued: California, USA
        Output: identity

        **Example 5:**
        Input: 
        This is a random paragraph discussing the benefits of exercise for mental health. It contains no financial or personal identification information.
        Output: Uncategorized

        Output will be just a single word : bank or finance or identity or receipt
        no other text will be there in output.
        Now, classify the following text:
        {input_text}
        """
        
        messages = [
            ("system", system_prompt),
            ("human", input_text),
        ]

        
        # Calculate prompt tokens
        prompt_tokens = self.analyze_tokens(system_prompt + input_text)
        
        # Record queue time and start completion
        queue_time = time.time() - queue_start
        completion_start = time.time()
        
        # Get response
        ai_msg = self.llm.invoke(messages)
        response = ai_msg.content.strip()
        
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
    categorizer = DocumentCategoryAgentGemini()


    sample_text = """
----------------------------------------
              SHOP NAME
           Address: 123 Main St
           Phone: (123) 456-7890
----------------------------------------
Date: 2023-10-01
Time: 14:30
----------------------------------------
Item                Qty     Price
----------------------------------------
Item 1              2       $10.00
Item 2              1       $5.50
Item 3              3       $7.25
----------------------------------------
Subtotal:                     $32.75
Tax (5%):                    $1.64
----------------------------------------
Total:                      $34.39
----------------------------------------
Thank you for shopping with us!
----------------------------------------
"""
    analysis = categorizer.categorize_document(sample_text)
    
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
    print(f"Category: {analysis.response}")
    
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
    