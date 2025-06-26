from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time
from dataclasses import dataclass

load_dotenv()

@dataclass
class CategorizationAnalysis:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_time: float
    completion_time: float
    total_time: float
    cost: float
    response: str

class DocumentCategoryAgent:
    def __init__(self, model="mixtral-8x7b-32768", temperature=0, max_tokens=None, timeout=None, max_retries=2, cost_per_token=0.00001):
        """
        Initialize the Document Category Agent with the Mistral AI model and configuration parameters.
        """
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.cost_per_token = cost_per_token

    def analyze_tokens(self, text: str) -> int:
        return len(text.split())  # Rough token count estimation

    def categorize_document(self, input_text: str) -> CategorizationAnalysis:
        """
        Classify the given input text into one of the predefined categories:
        - Bank
        - Finance
        - Receipt
        - Identity
        If none apply, return 'Uncategorized'.
        """
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

        ### Chain-of-Thought Reasoning:
        Please carefully consider the text and think through the category it most likely falls into. 
        - Identify keywords that could be related to financial operations, banking, identification, or receipts.
        - Focus on the structure and content. 
        Only one word should be returned in the output: bank, finance, receipt, identity, or Uncategorized.
        Don't put any other details in the output

        Now, classify the following text:
        {input_text}
        """
        
        # Token analysis
        start_time = time.time()
        prompt_tokens = self.analyze_tokens(system_prompt.format(input_text=input_text))
        queue_start = time.time()

        # Get response
        ai_msg = self.llm.invoke([
            ("system", system_prompt),
            ("human", input_text),
        ])
        response = ai_msg.content.strip()

        queue_time = time.time() - queue_start
        completion_start = time.time()

        # Completion timing and token calculation
        completion_tokens = self.analyze_tokens(response)
        completion_time = time.time() - completion_start

        # Total metrics
        total_time = time.time() - start_time
        total_tokens = prompt_tokens + completion_tokens
        prompt_time = queue_time - completion_time
        cost = total_tokens * self.cost_per_token

        return CategorizationAnalysis(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_time=prompt_time,
            completion_time=completion_time,
            total_time=total_time,
            cost=cost,
            response=response
        )

if __name__ == "__main__":
    categorizer = DocumentCategoryAgent()

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

    # Print analysis
    print("\n=== Token Analysis ===")
    print(f"Prompt Tokens: {analysis.prompt_tokens}")
    print(f"Completion Tokens: {analysis.completion_tokens}")
    print(f"Total Tokens: {analysis.total_tokens}")

    print("\n=== Time Analysis (seconds) ===")
    print(f"Prompt Processing Time: {analysis.prompt_time:.3f}")
    print(f"Completion Time: {analysis.completion_time:.3f}")
    print(f"Total Time: {analysis.total_time:.3f}")

    print("\n=== Cost Analysis ===")
    print(f"Cost of Processing: ${analysis.cost:.6f}")

    print("\n=== Response ===")
    print(f"Document Category: {analysis.response}")
