from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentCategoryAgentGemini:
    def __init__(self, model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None, max_retries=2):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens if max_tokens else 2048,
        )

    def categorize_document(self, input_text: str) -> str:
        """
        Classify bank documents into specific subcategories using chain-of-thought reasoning.
        Categories: Credit Card, Debit Card, Fixed Deposit, KYC, Loan, Passbook
        """
        system_prompt = """
        You are an expert bank document classifier. Your task is to analyze the given text and determine the specific type of banking document through step-by-step reasoning.

        The possible bank document categories are:
        - Credit Card: Documents related to credit card applications, statements, or terms
        - Debit Card: Documents about debit card issuance, usage, or terms
        - Fixed Deposit: Documents about FD applications, receipts, or statements
        - KYC: Know Your Customer documents, identity verification forms
        - Loan: Loan applications, statements, or agreements
        - Passbook: Bank account transaction history or passbook entries

        ### Analysis Steps:
        1. Identify key banking terms and phrases
        2. Look for specific document identifiers or headers
        3. Analyze the structure and format
        4. Consider the purpose of the document
        5. Conclude with the most specific category

        ### Few-Shot Examples:

        **Example 1 (Credit Card):**
        Input: 
        CREDIT CARD STATEMENT
        Card Number: XXXX-XXXX-XXXX-1234
        Statement Period: 01/03/2024 - 31/03/2024
        Credit Limit: $5,000
        Available Credit: $3,200
        Payment Due Date: 15/04/2024
        Transactions:
        - Grocery Store    $120.50
        - Gas Station     $45.00
        - Online Shopping $250.00

        Reasoning:
        1. Contains "CREDIT CARD STATEMENT" header
        2. Shows credit card number format
        3. Mentions credit limit and available credit
        4. Lists card transactions
        Therefore, this is a: credit_card

        **Example 2 (Debit Card):**
        Input:
        DEBIT CARD ISSUANCE LETTER
        Account Number: 1234567890
        Card Number: XXXX-XXXX-XXXX-5678
        Card Type: Visa Platinum Debit
        Daily ATM Limit: $1,000
        Daily POS Limit: $2,000
        Card Validity: 5 years
        PIN: Will be sent separately

        Reasoning:
        1. Contains "DEBIT CARD" in header
        2. Shows debit card details and limits
        3. Mentions ATM and POS limits
        4. Includes card validity information
        Therefore, this is a: debit_card

        **Example 3 (Fixed Deposit):**
        Input:
        FIXED DEPOSIT RECEIPT
        FD Account Number: FD987654321
        Principal Amount: $10,000
        Interest Rate: 5.5% p.a.
        Tenure: 24 months
        Start Date: 01/04/2024
        Maturity Date: 01/04/2026
        Maturity Amount: $11,155
        Interest Payout: At Maturity

        Reasoning:
        1. Contains "FIXED DEPOSIT" in header
        2. Shows principal amount and interest rate
        3. Specifies tenure and maturity date
        4. Includes maturity amount calculation
        Therefore, this is a: fixed_deposit

        **Example 4 (KYC):**
        Input:
        KNOW YOUR CUSTOMER (KYC) FORM
        Customer ID: CUS123456
        Personal Details:
        - Full Name: John Smith
        - Date of Birth: 01/01/1980
        - Address: 123 Main St, City
        - ID Type: Passport
        - ID Number: AB123456
        Documents Attached:
        - Proof of Identity
        - Proof of Address
        - Recent Photograph

        Reasoning:
        1. Contains "KYC" in header
        2. Requests personal identification details
        3. Lists required documentation
        4. Focus on identity verification
        Therefore, this is a: kyc

        **Example 5 (Loan):**
        Input:
        HOME LOAN APPLICATION
        Application Number: LA789012
        Loan Amount Requested: $200,000
        Purpose: House Purchase
        Term: 20 years
        Interest Rate: 6.5%
        Monthly Payment: $1,500
        Collateral: Property Details
        Income Details:
        - Monthly Income: $5,000
        - Other EMIs: $500

        Reasoning:
        1. Contains "LOAN APPLICATION" in header
        2. Shows loan amount and purpose
        3. Includes term and interest details
        4. Mentions collateral and income details
        Therefore, this is a: loan

        **Example 6 (Passbook):**
        Input:
        SAVINGS ACCOUNT PASSBOOK
        Account Number: SB123456789
        Account Holder: Jane Smith
        Branch: Main Street Branch

        Transaction History:
        Date       | Description      | Debit  | Credit | Balance
        01/03/2024 | Salary Credit   |        | 5000   | 15000
        05/03/2024 | ATM Withdrawal  | 1000   |        | 14000
        10/03/2024 | Bill Payment    | 500    |        | 13500

        Reasoning:
        1. Contains "PASSBOOK" in header
        2. Shows account transaction history
        3. Includes credits and debits
        4. Maintains running balance
        Therefore, this is a: passbook

        Now, analyze the following text step by step and determine the specific type of bank document:
        {input_text}

        Output should be exactly one of: credit_card, debit_card, fixed_deposit, kyc, loan, passbook
        """
        
        messages = [
            ("system", system_prompt),
            ("human", input_text),
        ]
        
        ai_msg = self.llm.invoke(messages)
        return ai_msg.content.strip()

if __name__ == "__main__":
    categorizer = DocumentCategoryAgentGemini()

    sample_text = """
----------------------------------------
         CITYBANK CREDIT CARD
         APPLICATION FORM
         Branch: Downtown Branch
----------------------------------------
Application Date: 2024-03-15
Reference No: CCF-2024-03150123
----------------------------------------
PERSONAL DETAILS
----------------------------------------
Name: Sarah Johnson
Date of Birth: 15/05/1985
Contact: (555) 123-4567
Email: sarah.j@email.com
Annual Income: $75,000

CARD DETAILS REQUESTED
----------------------------------------
Card Type: Platinum Rewards
Credit Limit: $10,000
Rewards Program: Yes
Annual Fee: $99

EMPLOYMENT INFORMATION
----------------------------------------
Employer: Tech Solutions Inc.
Position: Senior Developer
Years Employed: 5
Monthly Income: $6,250

REFERENCES
----------------------------------------
1. John Smith
   Relationship: Friend
   Contact: (555) 012-3456

2. Mary Wilson
   Relationship: Colleague
   Contact: (555) 045-6789
----------------------------------------
FOR OFFICE USE ONLY
Application Status: Pending Review
----------------------------------------
"""

    category = categorizer.categorize_document(sample_text)
    print("Document Category:", category)