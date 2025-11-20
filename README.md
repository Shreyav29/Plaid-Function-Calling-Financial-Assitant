📊 Plaid Function-Calling Financial Assistant

A two-stage Gemini LLM pipeline that routes questions, fetches (fake) Plaid data, and returns natural-language financial insights.

🚀 Overview

This project is a prototype of an AI-powered personal finance assistant that uses Gemini function calling to decide when and how to fetch bank transaction data. It demonstrates an agent-like workflow:

Router LLM → Should I call Plaid?

Tool Call → Fetch fake Plaid data

Analyst LLM → Summarize, analyze, and answer

The system currently uses a fake Plaid API (mock data) for low-friction prototyping.
Later, the mock API can be replaced with real Plaid SDK calls.

This project is ideal for learning:

LLM routing logic

Gemini function-calling patterns

Two-stage LLM architecture (planner → executor)

Parsing, transforming, and analyzing financial data

Building a foundation that later integrates banking APIs

Debugging agent pipelines with global state

✨ Key Features
🔹 1. Smart Question Routing (LLM #1)

The first model decides:

If the question can be answered using Plaid data, it triggers a function call

If not, it returns CANNOT_ANSWER_WITH_PLAID

Examples:

Question	Router Output
“How much did I spend on groceries last month?”	Calls get_plaid_transactions
“What is the S&P500?”	CANNOT_ANSWER_WITH_PLAID
“Show my last 10 Starbucks charges.”	Calls get_plaid_transactions
🔹 2. Fake Plaid API (Prototype Only)

Instead of calling real Plaid, this project provides a mock Plaid function that returns:

Dummy accounts

Dummy transactions

Realistic categories, merchants, dates, and amounts

This allows the LLM to run end-to-end without needing credentials or OAuth setup.

🔹 3. Analyst LLM (LLM #2)

After transactions are retrieved, a second LLM run:

Analyzes spending

Groups transactions by category

Computes totals

Answers the question naturally and clearly

Example output:

“You spent $97.50 at restaurants between Oct 1–5, mostly at Uber Eats and Starbucks.”

🔹 4. Global State Debugger

The system keeps a global dictionary storing:

Last user question

Router LLM raw output

Function call arguments

Mock Plaid results

Analysis prompt

Final LLM answer

This makes the pipeline fully transparent and easy to debug.

🧩 Architecture
User Question
      │
      ▼
┌──────────────────────────┐
│   LLM #1: Router Model   │
│  (Decides: Plaid or Not) │
└──────────┬───────────────┘
           │
   if Plaid relevant
           ▼
┌──────────────────────────┐
│  get_plaid_transactions  │   ← Fake Plaid API
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   LLM #2: Analyst Model  │
│ (Explain + Summarize)    │
└──────────────────────────┘
           │
           ▼
      Final Answer

🛠 Tech Stack

Python 3.10+

Google Gemini API (google-genai)

Function Calling

Mock Plaid API

CLI interface

📁 File Structure
project/
│
├── plaid_assistant.py     # Main script (router, tools, analyst, orchestrator)
├── README.md              # Documentation
└── requirements.txt       # Dependencies

▶️ Usage

Install dependencies:

pip install google-genai python-dotenv


Set your Gemini API key:

export GEMINI_API_KEY="your_key_here"


Run the assistant:

python plaid_assistant.py


Example interaction:

You: How much did I spend on restaurants last month?
Assistant: You spent $97.50…

🔮 Future Enhancements

Replace fake Plaid API with real Plaid SDK calls

Add more tools:

get_accounts()

get_balances()

summarize_expenses()

detect_recurring_subscriptions()

Add a web UI (Streamlit or React)

Add multi-tool planning

Add budgets + alerts

Add multi-account support

📘 Summary

This project is a clean learning template for building agent-like LLM systems with tool calling, decomposition, and financial data analysis. It provides an extensible foundation for a full personal finance AI assistant.
