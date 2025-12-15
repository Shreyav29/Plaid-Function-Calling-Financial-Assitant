# 📊 Plaid Function-Calling Financial Assistant

A two-stage Gemini LLM pipeline that routes user questions, fetches (mock) Plaid data, and returns natural-language financial insights.

---

## 🚀 Overview

This project is a prototype of an AI-powered personal finance assistant that uses Gemini function calling to decide when and how to fetch bank transaction data.

It demonstrates an agent-like workflow:

- Router LLM → Decide if Plaid data is needed
- Tool Call → Fetch fake Plaid transactions
- Analyst LLM → Analyze and answer

NOTE: The system currently uses a fake Plaid API (mock data) for friction-free prototyping.
This can later be replaced with real Plaid SDK calls.

---

## 🎯 What This Project Is For

- Learning LLM routing logic
- Gemini function-calling patterns
- Two-stage LLM architecture (planner → executor)
- Parsing and analyzing financial transaction data
- Debugging agent pipelines with global state
- Building a base for real banking API integration

---

## ✨ Key Features

### 1. Smart Question Routing (LLM #1)

The first model decides:
- If the question requires Plaid data → triggers a function call
- Otherwise → returns CANNOT_ANSWER_WITH_PLAID

Examples:

User Question: How much did I spend on groceries last month?
Router Output: Calls get_plaid_transactions

User Question: What is the S&P 500?
Router Output: CANNOT_ANSWER_WITH_PLAID

User Question: Show my last 10 Starbucks charges
Router Output: Calls get_plaid_transactions

---

### 2. Fake Plaid API (Prototype Only)

Instead of calling real Plaid endpoints, this project includes a mock Plaid function that returns:
- Dummy accounts
- Dummy transactions
- Realistic merchants, categories, dates, and amounts

This enables end-to-end testing without OAuth or credentials.

---

### 3. Analyst LLM (LLM #2)

After transactions are retrieved, a second LLM run:
- Groups transactions by category
- Computes totals
- Detects patterns
- Answers the question clearly

Example output:
You spent $97.50 at restaurants between Oct 1–5, mostly at Uber Eats and Starbucks.

---

### 4. Global State Debugger

The system maintains a global dictionary containing:
- Last user question
- Router LLM raw output
- Function call arguments
- Mock Plaid results
- Analyst prompt
- Final LLM answer

This makes the pipeline transparent and easy to debug.

---

## 🧩 Architecture

User Question  
↓  
Router LLM (LLM #1)  
- If NOT Plaid related → `CANNOT_ANSWER_WITH_PLAID`  
- If Plaid related → continue  

↓  
get_plaid_transactions (Mock Plaid API)  

↓  
Analyst LLM (LLM #2)  

↓  
Final Natural-Language Answer

![Architecture Diagram](assets/architecture.png)

---

## 🛠 Tech Stack

- Python 3.10+
- Google Gemini API (google-genai)
- Function Calling
- Mock Plaid API
- CLI interface

---

## 📁 File Structure

project/
  ├── plaid_assistant.py     # Main script (router, tools, analyst, orchestrator)
  ├── README.md              # Documentation
  └── requirements.txt       # Dependencies

---

## ▶️ Usage

Install dependencies:
pip install google-genai python-dotenv

Set Gemini API key:
export GEMINI_API_KEY="your_key_here"

Run the assistant:
python plaid_assistant.py

Example interaction:
You: How much did I spend on restaurants last month?
Assistant: You spent $97.50 across 6 transactions…

---

## 🔮 Future Enhancements

- Replace mock Plaid API with real Plaid SDK
- Add more tools:
  - get_accounts()
  - get_balances()
  - summarize_expenses()
  - detect_recurring_subscriptions()
- Add multi-tool planning
- Add budgets and alerts
- Add multi-account support
- Add a web UI (Streamlit or React)

---

## 📘 Summary

This project is a clean learning template for building agent-like LLM systems using function calling, multi-stage reasoning, and financial data analysis. It provides an extensible foundation for a full personal finance AI assistant.
