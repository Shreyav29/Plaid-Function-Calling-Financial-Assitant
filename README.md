# 📊 Plaid Function-Calling Financial Assistant

A two-stage Gemini LLM pipeline that routes user questions, fetches (mock) Plaid data, and returns natural-language financial insights.

---

## 🚀 Overview

This project is a prototype of an AI-powered personal finance assistant that uses Gemini function calling to decide when and how to fetch bank transaction data.

It demonstrates an agent-like workflow:

- **Router LLM** → Decide if Plaid data is needed  
- **Tool Call** → Fetch fake Plaid transactions  
- **Analyst LLM** → Analyze and answer  

> ⚠️ The system currently uses a **fake Plaid API (mock data)** for friction-free prototyping.  
> This can later be replaced with real Plaid SDK calls.

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

### 🔹 1. Smart Question Routing (LLM #1)

The first model decides:

- If the question requires Plaid data → triggers a function call  
- Otherwise → returns `CANNOT_ANSWER_WITH_PLAID`

**Examples:**

| User Question                                   | Router Output                  |
|-----------------------------------------------|--------------------------------|
| How much did I spend on groceries last month? | Calls `get_plaid_transactions` |
| What is the S&P 500?                           | CANNOT_ANSWER_WITH_PLAID       |
| Show my last 10 Starbucks charges              | Calls `get_plaid_transactions` |

---

### 🔹 2. Fake Plaid API (Prototype Only)

Instead of calling real Plaid endpoints, this project includes a mock Plaid function that returns:

- Dummy accounts  
- Dummy transactions  
- Realistic merchants, categories, dates, and amounts  

This enables end-to-end testing without OAuth or credentials.

---

### 🔹 3. Analyst LLM (LLM #2)

After transactions are retrieved, a second LLM run:

- Groups transactions by category  
- Computes totals  
- Detects patterns  
- Answers the question clearly  

**Example Output:**

> “You spent $97.50 at restaurants between Oct 1–5, mostly at Uber Eats and Starbucks.”

---

### 🔹 4. Global State Debugger

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
│
▼
┌──────────────────────────┐
│ LLM #1: Router Model │
│ (Decides: Plaid or Not) │
└──────────┬───────────────┘
│
Plaid Relevant?
│
▼
┌──────────────────────────┐
│ get_plaid_transactions │ ← Fake Plaid API
└──────────┬───────────────┘
│
▼
┌──────────────────────────┐
│ LLM #2: Analyst Model │
│ (Analyze + Summarize) │
└──────────────────────────┘
│
▼
Final Answer
