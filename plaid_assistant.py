"""
plaid_assistant.py

Prototype: Plaid Function-Calling Financial Assistant

This script demonstrates a two-stage LLM pipeline using Google's Gemini API
via the `google-genai` SDK:

1) Router LLM (LLM #1)
   - Decides whether a user question CAN be answered using Plaid-style
     bank transaction data.
   - If yes, it emits a function call to `get_plaid_transactions`.
   - If no, it returns a specific string: "CANNOT_ANSWER_WITH_PLAID".

2) Plaid API
   - In Sandbox/real envs, we call Plaid's /transactions/get endpoint.
   - For quick prototyping, you can still use a fake Plaid function.

3) Analyst LLM (LLM #2)
   - Receives:
       - the original user question
       - the JSON result from the Plaid function
   - Produces a natural-language answer (e.g., total spending, breakdowns).

4) Orchestrator
   - Glue code that:
       - Calls Router LLM
       - Executes Plaid (real or fake) if needed
       - Calls Analyst LLM
       - Returns a final answer to the user

5) Global State
   - A dictionary that stores:
       - last_question
       - router_raw_response
       - tool_call
       - plaid_args
       - plaid_result
       - analysis_prompt
       - analysis_raw_response
       - final_answer
   - Useful for debugging and understanding how the system behaves.
"""

import os
import json
from datetime import date, timedelta
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Plaid imports
import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
from plaid.model.accounts_balance_get_request import AccountsBalanceGetRequest  #Get balances for different accounts


# Load environment variables from .env
load_dotenv()


# =============================================================================
# 0. FUNCTION TO PARSE NATURAL LANGUAGE DATE RANGES FROM USER QUESTIONS
# =============================================================================


import re
import calendar
from datetime import date, timedelta
from typing import Optional, Tuple, Dict, Any

def parse_natural_date_range(
    text: str,
    today: Optional[date] = None,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Parse a natural-language-ish date range from the user's question,
    using ONLY Python's datetime and the real system date.

    Supported patterns (case-insensitive, anywhere in the string):

      - "last N days"       → [today - N days, today]
      - "last N weeks"      → [today - 7*N days, today]
      - "last N months"     → [today shifted N months back, same day or last valid day]
      - "last N years"      → [today shifted N years back, same month/day (or clipped)]

      - "last day"          → [yesterday, today]
      - "last week"         → previous calendar week (Mon–Sun)
      - "last month"        → previous calendar month (1st → last day)
      - "last year"         → previous calendar year (Jan 1 → Dec 31)

      - "this week"         → current calendar week (Mon → today)
      - "this month"        → current calendar month (1st → today)
      - "this year"         → current calendar year (Jan 1 → today)

      - "today"             → [today, today]
      - "yesterday"         → [yesterday, yesterday]

      - "between YYYY-MM-DD and YYYY-MM-DD"
      - "from YYYY-MM-DD to YYYY-MM-DD"

    Returns:
        (start_date_str, end_date_str, meta) where:
          - start_date_str, end_date_str are "YYYY-MM-DD"
          - meta is a dict with things like:
              {
                "kind": "last_n_days" | "last_month" | ...,
                "original_text": "<lowercased text>",
                "n": <int>  # when applicable
              }

        Returns None if no supported pattern is detected.

    NOTE: Weeks are ISO weeks with Monday as the first day.
    """

    if today is None:
        today = date.today()

    t = text.lower()

    meta: Dict[str, Any] = {"original_text": t}

    # ------------------------------------------------------------------
    # 1. Explicit "between" / "from ... to ..." with ISO dates
    # ------------------------------------------------------------------
    # between 2024-01-01 and 2024-02-01
    m_between = re.search(
        r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", t
    )
    if m_between:
        start_str, end_str = m_between.group(1), m_between.group(2)
        meta["kind"] = "between_explicit"
        return start_str, end_str, meta

    # from 2024-01-01 to 2024-02-01
    m_from_to = re.search(
        r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", t
    )
    if m_from_to:
        start_str, end_str = m_from_to.group(1), m_from_to.group(2)
        meta["kind"] = "from_to_explicit"
        return start_str, end_str, meta

    # ------------------------------------------------------------------
    # 2. Today / Yesterday
    # ------------------------------------------------------------------
    if "today" in t:
        meta["kind"] = "today"
        s = e = today
        return s.isoformat(), e.isoformat(), meta

    if "yesterday" in t:
        meta["kind"] = "yesterday"
        y = today - timedelta(days=1)
        return y.isoformat(), y.isoformat(), meta

    # ------------------------------------------------------------------
    # 3. "last N days/weeks/months/years"
    # ------------------------------------------------------------------
    m_last_n = re.search(
        r"last\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)", t
    )
    if m_last_n:
        n = int(m_last_n.group(1))
        unit = m_last_n.group(2)  # as written in text
        meta["n"] = n

        if "day" in unit:
            meta["kind"] = "last_n_days"
            end = today
            start = today - timedelta(days=n)
            return start.isoformat(), end.isoformat(), meta

        if "week" in unit:
            meta["kind"] = "last_n_weeks"
            end = today
            start = today - timedelta(days=7 * n)
            return start.isoformat(), end.isoformat(), meta

        if "month" in unit:
            meta["kind"] = "last_n_months"
            end = today
            # shift back n months carefully
            year = today.year
            month = today.month - n
            while month <= 0:
                month += 12
                year -= 1
            # clamp day to last valid day in target month
            last_day = calendar.monthrange(year, month)[1]
            day = min(today.day, last_day)
            start = date(year, month, day)
            return start.isoformat(), end.isoformat(), meta

        if "year" in unit:
            meta["kind"] = "last_n_years"
            end = today
            year = today.year - n
            month = today.month
            day = today.day
            # guard against Feb 29 issues: clamp to end of month if invalid
            last_day = calendar.monthrange(year, month)[1]
            day = min(day, last_day)
            start = date(year, month, day)
            return start.isoformat(), end.isoformat(), meta

    # ------------------------------------------------------------------
    # 4. "last day/week/month/year" (no number)
    # ------------------------------------------------------------------
    if "last day" in t:
        meta["kind"] = "last_day"
        end = today
        start = today - timedelta(days=1)
        return start.isoformat(), end.isoformat(), meta

    if "last week" in t:
        meta["kind"] = "last_week"
        # ISO: Monday is 0, Sunday is 6
        # we define "this week" as Monday..Sunday containing 'today'
        # so "last week" is the previous Monday..Sunday
        weekday = today.weekday()  # 0..6
        this_monday = today - timedelta(days=weekday)
        last_week_end = this_monday - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        return last_week_start.isoformat(), last_week_end.isoformat(), meta

    if "last month" in t:
        meta["kind"] = "last_month"
        year = today.year
        month = today.month
        # previous month
        if month == 1:
            p_month = 12
            p_year = year - 1
        else:
            p_month = month - 1
            p_year = year
        start = date(p_year, p_month, 1)
        last_day = calendar.monthrange(p_year, p_month)[1]
        end = date(p_year, p_month, last_day)
        return start.isoformat(), end.isoformat(), meta

    if "last year" in t:
        meta["kind"] = "last_year"
        prev_year = today.year - 1
        start = date(prev_year, 1, 1)
        end = date(prev_year, 12, 31)
        return start.isoformat(), end.isoformat(), meta

    # ------------------------------------------------------------------
    # 5. "this week/month/year"
    # ------------------------------------------------------------------
    if "this week" in t:
        meta["kind"] = "this_week"
        weekday = today.weekday()
        this_monday = today - timedelta(days=weekday)
        # up to "today" (not full future week)
        start = this_monday
        end = today
        return start.isoformat(), end.isoformat(), meta

    if "this month" in t:
        meta["kind"] = "this_month"
        start = date(today.year, today.month, 1)
        end = today
        return start.isoformat(), end.isoformat(), meta

    if "this year" in t:
        meta["kind"] = "this_year"
        start = date(today.year, 1, 1)
        end = today
        return start.isoformat(), end.isoformat(), meta

    # ------------------------------------------------------------------
    # If nothing matched, return None and let caller pick defaults
    # ------------------------------------------------------------------
    return None







# =============================================================================
# 0. GLOBAL STATE
# =============================================================================

GLOBAL_STATE: Dict[str, Any] = {
    "last_question": None,
    "router_raw_response": None,
    "tool_call": None,
    "plaid_args": None,
    "plaid_result": None,
    "analysis_prompt": None,
    "analysis_raw_response": None,
    "final_answer": None,
}

# Optional toggle: if set to "true", we will use the fake Plaid function instead
# of calling the real Plaid API. Helpful if you want to test without network.
USE_FAKE_PLAID = os.getenv("USE_FAKE_PLAID", "false").lower() == "true"

# =============================================================================
# 1. GEMINI CLIENT SETUP
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# =============================================================================
# 2. PLAID CLIENT SETUP + REAL/Fake TRANSACTIONS CALLS
# =============================================================================

"""
We support two modes:

1) REAL Plaid (Sandbox or Dev/Prod):
   - Uses PLAID_CLIENT_ID, PLAID_SECRET, PLAID_ACCESS_TOKEN, PLAID_ENV.
   - Calls /transactions/get to fetch real-looking (or real) transactions.

2) FAKE Plaid:
   - Returns a static, hard-coded JSON for quick local testing.
   - Controlled via USE_FAKE_PLAID env var.
"""

PLAID_CLIENT_ID = os.environ.get("PLAID_CLIENT_ID")
PLAID_SECRET = os.environ.get("PLAID_SECRET")
PLAID_ACCESS_TOKEN = os.environ.get("PLAID_ACCESS_TOKEN")
PLAID_ENV = (os.environ.get("PLAID_ENV") or "sandbox").lower()

if PLAID_ENV == "production":
    plaid_host = plaid.Environment.Production
elif PLAID_ENV == "development":
    plaid_host = plaid.Environment.Development
else:
    plaid_host = plaid.Environment.Sandbox  # default

# Configure Plaid client (even if keys are missing; calls will just fail later)
plaid_config = plaid.Configuration(
    host=plaid_host,
    api_key={
        "clientId": PLAID_CLIENT_ID or "",
        "secret": PLAID_SECRET or "",
    },
)
plaid_api_client = plaid.ApiClient(plaid_config)
plaid_client = plaid_api.PlaidApi(plaid_api_client)


def fake_plaid_transactions(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Stand-in for a real Plaid /transactions/get call.

    Returns a small, hard-coded JSON structure that *looks* like
    Plaid-style output, so that the LLM has something realistic to work with.
    """
    return {
        "start_date": start_date,
        "end_date": end_date,
        "accounts": [
            {
                "account_id": "acc_123",
                "name": "Checking",
                "type": "depository",
                "subtype": "checking",
                "current_balance": 2435.17,
                "iso_currency_code": "USD",
            }
        ],
        "transactions": [
            {
                "date": "2025-10-01",
                "name": "UBER EATS",
                "amount": 24.50,
                "account_id": "acc_123",
                "category": ["Food and Drink", "Restaurants"],
            },
            {
                "date": "2025-10-02",
                "name": "WHOLE FOODS",
                "amount": 65.20,
                "account_id": "acc_123",
                "category": ["Groceries"],
            },
            {
                "date": "2025-10-05",
                "name": "STARBUCKS",
                "amount": 7.80,
                "account_id": "acc_123",
                "category": ["Food and Drink", "Coffee Shop"],
            },
        ],
    }


def real_plaid_transactions(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Call Plaid's /transactions/get for the configured PLAID_ACCESS_TOKEN.

    Args:
        start_date: Inclusive start date in YYYY-MM-DD.
        end_date:   Inclusive end date in YYYY-MM-DD.

    Returns:
        Dict with:
            - "start_date"
            - "end_date"
            - "accounts": currently empty (can be filled later)
            - "transactions": list of transaction dicts
            - "plaid_error": any Plaid error (if occurred)
    """
    if not PLAID_ACCESS_TOKEN:
        return {
            "start_date": start_date,
            "end_date": end_date,
            "accounts": [],
            "transactions": [],
            "plaid_error": {
                "message": "PLAID_ACCESS_TOKEN is not set. Using empty transactions."
            },
        }

    # Plaid SDK expects datetime.date objects
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    request = TransactionsGetRequest(
        access_token=PLAID_ACCESS_TOKEN,
        start_date=start,
        end_date=end,
        options=TransactionsGetRequestOptions(
            count=100,
            offset=0,
        ),
    )

    try:
        response = plaid_client.transactions_get(request)
        response_dict = response.to_dict()
        transactions = response_dict.get("transactions", [])
        return {
            "start_date": start_date,
            "end_date": end_date,
            "accounts": [],
            "transactions": transactions,
            "raw_plaid_response": {
                "total_transactions": response_dict.get("total_transactions"),
            },
            "plaid_error": None,
        }
    except plaid.ApiException as e:
        try:
            error_json = json.loads(e.body)
        except Exception:
            error_json = {"raw_body": e.body}
        return {
            "start_date": start_date,
            "end_date": end_date,
            "accounts": [],
            "transactions": [],
            "plaid_error": error_json,
        }


def get_transactions_from_plaid(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Wrapper to decide whether to use real Plaid or fake Plaid
    based on the USE_FAKE_PLAID flag.
    """
    if USE_FAKE_PLAID:
        return fake_plaid_transactions(start_date, end_date)
    else:
        return real_plaid_transactions(start_date, end_date)


def fake_plaid_accounts() -> Dict[str, Any]:
    """
    Stand-in for a real Plaid /accounts/balance/get call.
    Returns a small, hard-coded set of accounts.
    """
    return {
        "accounts": [
            {
                "account_id": "acc_123",
                "name": "Checking",
                "official_name": "Everyday Checking",
                "type": "depository",
                "subtype": "checking",
                "mask": "1111",
                "balances": {
                    "available": 2350.17,
                    "current": 2435.17,
                    "iso_currency_code": "USD",
                },
            },
            {
                "account_id": "acc_456",
                "name": "Savings",
                "official_name": "High Yield Savings",
                "type": "depository",
                "subtype": "savings",
                "mask": "2222",
                "balances": {
                    "available": 10250.00,
                    "current": 10250.00,
                    "iso_currency_code": "USD",
                },
            },
        ],
        "plaid_error": None,
    }


def real_plaid_accounts() -> Dict[str, Any]:
    """
    Call Plaid's /accounts/balance/get for the configured PLAID_ACCESS_TOKEN.

    Returns:
        Dict with:
          - "accounts": list of account dicts with balances
          - "plaid_error": any Plaid error if occurred
    """
    if not PLAID_ACCESS_TOKEN:
        return {
            "accounts": [],
            "plaid_error": {
                "message": "PLAID_ACCESS_TOKEN is not set. Using empty accounts list."
            },
        }

    request = AccountsBalanceGetRequest(access_token=PLAID_ACCESS_TOKEN)

    try:
        response = plaid_client.accounts_balance_get(request)
        response_dict = response.to_dict()
        accounts = response_dict.get("accounts", [])
        return {
            "accounts": accounts,
            "plaid_error": None,
            "raw_plaid_response": {
                "item": response_dict.get("item"),
            },
        }
    except plaid.ApiException as e:
        try:
            error_json = json.loads(e.body)
        except Exception:
            error_json = {"raw_body": e.body}
        return {
            "accounts": [],
            "plaid_error": error_json,
        }


def get_accounts_from_plaid() -> Dict[str, Any]:
    """
    Wrapper to decide whether to use real Plaid or fake Plaid for accounts.
    """
    if USE_FAKE_PLAID:
        return fake_plaid_accounts()
    else:
        return real_plaid_accounts()





# =============================================================================
# 3. TOOL (FUNCTION) DECLARATION FOR GEMINI
# =============================================================================

get_plaid_transactions_decl = types.FunctionDeclaration(
    name="get_plaid_transactions",
    description=(
        "Fetch the user's bank transactions via Plaid for a given date range. "
        "Use this ONLY if the question can be answered purely from account "
        "and transaction data (spending, merchants, categories, balances)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "start_date": {
                "type": "string",
                "description": "Start date (inclusive), in YYYY-MM-DD format.",
            },
            "end_date": {
                "type": "string",
                "description": "End date (inclusive), in YYYY-MM-DD format.",
            },
        },
        "required": ["start_date", "end_date"],
    },
)

get_plaid_accounts_decl = types.FunctionDeclaration(
    name="get_plaid_accounts",
    description=(
        "Fetch the user's account list and balances via Plaid. "
        "Use this for questions about balances, accounts, or cash on hand."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)

plaid_tool = types.Tool(function_declarations=[get_plaid_transactions_decl, get_plaid_accounts_decl])

def merge_accounts_into_transactions(
    plaid_result: Dict[str, Any],
    accounts_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach account details to each transaction based on account_id.

    - plaid_result: dict with at least "transactions": [...]
    - accounts_result: dict with "accounts": [...]

    For each transaction:
      - add transaction["account_type"]
      - add transaction["account_subtype"]
      - add transaction["account_name"]
      - add transaction["account_mask"]

    Also copy accounts into plaid_result["accounts"] so the analyst
    has both views.

    Returns the updated plaid_result.
    """
    txns = plaid_result.get("transactions", []) or []
    accounts = accounts_result.get("accounts", []) or []

    # Build lookup from account_id → account dict
    by_id = {}
    for acc in accounts:
        acc_id = acc.get("account_id")
        if not acc_id:
            continue
        by_id[acc_id] = acc

    # Attach account metadata to each transaction
    for t in txns:
        acc_id = t.get("account_id")
        acc = by_id.get(acc_id)
        if not acc:
            continue
        t["account_type"] = acc.get("type")
        t["account_subtype"] = acc.get("subtype")
        t["account_name"] = acc.get("name") or acc.get("official_name")
        t["account_mask"] = acc.get("mask")

    plaid_result["transactions"] = txns
    plaid_result["accounts"] = accounts
    return plaid_result


# =============================================================================
# 4. ROUTER LLM (LLM #1) – DECIDE IF PLAID IS APPLICABLE
# =============================================================================

ROUTER_SYSTEM_INSTRUCTION = """
You are a strict router model.

Your ONLY job is to decide whether a user question CAN be fully answered using
Plaid data, and which Plaid tool to call.

You have exactly three allowed behaviors:

────────────────────────────────────────────────────────
1) If the question is about SPENDING / TRANSACTIONS:
   → You MUST call: get_plaid_transactions(start_date, end_date)
   → NEVER answer in natural language.
   → NEVER provide explanations.
   → ALWAYS produce a tool call.

   Examples:
   - "How much did I spend...?"
   - "What are my last N transactions?"
   - "Show spending by category"
   - "What did I spend at Uber / Starbucks?"
   - "Spending last 7/30/60 days"
   - "Spending last month"
   - "How much did I spend between 2024-01-01 and 2024-02-01?"

   If the user does not specify dates:
   → Default to the last 30 days in the tool arguments.
   (The orchestrator may override dates with Python date parsing.)

────────────────────────────────────────────────────────
2) If the question is about ACCOUNTS / BALANCES / CASH:
   → You MUST call: get_plaid_accounts()
   → NEVER answer in natural language.
   → NEVER provide explanations.
   → ALWAYS produce a tool call.

   Examples:
   - "What is my checking balance?"
   - "How much money do I have in savings?"
   - "List all my accounts and balances."
   - "What is my total cash across all accounts?"

────────────────────────────────────────────────────────
3) If the question CANNOT be answered using Plaid data at all:
   → Respond with EXACTLY:
       CANNOT_ANSWER_WITH_PLAID
   → No punctuation.
   → No natural language.
   → No explanation.
────────────────────────────────────────────────────────

ADDITIONAL RULE FOR COMBINED QUESTIONS:
- If the user’s question asks about BOTH:
  - balances/accounts AND
  - spending/transactions
  THEN:
    → Treat it as a SPENDING question.
    → You MUST call get_plaid_transactions(start_date, end_date)
    → NEVER answer in natural language.
    → NEVER call get_plaid_accounts directly.
    (The orchestrator will fetch account balances separately.)

Examples that MUST call get_plaid_transactions:
- "what is my checking balance and how much did I spend on that account in the last month"
- "what is my savings balance and how much did I deposit there this year"
- "how much did I spend from my checking account in the last month"
- "how much did I spend on restaurants from my credit card"

Examples that MUST call get_plaid_accounts:
- "what is my checking balance"
- "list all my accounts and balances"
- "how much cash do I have in my depository accounts"
────────────────────────────────────────────────────────
ABSOLUTE RULES:
- DO NOT answer questions in natural language.
- DO NOT paraphrase the user's question.
- DO NOT return text unless it is exactly: CANNOT_ANSWER_WITH_PLAID.
- Any spending-related question MUST use get_plaid_transactions.
- Any balance/account-related question MUST use get_plaid_accounts.
"""



def call_router_model(user_question: str) -> types.GenerateContentResponse:
    """
    Call Gemini (LLM #1) with the router system instruction and the Plaid tool.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_question,  # plain string; SDK wraps it
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_SYSTEM_INSTRUCTION,
            tools=[plaid_tool],
        ),
    )
    return response


# =============================================================================
# 5. MERCHANT OR CATEGORY MAPPING HELPER
# =============================================================================

def _map_merchant_category(name: str, categories: Optional[list]) -> Dict[str, str]:
    """
    Heuristic merchant/category consolidation.

    Returns a dict with:
      - normalized_name
      - consolidated_category
    """
    if not name:
        name = ""
    lname = name.lower()
    categories = categories or []

    normalized_name = name.strip()

    # Simple substring rules for common merchants
    if "starbucks" in lname:
        return {"normalized_name": "Starbucks", "consolidated_category": "Coffee"}
    if "mcdonald" in lname:
        return {"normalized_name": "McDonald's", "consolidated_category": "Fast Food"}
    if "kfc" in lname:
        return {"normalized_name": "KFC", "consolidated_category": "Fast Food"}
    if "whole foods" in lname:
        return {"normalized_name": "Whole Foods", "consolidated_category": "Groceries"}
    if "uber" in lname or "lyft" in lname:
        return {"normalized_name": "Uber/Lyft", "consolidated_category": "Transportation"}
    if "gusto pay" in lname or "payroll" in lname:
        return {"normalized_name": "Payroll", "consolidated_category": "Income"}
    if "cd deposit" in lname or "deposit" in lname:
        return {"normalized_name": "Deposit", "consolidated_category": "Savings/Deposit"}
    if "automatic payment" in lname:
        return {"normalized_name": "Automatic Payment", "consolidated_category": "Bill/Loan Payment"}
    if "touchstone" in lname:
        return {"normalized_name": "Touchstone Climbing", "consolidated_category": "Fitness"}
    if "united airlines" in lname or "delta" in lname or "american airlines" in lname:
        return {"normalized_name": "Airline", "consolidated_category": "Travel"}
    if "sparkfun" in lname:
        return {"normalized_name": "SparkFun", "consolidated_category": "Electronics/Hobby"}
    if "madison bicycle" in lname:
        return {"normalized_name": "Bike Shop", "consolidated_category": "Sporting Goods"}

    # Fallback using Plaid category if present
    if categories:
        # e.g., ["Food and Drink", "Restaurants"]
        top = categories[0]
        return {"normalized_name": normalized_name, "consolidated_category": top}

    # Final fallback
    return {"normalized_name": normalized_name, "consolidated_category": "Other"}


from datetime import datetime

def _tag_transaction(t: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a single Plaid transaction dict with:
      - is_spend: bool
      - type_tag: 'spend' | 'income' | 'transfer_or_savings' | 'refund_or_inflow' | 'other'
      - normalized_name: str
      - consolidated_category: str

    This is heuristic and designed to be understandable by the LLM.
    """
    amount = t.get("amount", 0.0) or 0.0
    name = t.get("name") or t.get("merchant_name") or ""
    categories = t.get("category") or []
    ttype = t.get("transaction_type") or ""

    lname = name.lower()
    is_spend = False
    type_tag = "other"

    # Basic merchant/category normalization
    mapping = _map_merchant_category(name, categories)
    normalized_name = mapping["normalized_name"]
    consolidated_category = mapping["consolidated_category"]

    # Negative amounts: refunds or inbound flows
    if amount < 0:
        type_tag = "refund_or_inflow"
        is_spend = False
    else:
        # Positive amounts: could be spend, income, transfer, or deposit
        income_keywords = ["gusto pay", "payroll", "salary", "income"]
        transfer_keywords = ["transfer", "cd deposit", "deposit", "payment", "p2p"]
        refund_keywords = ["refund", "reversal"]

        # Check name + categories + transaction_type
        low_all = " ".join([lname] + [c.lower() for c in categories] + [ttype.lower()])

        if any(k in low_all for k in income_keywords):
            type_tag = "income"
            is_spend = False
        elif any(k in low_all for k in refund_keywords):
            type_tag = "refund_or_inflow"
            is_spend = False
        elif any(k in low_all for k in transfer_keywords):
            type_tag = "transfer_or_savings"
            is_spend = False
        else:
            type_tag = "spend"
            is_spend = True

    t["is_spend"] = is_spend
    t["type_tag"] = type_tag
    t["normalized_name"] = normalized_name
    t["consolidated_category"] = consolidated_category
    return t


def _detect_recurring_subscriptions(transactions: list) -> list:
    """
    Very simple recurring subscription detection:

    - Group by normalized_name
    - If there are >= 3 transactions
    - With similar amounts
    - Occurring at roughly regular intervals (weekly or monthly)
    Then mark as a recurring subscription candidate.

    Returns:
        A list of dicts:
          {
            "merchant": str,
            "period": "weekly" | "monthly",
            "average_amount": float,
            "count": int,
            "first_date": "YYYY-MM-DD",
            "last_date": "YYYY-MM-DD",
          }
    """
    by_merchant: Dict[str, list] = {}
    for t in transactions:
        name = t.get("normalized_name") or t.get("name") or ""
        if not name:
            continue
        by_merchant.setdefault(name, []).append(t)

    candidates = []

    for merchant, txns in by_merchant.items():
        if len(txns) < 3:
            continue

        # Parse dates and sort ascending
        dated = []
        for t in txns:
            d_str = t.get("date")
            if not d_str:
                continue
            try:
                d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
                dated.append((d_obj, t))
            except Exception:
                continue

        if len(dated) < 3:
            continue

        dated.sort(key=lambda x: x[0])
        dates = [d for d, _ in dated]
        amounts = [float(t.get("amount", 0.0) or 0.0) for _, t in dated]

        # Require non-zero spend-like amounts
        positive_amounts = [a for a in amounts if a > 0]
        if len(positive_amounts) < 3:
            continue

        avg_amount = sum(positive_amounts) / len(positive_amounts)
        max_amount = max(positive_amounts)
        min_amount = min(positive_amounts)

        # Amounts must be similar (within ~10% or $1)
        if max_amount - min_amount > max(1.0, 0.1 * avg_amount):
            continue

        # Look at gaps between dates
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        if not gaps:
            continue
        avg_gap = sum(gaps) / len(gaps)

        period = None
        if 27 <= avg_gap <= 33:
            period = "monthly"
        elif 6 <= avg_gap <= 8:
            period = "weekly"
        else:
            continue

        candidates.append(
            {
                "merchant": merchant,
                "period": period,
                "average_amount": round(avg_amount, 2),
                "count": len(dates),
                "first_date": dates[0].isoformat(),
                "last_date": dates[-1].isoformat(),
            }
        )

    return candidates


def preprocess_plaid_result(plaid_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches plaid_result in-place with:
      - tagged transactions (is_spend, type_tag, normalized_name, consolidated_category)
      - recurring_subscriptions: list of inferred recurring merchants

    Returns the modified plaid_result.
    """
    txns = plaid_result.get("transactions", [])
    enriched = []
    for t in txns:
        enriched.append(_tag_transaction(t))

    plaid_result["transactions"] = enriched
    plaid_result["recurring_subscriptions"] = _detect_recurring_subscriptions(enriched)
    return plaid_result




# =============================================================================
# 5. ANALYST LLM (LLM #2) – INTERPRET PLAID DATA & ANSWER QUESTION
# =============================================================================

ANALYST_SYSTEM_INSTRUCTION = """
You are a concise, opinionated financial analysis assistant.

You receive:
- The user's original question.
- A JSON object called plaid_result that may contain:
  - start_date, end_date (for transaction queries)
  - transactions: list of transaction dicts
    Each transaction may include:
      - date, name, amount, category, transaction_type, account_id
      - is_spend: bool (Python-tagged)
      - type_tag: 'spend' | 'income' | 'transfer_or_savings' | 'refund_or_inflow' | 'other'
      - normalized_name: normalized merchant name
      - consolidated_category: high-level category (e.g., 'Coffee', 'Groceries')
      - account_type: e.g., 'depository', 'credit', 'investment'
      - account_subtype: e.g., 'checking', 'savings', 'credit card'
      - account_name, account_mask
  - recurring_subscriptions: list of recurring subscription candidates
      {merchant, period, average_amount, count, first_date, last_date}
  - accounts: list of accounts (for balance queries)
      Each account may include:
        - name, official_name, type, subtype, mask
        - balances: {available, current, iso_currency_code}
  - plaid_error: any error from Plaid

Your job:

1) Respect the data:
   - ONLY use the data in plaid_result.
   - If plaid_error is present and non-null, acknowledge it and explain that the data may be incomplete.
   - For transaction-based questions, use only plaid_result.transactions.
   - For balance/account questions, use only plaid_result.accounts.

2) Account filters:
   - If the user asks about "checking", "savings", or "credit card" specifically:
     - Filter transactions to those whose account_subtype matches:
       - checking → account_subtype == 'checking'
       - savings → account_subtype == 'savings'
       - credit card → account_subtype contains 'credit'
   - If the user says "from my checking accounts", only use those filtered transactions when computing totals.

3) Spending definition:
   - "Spending" = sum of amounts where transaction.is_spend is true AND type_tag == 'spend'.
   - EXCLUDE:
     - type_tag == 'income'
     - type_tag == 'transfer_or_savings'
     - type_tag == 'refund_or_inflow'
     - Any transaction where is_spend is false
   - Do NOT invent transactions or modify amounts.

4) Different question types:

   A) Spending total / category / merchant questions:
      - If the question asks "how much did I spend...":
        - Use the above spend definition.
        - Start with:
          - "Date range: START → END" (if present)
          - "Total spending (excluding income, transfers, refunds): $X.XX"
        - Then include 3–5 concise bullets, such as:
          - Top categories by spend using consolidated_category.
          - Top merchants by spend using normalized_name.
      - If the question is about a specific merchant (e.g., Starbucks):
        - Filter to transactions whose normalized_name or name mentions that merchant.
        - Summarize total spend and list 2–3 example transactions.

      - If the question is about a category (e.g., restaurants, coffee, groceries):
        - Use consolidated_category and/or category field to filter.
        - Summarize total spend and key merchants.

      - Keep answers under ~8–10 lines unless the user explicitly asks for more detail.

   B) "Last N transactions" / recent transactions:
      - Sort transactions by date descending.
      - Show up to N rows if N is specified; otherwise 5–10.
      - Each row:
        - "YYYY-MM-DD – normalized_name (or name) – $AMOUNT – consolidated_category"
      - Be compact and avoid long explanations.

   C) Recurring subscriptions:
      - If the user asks about recurring payments, subscriptions, or "what subscriptions do I have":
        - Use plaid_result.recurring_subscriptions.
        - For each candidate, report:
          - merchant, period (weekly/monthly), average_amount, count, first_date, last_date.
        - If there are none, say clearly that no recurring patterns were detected.

   D) Accounts / balances:
        - If the question is about balances, accounts, or "how much money do I have":
            - Use plaid_result.accounts.
            - Summarize each account:
            - "Account: NAME (SUBTYPE • ****MASK) – Current: $X.XX, Available: $Y.YY"
        - If the question asks BOTH for a balance and for spending from that account:
            - First, identify the relevant account(s) (e.g., checking).
            - Report that account's balance.
            - Then, filter transactions to that same account_subtype/account_id
            and compute spending using the spend definition.
            - Answer both parts in one concise response.

5) Style:
   - Be concise and structured.
   - Use short paragraphs and bullet lists.
   - Do NOT show raw JSON.
   - Do NOT reveal internal tags like is_spend/type_tag explicitly unless needed; use them internally to answer correctly.
"""



def call_analysis_model(user_question: str, plaid_result: Dict[str, Any]) -> str:
    """
    Call Gemini (LLM #2) to analyze Plaid-style JSON and return a human answer.

    We:
      - Serialize plaid_result with default=str so dates are safe.
      - Provide a structured prompt so the model knows:
          * user question
          * JSON data
          * that it must apply our "spend" definition
          * that it must be concise and formatted.
    """
    plaid_json_str = json.dumps(plaid_result, indent=2, default=str)

    start_date = plaid_result.get("start_date")
    end_date = plaid_result.get("end_date")

    analysis_prompt = (
        "You are given a user question and a Plaid-style JSON result.\n\n"
        "User question:\n"
        f"{user_question}\n\n"
        "Effective date range from Plaid JSON:\n"
        f"start_date: {start_date}\n"
        f"end_date:   {end_date}\n\n"
        "Plaid-style result JSON:\n"
        f"{plaid_json_str}\n\n"
        "Instructions:\n"
        "- Apply the spend definition from the system instruction.\n"
        "- Use ONLY the transactions and fields that appear in this JSON.\n"
        "- Do NOT fabricate extra data.\n"
        "- If relevant, compute totals and short category/merchant breakdowns.\n"
        "- Keep the answer concise, following the formats described in the system instruction.\n"
    )

    GLOBAL_STATE["analysis_prompt"] = analysis_prompt

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=analysis_prompt,
        config=types.GenerateContentConfig(
            system_instruction=ANALYST_SYSTEM_INSTRUCTION,
        ),
    )

    GLOBAL_STATE["analysis_raw_response"] = response
    final_text = response.text or ""
    GLOBAL_STATE["final_answer"] = final_text
    return final_text



# =============================================================================
# 6. ORCHESTRATOR – FULL END-TO-END PIPELINE
# =============================================================================

def extract_first_function_call(
    response: types.GenerateContentResponse,
) -> Optional[types.FunctionCall]:
    """
    Helper to get the first function call from a GenerateContentResponse.

    Newer google-genai responses store function calls inside:
      response.candidates[i].content.parts[j].function_call

    This helper scans all candidates/parts and returns the first function_call found.
    """
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None:
                return fc
    return None


def handle_user_question(user_question: str) -> str:
    """
    Full pipeline:
      1) Router LLM decides if Plaid is relevant and may call get_plaid_transactions.
      2) If router says "CANNOT_ANSWER_WITH_PLAID" → return a friendly message.
      3) If router calls the function → execute real/fake Plaid, then call Analyst LLM.
    """
    GLOBAL_STATE["last_question"] = user_question

    # Step 1: Router
    router_response = call_router_model(user_question)
    GLOBAL_STATE["router_raw_response"] = router_response

    
    fc = extract_first_function_call(router_response)
    print("[DEBUG] Extracted function_call from candidates/parts:", fc)

    router_text = ""
    try:
        router_text = (router_response.text or "").strip()
    except Exception:
        router_text = ""

    if router_text == "CANNOT_ANSWER_WITH_PLAID":
        return "This question cannot be answered using your Plaid transaction data."

    function_call = extract_first_function_call(router_response)
    if function_call is None:
        return (
            "I couldn't determine how to answer this question using Plaid data. "
            "Try rephrasing or specifying a time period for your transactions."
        )

    GLOBAL_STATE["tool_call"] = function_call

    # Step 2: Extract args and call Plaid (real or fake)
    args = dict(function_call.args or {})
    GLOBAL_STATE["plaid_args"] = args

    tool_name = function_call.name

    if tool_name == "get_plaid_transactions":
        # ---- Transactions path: use date parsing + transactions call ----
        parsed = parse_natural_date_range(user_question)
        print("---- DEBUG: parsed date range:", parsed)
        today = date.today()

        # Default window: last 30 days
        default_end = today
        default_start = today - timedelta(days=30)

        # If the question is about subscriptions / recurring payments,
        # and the user did NOT specify a range, use last 12 months by default.
        subscription_keywords = ["subscription", "subscriptions", "recurring", "monthly payment", "monthly payments"]
        is_subscription_query = any(k in user_question.lower() for k in subscription_keywords)

        if parsed is not None:
            start_date, end_date, meta = parsed
            GLOBAL_STATE["parsed_date_meta"] = meta
        else:
            if is_subscription_query:
                # Widen to last 12 months for better recurrence detection
                start_date = (today.replace(year=today.year - 1)).isoformat()
                end_date = today.isoformat()
            else:
                # Fallback: router args or last 30 days
                start_date = args.get("start_date") or default_start.isoformat()
                end_date = args.get("end_date") or default_end.isoformat()

        GLOBAL_STATE["effective_date_range"] = {
            "start_date": start_date,
            "end_date": end_date,
        }


        # 1) Get transactions for the date range
        plaid_result = get_transactions_from_plaid(start_date, end_date)

        # 2) Get accounts (for account type/subtype/name)
        accounts_result = get_accounts_from_plaid()

        # 3) Merge account info into each transaction and copy accounts into plaid_result
        plaid_result = merge_accounts_into_transactions(plaid_result, accounts_result)

        # 4) Preprocess transactions (spend tags, subscriptions, etc.)
        plaid_result = preprocess_plaid_result(plaid_result)

        GLOBAL_STATE["plaid_result"] = plaid_result

    elif tool_name == "get_plaid_accounts":
        # ---- Accounts/balances path ----
        plaid_result = get_accounts_from_plaid()
        GLOBAL_STATE["effective_date_range"] = None  # not applicable
        GLOBAL_STATE["plaid_result"] = plaid_result

    else:
        # Unknown tool name (should not happen)
        return (
            f"Router called an unknown tool '{tool_name}'. "
            "Unable to answer using Plaid data."
        )


    # Step 3: Analyst LLM
    final_answer = call_analysis_model(user_question, plaid_result)

    # --- DEBUG: inspect router response ---
    print("\n[DEBUG] Router response.text:", repr(getattr(router_response, "text", None)))
    print("[DEBUG] Router response.function_calls:", getattr(router_response, "function_calls", None))

    return final_answer

# =============================================================================
# 7. SIMPLE CLI LOOP FOR MANUAL TESTING
# =============================================================================

if __name__ == "__main__":
    print("Plaid Function-Calling Financial Assistant (Sandbox/Real Prototype)")
    print("Type 'exit' or 'quit' to stop.\n")

    if not GEMINI_API_KEY:
        print(
            "WARNING: GEMINI_API_KEY is not set.\n"
            "Set it in .env or your environment before running.\n"
        )

    if not USE_FAKE_PLAID and not PLAID_ACCESS_TOKEN:
        print(
            "WARNING: USE_FAKE_PLAID is false and PLAID_ACCESS_TOKEN is missing.\n"
            "Either set USE_FAKE_PLAID=true in .env, or provide a valid PLAID_ACCESS_TOKEN.\n"
        )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        answer = handle_user_question(user_input)
        print(f"\nUser Input: {user_input}\n")
        print(f"\nAssistant: {answer}\n")

        debug_view = {
            "last_question": GLOBAL_STATE.get("last_question"),
            "plaid_args": GLOBAL_STATE.get("plaid_args"),
            "effective_date_range": GLOBAL_STATE.get("effective_date_range"),
            "transaction_count": len(
                (GLOBAL_STATE.get("plaid_result") or {}).get("transactions", [])
            ),
            "plaid_error": (GLOBAL_STATE.get("plaid_result") or {}).get("plaid_error"),
            "use_fake_plaid": USE_FAKE_PLAID,
        }
        print("---- DEBUG: GLOBAL_STATE (trimmed) ----")
        print(json.dumps(debug_view, indent=2, default=str))
        print("---------------------------------------\n")
