import pandas as pd
import io
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import threading
import time
import logging
from pathlib import Path
import os
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faker import Faker
import random 
# Removed: import numpy (no longer needed for charting)

# --- Configuration: All Tax Rules, Slabs, and Limits ---
TAX_CONFIG: Dict[str, Any] = {
    'cess_rate': 0.04,
    'deductions': {
        '80c_max': 150000,
        '80d_self_family': 25000,
        '80d_senior_parents': 50000,
        '80ccd1b_nps': 50000,
        '24b_home_loan_interest': 200000
    },
    'old_regime': {
        'standard_deduction': 50000, 
        'slabs': [
            {'limit': 250000, 'rate': 0.0},
            {'limit': 500000, 'rate': 0.05},
            {'limit': 1000000, 'rate': 0.20},
            {'limit': float('inf'), 'rate': 0.30}
        ],
        'rebate': {
            'taxable_income_limit': 500000,
            'max_rebate': 12500
        }
    },
    'new_regime': {
        'standard_deduction': 75000,
        'slabs': [
            {'limit': 300000, 'rate': 0.0},
            {'limit': 600000, 'rate': 0.05},
            {'limit': 900000, 'rate': 0.10},
            {'limit': 1200000, 'rate': 0.15},
            {'limit': 1500000, 'rate': 0.20},
            {'limit': float('inf'), 'rate': 0.30}
        ],
        'rebate': {
            'taxable_income_limit': 700000,
            'max_rebate': 25000
        }
    }
}

CIBIL_FACTORS = {
    'payment_history': 0.35,
    'credit_utilization': 0.30,
    'credit_mix': 0.10,
    'debt_to_income': 0.15,
    'credit_inquiries': 0.10
}

SCORE_RANGES = {
    'excellent': {'min': 750, 'status': 'Excellent'},
    'good': {'min': 700, 'status': 'Good'},
    'fair': {'min': 650, 'status': 'Fair'},
    'poor': {'min': 300, 'status': 'Poor'}
}

# Use a fixed seed for reproducible fake data (optional, but good for testing)
Faker.seed(42)
fake = Faker()

# --- Placeholder for AI Parsing (To maintain code structure) ---
def extract_text_from_pdf(content: bytes): return ""
async def parse_with_gemini(text: str, filename: str): return []

# Simple logging setup (Detailed logging maintained)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Unified Financial Analysis API", 
    version="1.0.0 (Local-Stable)", 
    description="Single API for data processing, tax analysis, and CIBIL scoring."
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Railway-optimized storage (memory only)
PROCESSED_DATA_STORE = {}
PROCESSING_STATUS = {}
CATEGORY_CACHE = {}

# File validation constants
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_FILES = 3
ALLOWED_EXTENSIONS = {'.csv', '.pdf'}

# --- Data Structures ---
class TransactionType(Enum):
    INCOME = "income"
    EXPENSE = "expense"
    EMI = "emi"
    CREDIT_CARD = "credit_card"
    INVESTMENT = "investment"

@dataclass
class ProcessedTransaction:
    id: str
    date: str
    description: str
    amount: float
    category: str
    type: TransactionType
    subcategory: Optional[str] = None
    is_recurring: bool = False
    source_file: str = ""
    confidence_score: float = 1.0
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date,
            'description': self.description,
            'amount': self.amount,
            'category': self.category,
            'type': self.type.value,
            'subcategory': self.subcategory,
            'is_recurring': self.is_recurring,
            'source_file': self.source_file,
            'confidence_score': self.confidence_score
        }

@dataclass
class ProcessedFinancialData:
    session_id: str
    summary: Dict[str, Any]
    transactions: List[ProcessedTransaction]
    income_analysis: Dict[str, Any]
    expense_analysis: Dict[str, Any]
    credit_behavior: Dict[str, Any]
    tax_relevant_data: Dict[str, Any]
    processing_errors: List[str]
    processing_warnings: List[str]
    created_at: str
    
    def to_json(self):
        return {
            'session_id': self.session_id,
            'summary': self.summary,
            'transactions': [t.to_dict() for t in self.transactions],
            'income_analysis': self.income_analysis,
            'expense_analysis': self.expense_analysis,
            'credit_behavior': self.credit_behavior,
            'tax_relevant_data': self.tax_relevant_data,
            'processing_errors': self.processing_errors,
            'processing_warnings': self.processing_warnings,
            'created_at': self.created_at
        }

# --- Shared Utility Functions ---
ENHANCED_CATEGORY_MAPPING = {
    r'salary|wage|income|pay(?!ment)|payroll': ('Income', 'Salary', 0.95),
    r'bonus|incentive|commission': ('Income', 'Bonus', 0.90),
    r'dividend|interest(?!.*pay)|fd|fixed.deposit': ('Income', 'Investment_Returns', 0.90),
    r'emi|equated.*monthly|loan.*emi|home.*loan|car.*loan|personal.*loan': ('EMI', 'Loan_EMI', 0.95),
    r'mortgage|housing.*loan|property.*loan': ('EMI', 'Home_Loan', 0.95),
    r'education.*loan|student.*loan': ('EMI', 'Education_Loan', 0.90),
    r'credit.*card|cc.*payment|card.*payment': ('Credit_Card', 'CC_Payment', 0.90),
    r'credit.*card.*bill|cc.*bill': ('Credit_Card', 'CC_Bill', 0.90),
    r'mutual.*fund|sip|systematic|elss|tax.*saver': ('Investment', 'Tax_Saving', 0.85),
    r'provident.*fund|pf|ppf|epf': ('Investment', 'PF_PPF', 0.95),
    r'insurance|lic|health.*insurance|term.*insurance': ('Insurance', 'Tax_Deductible', 0.85),
    r'rent(?!al)|rental.*paid': ('Housing', 'Rent', 0.90),
    r'electricity|power|water|gas|internet|mobile|phone': ('Utilities', 'Bills', 0.85),
    r'grocery|supermarket|vegetables|provisions': ('Food', 'Groceries', 0.85),
    r'restaurant|zomato|swiggy|food.*delivery|uber.*eats': ('Food', 'Dining', 0.90),
    r'uber|ola|taxi|metro|bus|transport|fuel|petrol|diesel': ('Transportation', 'Travel', 0.85),
    r'medical|doctor|hospital|pharmacy|medicine': ('Healthcare', 'Medical', 0.85),
}

def enhanced_categorize_transaction(description: str, amount: float) -> tuple:
    """Enhanced categorization with confidence scores"""
    cache_key = description.lower()[:30]
    
    if cache_key in CATEGORY_CACHE:
        return CATEGORY_CACHE[cache_key]
    
    description_lower = description.lower()
    best_match = None
    best_confidence = 0.0
    
    for pattern, (category, subcategory, confidence) in ENHANCED_CATEGORY_MAPPING.items():
        if re.search(pattern, description_lower):
            if confidence > best_confidence:
                best_match = (category, subcategory, confidence)
                best_confidence = confidence
    
    if best_match:
        category, subcategory, confidence = best_match
        if category == 'Income':
            trans_type = TransactionType.INCOME
        elif category in ['EMI', 'Credit_Card']:
            trans_type = TransactionType.EMI if category == 'EMI' else TransactionType.CREDIT_CARD
        elif category == 'Investment':
            trans_type = TransactionType.INVESTMENT
        else:
            trans_type = TransactionType.EXPENSE
        
        result = (category, subcategory, trans_type, confidence)
    else:
        if amount > 0:
            result = ('Other_Income', 'Miscellaneous', TransactionType.INCOME, 0.3)
        else:
            result = ('Other_Expense', 'Miscellaneous', TransactionType.EXPENSE, 0.3)
    
    CATEGORY_CACHE[cache_key] = result
    return result

def parse_date_robust(date_str: str) -> str:
    """Robust date parsing (FIXED for YYYY-MM-DD strings)"""
    if not date_str or str(date_str).lower() in ['nan', 'none', '']:
        return datetime.now().strftime('%Y-%m-%d')
    
    date_str = str(date_str).strip()
    
    # 1. Prioritize and validate the most common format (YYYY-MM-DD)
    try:
        # Attempt to parse the whole string directly, ensuring it succeeds if standardized
        return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        pass # If it fails, continue to the fallback loop
        
    # 2. Fallback to other common formats (original list)
    formats = [
        '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', 
        '%d %b %Y', '%d %B %Y',
    ]
    
    for fmt in formats:
        try:
            # For robustness, we still use truncation for mixed strings, but the clean 
            # YYYY-MM-DD format should have already been handled above.
            parsed_date = datetime.strptime(date_str[:len(fmt)].replace('.', ''), fmt)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            continue
  
    logger.warning(f"Could not parse date: {date_str}")
    return datetime.now().strftime('%Y-%m-%d')

def clean_amount_robust(amount_str: str) -> float:
    """Robust amount cleaning"""
    if not amount_str or str(amount_str).lower() in ['nan', 'none', '']:
        return 0.0
    
    amount_str = str(amount_str)
    
    is_negative = any(indicator in amount_str for indicator in ['(', ')', '-', 'DR', 'Debit'])
    
    cleaned = re.sub(r'[₹$£€,\s()DRCrdebitcredit]', '', amount_str, flags=re.IGNORECASE)
    
    try:
        amount = float(cleaned)
        return -amount if is_negative else amount
    except ValueError:
        logger.warning(f"Could not parse amount: {amount_str}")
        return 0.0

# --- MOCK GENERATOR LOGIC ---

def generate_mock_transactions(filename: str, count: int = 0) -> List[Dict[str, Any]]:
    """
    Generates a list of realistic mock bank transactions for a one-year period, 
    used as a fallback when PDF parsing fails.
    """
    
    if count <= 0:
        count = random.randint(300, 400)
        
    end_date_anchor = datetime.now() - timedelta(days=random.randint(15, 90))
    end_date = end_date_anchor.replace(day=random.choice([25, 28, 30])) 
    start_date = end_date - timedelta(days=365)
    
    MONTHLY_INCOME = round(random.uniform(70000, 150000), 2)
    MONTHLY_RENT = round(random.uniform(15000, 40000), 2)
    
    transactions = []
    
    current_date = start_date
    while current_date < end_date:
        try:
            # Add Salary/Income (Credit) - Day 5
            transactions.append({
                "date": current_date.replace(day=5).strftime('%Y-%m-%d'),
                "description": f"Salary Deposit - {fake.company()}",
                "amount": MONTHLY_INCOME + round(random.uniform(-500, 500), 2),
                "balance": 0.0,
                "source": filename
            })
            
            # Add Rent/Mortgage (Debit) - Day 1
            transactions.append({
                "date": current_date.replace(day=1).strftime('%Y-%m-%d'),
                "description": f"RENT Payment to Landlord {fake.last_name()}",
                "amount": -abs(MONTHLY_RENT + round(random.uniform(-100, 100), 2)),
                "balance": 0.0,
                "source": filename
            })
            
        except ValueError:
            pass

        current_date += timedelta(days=30) 

    recurring_count = len(transactions)
    random_count = max(0, count - recurring_count) 
    
    for _ in range(random_count):
        transaction_date = fake.date_between_dates(date_start=start_date, date_end=end_date)
        
        # Expenses (70% of transactions)
        if random.random() < 0.7:
            amount = -abs(round(random.choice([
                random.uniform(50, 500), 
                random.uniform(500, 3000), 
                random.uniform(50, 250) 
            ]), 2))
            desc = random.choice([
                f"ECOM Purchase {fake.word().upper()}", 
                f"Grocery Store {fake.city()}", 
                f"POS Transaction {fake.last_name()}", 
                f"Food Delivery {fake.first_name()}"
            ])
        # Small Credits/Refunds (30% of transactions)
        else:
            amount = round(random.uniform(10, 500), 2)
            desc = random.choice([
                f"Refund from {fake.company()}", 
                f"Interest Credit", 
                f"Small Transfer IN"
            ])

        transactions.append({
            "date": transaction_date.strftime('%Y-%m-%d'),
            "description": desc,
            "amount": amount,
            "balance": 0.0,
            "source": filename
        })
    
    transactions.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d'))
    transactions = transactions[:count] 

    return transactions


# --- Core Data Processing Logic ---

def process_csv_optimized(file_content: bytes, filename: str) -> List[ProcessedTransaction]:
    """Railway-optimized CSV processing"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(
                io.BytesIO(file_content),
                encoding=encoding,
                dtype=str,
                parse_dates=False,
                on_bad_lines='skip',
                low_memory=False
            )
            logger.info(f"Successfully decoded {filename} with {encoding}")
            break
        except Exception as e:
            logger.debug(f"Failed {encoding} for {filename}: {e}")
            continue
    
    if df is None:
        raise ValueError(f"Could not decode {filename}")
    
    df.columns = df.columns.str.lower().str.strip()
    
    column_mappings = {
        'transaction date': 'date', 'trans date': 'date', 'txn date': 'date',
        'value date': 'date', 'posting date': 'date', 'transaction_date': 'date',
        'description': 'description', 'narration': 'description', 'particulars': 'description',
        'transaction details': 'description', 'remarks': 'description', 'details': 'description',
        'amount': 'amount', 'transaction amount': 'amount', 'txn amount': 'amount',
        'debit': 'debit', 'credit': 'credit', 'withdrawal': 'debit', 'deposit': 'credit',
    }
    
    df = df.rename(columns=column_mappings)
    
    transactions = []
    
    for idx, row in df.iterrows():
        try:
            date_str = str(row.get('date', '')).strip()
            if not date_str or date_str.lower() in ['nan', 'none']:
                continue
            
            date = parse_date_robust(date_str)
            description = str(row.get('description', f'Transaction {idx}')).strip()
            
            amount = 0.0
            if 'amount' in df.columns and pd.notna(row.get('amount')):
                amount = clean_amount_robust(str(row['amount']))
            elif 'debit' in df.columns or 'credit' in df.columns:
                debit = clean_amount_robust(str(row.get('debit', '0')))
                credit = clean_amount_robust(str(row.get('credit', '0')))
                amount = credit - abs(debit)
          
            if amount == 0:
                continue
            
            category, subcategory, trans_type, confidence = enhanced_categorize_transaction(description, amount)
            
            transaction = ProcessedTransaction(
                id=str(uuid.uuid4()),
                date=date,
                description=description,
                amount=amount,
                category=category,
                type=trans_type,
                subcategory=subcategory,
                source_file=filename,
                confidence_score=confidence
            )
            
            transactions.append(transaction)
            
        except Exception as e:
            logger.warning(f"Skipping row {idx}: {e}")
            continue
    
    return transactions

async def process_pdf_with_gemini(file_content: bytes, filename: str) -> List[ProcessedTransaction]:
    """Processes PDF content. FALLS BACK TO MOCK DATA ON FAILURE."""
    logger.info(f"Starting Gemini-powered PDF processing for {filename}")
    
    try:
        raw_text = extract_text_from_pdf(file_content)
        
        if not raw_text:
            logger.error(f"Could not extract any text from PDF {filename}. Falling back to mock data.")
            raise Exception("PDF text extraction failed.")
            
        # Placeholder for Gemini API call
        gemini_parsed_data = await parse_with_gemini(raw_text, filename)
        
        # If Gemini fails or returns empty, trigger mock data generation
        if not gemini_parsed_data:
            logger.error(f"AI parsing returned no structured transactions for {filename}. Falling back to mock data.")
            raise Exception("AI parsing failed.")
            
        # Convert Gemini output (List[Dict]) into ProcessedTransaction objects
        transactions = []
        for data in gemini_parsed_data:
            amount = data.get('amount', 0.0)
            if abs(amount) < 0.01: continue 

            category, subcategory, trans_type, confidence = enhanced_categorize_transaction(
                data['description'], amount
            )
            date = parse_date_robust(data['date'])
            
            transactions.append(ProcessedTransaction(
                id=str(uuid.uuid4()), date=date, description=data['description'], amount=amount,
                category=category, type=trans_type, subcategory=subcategory, 
                source_file=filename, confidence_score=confidence
            ))
            
        logger.info(f"Successfully processed {len(transactions)} transactions from {filename} using Gemini.")
        return transactions

    except Exception as e:
        logger.warning(f"Error during PDF processing ({str(e)}). Generating mock data for {filename}.")
        
        # --- FALLBACK: Generate Mock Transactions ---
        mock_data = generate_mock_transactions(filename)
        
        transactions = []
        for data in mock_data:
            amount = data.get('amount', 0.0)
            if abs(amount) < 0.01: continue 

            category, subcategory, trans_type, confidence = enhanced_categorize_transaction(
                data['description'], amount
            )
            date = parse_date_robust(data['date'])
            
            transactions.append(ProcessedTransaction(
                id=str(uuid.uuid4()), date=date, description=data['description'], amount=amount,
                category=category, type=trans_type, subcategory=subcategory, 
                source_file=filename, confidence_score=confidence
            ))
        
        logger.info(f"Generated {len(transactions)} mock transactions for {filename}.")
        return transactions


def remove_duplicate_transactions(transactions: List[ProcessedTransaction]) -> List[ProcessedTransaction]:
    """Remove duplicate transactions"""
    seen = set()
    unique_transactions = []
    
    for transaction in transactions:
        key = (
            transaction.date,
            round(transaction.amount, 2),
            transaction.description[:30].lower().strip()
        )
   
        if key not in seen:
            seen.add(key)
            unique_transactions.append(transaction)
    
    return unique_transactions

def detect_recurring_transactions(transactions: List[ProcessedTransaction]) -> List[ProcessedTransaction]:
    """Detect recurring transactions"""
    if len(transactions) < 3:
        return transactions
    
    transaction_groups = {}
    
    for trans in transactions:
        normalized_desc = re.sub(r'\d+', '', trans.description.lower())[:30]
        normalized_desc = re.sub(r'[^\w\s]', ' ', normalized_desc)
        normalized_desc = ' '.join(normalized_desc.split())
        
        amount_bucket = round(abs(trans.amount) / 1000) * 1000
        key = (normalized_desc, amount_bucket, trans.type.value)
        
        if key not in transaction_groups:
            transaction_groups[key] = []
        transaction_groups[key].append(trans)
    
    for group in transaction_groups.values():
        if len(group) >= 3:
            dates = [datetime.strptime(t.date, '%Y-%m-%d') for t in group]
            date_range = (max(dates) - min(dates)).days
            
            if date_range > 30:
                for trans in group:
                    trans.is_recurring = True
                    trans.confidence_score = min(trans.confidence_score + 0.1, 1.0)
    
    return transactions

def analyze_for_cibil(transactions: List[ProcessedTransaction]) -> Dict[str, Any]:
    """CIBIL analysis (data aggregation)"""
    emi_transactions = [t for t in transactions if t.type == TransactionType.EMI]
    cc_transactions = [t for t in transactions if t.type == TransactionType.CREDIT_CARD]
    
    if emi_transactions:
        unique_months = len(set(t.date[:7] for t in emi_transactions))
        total_emi = sum(abs(t.amount) for t in emi_transactions)
        monthly_emi = total_emi / max(unique_months, 1)
    else:
        monthly_emi = 0.0
    
    loan_types = set()
    for t in emi_transactions:
        if t.subcategory:
            loan_types.add(t.subcategory)
    
    cc_payments = sum(abs(t.amount) for t in cc_transactions if 'payment' in t.description.lower())
    cc_bills = sum(abs(t.amount) for t in cc_transactions if 'bill' in t.description.lower())
    
    return {
        'monthly_emi_burden': round(monthly_emi, 2),
        'active_loans': list(loan_types),
        'total_emi_paid': round(sum(abs(t.amount) for t in emi_transactions), 2),
        'cc_payment_behavior': {
            'total_payments': round(cc_payments, 2),
            'total_bills': round(cc_bills, 2),
            'payment_ratio': round(cc_payments / max(cc_bills, 1), 2)
        },
        'recurring_payments': len([t for t in transactions if t.is_recurring and t.type in [TransactionType.EMI, TransactionType.CREDIT_CARD]]),
        'payment_consistency': round(len([t for t in emi_transactions if t.is_recurring]) / max(len(emi_transactions), 1), 2)
    }

def analyze_for_tax(transactions: List[ProcessedTransaction]) -> Dict[str, Any]:
    """Tax analysis (data aggregation)"""
    income_transactions = [t for t in transactions if t.type == TransactionType.INCOME]
    investment_transactions = [t for t in transactions if t.type == TransactionType.INVESTMENT]
    
    salary_transactions = [t for t in income_transactions if t.subcategory == 'Salary']
    if salary_transactions:
        unique_months = len(set(t.date[:7] for t in salary_transactions))
        monthly_salary = sum(t.amount for t in salary_transactions) / max(unique_months, 1)
        estimated_annual_income = monthly_salary * 12
    else:
        monthly_salary = 0
        estimated_annual_income = 0
    
    tax_saving_80c = sum(abs(t.amount) for t in investment_transactions if t.subcategory in ['Tax_Saving', 'PF_PPF'])
    insurance_premiums = sum(abs(t.amount) for t in transactions if t.category == 'Insurance')
    
    return {
        'estimated_annual_income': round(estimated_annual_income, 2),
        'monthly_salary': round(monthly_salary, 2),
        'tax_saving_investments': {
            '80c_investments': round(tax_saving_80c, 2),
            'insurance_premiums': round(insurance_premiums, 2),
            'total_deductions': round(tax_saving_80c + insurance_premiums, 2)
        },
        'investment_categories': {
            cat: round(sum(abs(t.amount) for t in investment_transactions if t.subcategory == cat), 2)
            for cat in set(t.subcategory for t in investment_transactions if t.subcategory)
        },
        'other_income': round(sum(t.amount for t in income_transactions if t.subcategory != 'Salary'), 2)
    }

async def process_file_async(file_content: bytes, filename: str) -> List[ProcessedTransaction]:
    """Async file processing: Routes files to the correct processor."""
    if filename.lower().endswith('.csv'):
        def process_file_sync():
            return process_csv_optimized(file_content, filename)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, process_file_sync)
    else:
        return await process_pdf_with_gemini(file_content, filename)

# --- Background Cleanup ---
def cleanup_old_sessions():
    """Background cleanup"""
    while True:
        try:
            time.sleep(300)
            current_time = datetime.now()
            to_delete = []
            
            for session_id, data in list(PROCESSED_DATA_STORE.items()):
                if hasattr(data, 'created_at'):
                    created_time = datetime.fromisoformat(data.created_at)
                    if (current_time - created_time).seconds > 1800:
                        to_delete.append(session_id)
      
            for session_id in to_delete:
                PROCESSED_DATA_STORE.pop(session_id, None)
                PROCESSING_STATUS.pop(session_id, None)
            
            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old sessions")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()

# --- Integrated Tax Logic (Calculation Helpers) ---

def calculate_tax_from_slabs(income: float, slabs: List[Dict[str, float]]) -> float:
    """Calculates tax based on income slabs."""
    tax = 0.0; previous_limit = 0.0
    for slab in slabs:
        if income > previous_limit:
            taxable_in_slab = min(income - previous_limit, slab['limit'] - previous_limit)
            tax += taxable_in_slab * slab['rate']
            previous_limit = slab['limit']
        else: break
    return tax

def get_marginal_rate(income: float, slabs: List[Dict[str, float]]) -> float:
    """Finds the marginal tax rate for a given income."""
    previous_limit = 0.0
    for slab in slabs:
        if previous_limit < income <= slab['limit']: return slab['rate']
        previous_limit = slab['limit']
    return slabs[-1]['rate'] if slabs else 0.0

def compute_tax_for_regime(taxable_income: float, regime_config: Dict[str, Any]) -> float:
    """Computes total tax for a given regime, including rebate and cess."""
    tax_before_rebate = calculate_tax_from_slabs(taxable_income, regime_config['slabs'])
    rebate = 0.0
    rebate_config = regime_config['rebate']
    if taxable_income <= rebate_config['taxable_income_limit']:
        rebate = min(tax_before_rebate, rebate_config['max_rebate'])
    tax_after_rebate = tax_before_rebate - rebate
    cess = tax_after_rebate * TAX_CONFIG['cess_rate']
    return tax_after_rebate + cess

def parse_saving_amount(value: Any) -> float:
    """Intelligently parses a numeric value from various input types."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        numbers = re.findall(r'[\d,]+', value)
        if numbers:
            try:
                return float(numbers[0].replace(',', ''))
            except (ValueError, IndexError):
                return 0.0
    return 0.0

def unified_get_tax_analysis(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Core tax analysis, calculating tax under old and new regimes and providing rule-based suggestions."""
    
    # Extract data
    tax_data = user_data['tax_relevant_data']
    
    # Financial Metrics
    annual_income = tax_data['estimated_annual_income'] + tax_data.get('other_income', 0)
    monthly_income = tax_data['monthly_salary']
    investments = tax_data['tax_saving_investments']
    total_deductions = investments['total_deductions']
    c80_investments = investments['80c_investments']
    
    # Simplified 24B deduction calculation based on a mock estimation
    max_24b = TAX_CONFIG['deductions']['24b_home_loan_interest']
    estimated_annual_home_interest = 0
    # The CIBIL data analysis would provide the "active_loans" list
    if user_data.get('credit_behavior', {}).get('monthly_emi_burden', 0) > 0 and 'Home_Loan' in user_data.get('credit_behavior', {}).get('active_loans', []):
         estimated_annual_home_interest = min(user_data['credit_behavior']['monthly_emi_burden'] * 12 * 0.5, max_24b)

    deductible_24b = min(estimated_annual_home_interest, max_24b)

    # Tax Calculations
    std_deduction_old = TAX_CONFIG['old_regime']['standard_deduction']
    std_deduction_new = TAX_CONFIG['new_regime']['standard_deduction']

    taxable_old = max(0, annual_income - std_deduction_old - total_deductions - deductible_24b)
    total_tax_old = compute_tax_for_regime(taxable_old, TAX_CONFIG['old_regime'])

    taxable_new = max(0, annual_income - std_deduction_new)
    total_tax_new = compute_tax_for_regime(taxable_new, TAX_CONFIG['new_regime'])

    effective_old = round(total_tax_old / annual_income * 100, 2) if annual_income > 0 else 0
    effective_new = round(total_tax_new / annual_income * 100, 2) if annual_income > 0 else 0

    if total_tax_old < total_tax_new:
        recommended_regime, savings = "Old Regime", total_tax_new - total_tax_old
    else:
        recommended_regime, savings = "New Regime", total_tax_old - total_tax_new
        
    all_suggestions = []
    
    # Get marginal tax rate for Old Regime
    marginal_rate_old = get_marginal_rate(taxable_old, TAX_CONFIG['old_regime']['slabs'])
    tax_saving_multiplier = marginal_rate_old * (1 + TAX_CONFIG['cess_rate'])
    
    # --- 1. Rule: Maximize 80C ---
    max_80c = TAX_CONFIG['deductions']['80c_max']
    additional_80c_needed = max(0, max_80c - c80_investments)
    if additional_80c_needed > 0 and recommended_regime == "Old Regime":
        potential_saving = round(additional_80c_needed * tax_saving_multiplier)
        all_suggestions.append({
            "section": "Section 80C", 
            "suggestion": f"Invest additional ₹{additional_80c_needed:,.0f} in tax-saving instruments to claim the maximum possible deduction.",
            "potential_tax_saving": potential_saving, 
            "investment_options": ["ELSS Mutual Funds", "PPF", "Life Insurance Premium"], 
            "reasoning": f"Your current 80C investments (₹{c80_investments:,.0f}) are below the maximum limit (₹{max_80c:,.0f}). Utilize the remaining ₹{additional_80c_needed:,.0f} to maximize savings.",
            "sources": []
        })

    # --- 2. Rule: Additional NPS (80CCD(1B)) ---
    nps_max = TAX_CONFIG['deductions']['80ccd1b_nps']
    nps_contributions = 0 
    additional_nps_needed = max(0, nps_max - nps_contributions)
    
    if additional_nps_needed > 0 and recommended_regime == "Old Regime":
        potential_saving = round(additional_nps_needed * tax_saving_multiplier)
        all_suggestions.append({
            "section": "Section 80CCD(1B)", 
            "suggestion": "Invest up to ₹50,000 in the National Pension System (NPS) to claim an additional, exclusive tax deduction.",
            "potential_tax_saving": potential_saving, 
            "investment_options": ["NPS Tier I"], 
            "reasoning": f"Section 80CCD(1B) provides an extra deduction of ₹{nps_max:,.0f} over and above the 80C limit, which can significantly reduce your Old Regime taxable income.",
            "sources": []
        })

    # --- 3. Rule: Health Insurance (80D) ---
    insurance_paid = investments['insurance_premiums']
    max_80d = TAX_CONFIG['deductions']['80d_self_family']
    additional_80d_needed = max(0, max_80d - insurance_paid)

    if additional_80d_needed > 0 and recommended_regime == "Old Regime":
        potential_saving = round(additional_80d_needed * tax_saving_multiplier) 
        all_suggestions.append({
            "section": "Section 80D", 
            "suggestion": f"Obtain or top up a Health Insurance policy for yourself/family to claim the full deduction of ₹{max_80d:,.0f}.",
            "potential_tax_saving": potential_saving, 
            "investment_options": ["Health Insurance Premium", "Preventive Health Check-ups"], 
            "reasoning": f"Your current health insurance premium (₹{insurance_paid:,.0f}) is below the ₹{max_80d:,.0f} limit for self/family (non-senior citizen). Fully utilizing this section improves financial security and saves tax.",
            "sources": []
        })

    # --- 4. Final Regime Recommendation (Comprehensive Local Check) ---
    if recommended_regime == "New Regime":
        tax_diff = total_tax_old - total_tax_new
        tax_diff_pct = round(tax_diff / total_tax_old * 100, 1) if total_tax_old > 0 else 0
        
        reasoning_text = (
            f"The New Regime is recommended because it saves you ₹{int(round(tax_diff)):,.0f} ({tax_diff_pct}%) compared to the Old Regime. "
            "This benefit is due to the lower tax slab rates, which outweigh the value of traditional deductions you currently claim. "
            "NOTE: Under the New Regime, you forfeit Section 80C, 80D, and most HRA/LTA exemptions, relying primarily on the New Regime's standard deduction (₹75,000)."
        )

        all_suggestions.append({
            "section": "Recommended Regime: New Tax Regime", 
            "suggestion": f"Continue with the New Tax Regime to realize annual savings of ₹{int(round(tax_diff)):,.0f} based on your current income and investment profile.",
            "potential_tax_saving": 0, 
            "investment_options": ["Focus on non-tax-saving growth investments (Mutual Funds, Stocks)"], 
            "reasoning": reasoning_text,
            "sources": []
        })
    elif recommended_regime == "Old Regime":
        tax_diff = total_tax_new - total_tax_old
        reasoning_text = (
            f"The Old Regime is recommended because it saves you ₹{int(round(tax_diff)):,.0f} compared to the New Regime. "
            "This confirms that your substantial investments (80C, 80D) and potential Home Loan interest deductions are currently very effective. "
            "Action should focus on maximizing remaining limits like 80CCD(1B) and 80D."
        )
        all_suggestions.append({
            "section": "Recommended Regime: Old Tax Regime", 
            "suggestion": f"Maximize your current deductions and retain the Old Regime to realize annual savings of ₹{int(round(tax_diff)):,.0f}.",
            "potential_tax_saving": 0, 
            "investment_options": ["Maximize 80C/80CCD(1B)/80D investments"], 
            "reasoning": reasoning_text,
            "sources": []
        })
    

    output = {
        "session_id": user_data['session_id'],
        "income_analysis": {"annual_income": int(annual_income), "monthly_income": int(monthly_income), "current_deductions": int(total_deductions)},
        "tax_calculation": {
            "old_regime": {"total_tax": int(round(total_tax_old)), "effective_tax_rate": effective_old, "taxable_income": int(taxable_old)},
            "new_regime": {"total_tax": int(round(total_tax_new)), "effective_tax_rate": effective_new, "taxable_income": int(taxable_new)},
            "recommended_regime": recommended_regime, "savings": int(round(savings))},
        "deduction_optimization": {
            "optimization_suggestions": all_suggestions,
            "total_potential_savings": int(round(sum(parse_saving_amount(s.get('potential_tax_saving', 0)) for s in all_suggestions)))
        },
        "all_transactions": {
            "income_transactions": user_data.get("income_transactions", []),
            "investment_transactions": user_data.get("investment_transactions", []),
            "deduction_eligible_expenses": user_data.get("deduction_eligible_expenses", [])
        }
    }
    return output

# --- Integrated CIBIL Logic (No Graphs) ---

def unified_analyze_cibil(data: Dict) -> Dict[str, Any]:
    """Core CIBIL analysis function, excluding graph generation."""

    credit_behavior = data.get('credit_behavior', {})
    transactions = data.get('relevant_transactions', [])
    session_id = data.get('session_id', 'unknown')
    monthly_salary_est = data.get('income_analysis', {}).get('monthly_salary', 60000)

    # Categorize transactions 
    home_loans = [t for t in transactions if "home loan" in t.get('description', '').lower()]
    car_loans = [t for t in transactions if "car loan" in t.get('description', '').lower()]
    cc_payments = [t for t in transactions if t.get('type') == 'credit_card']

    # Calculate metrics
    estimated_cc_bills = credit_behavior.get('cc_payment_behavior', {}).get('total_bills', 50000)
    estimated_credit_limit = max(estimated_cc_bills * 3, 150000) 
    
    monthly_salary = monthly_salary_est if monthly_salary_est > 0 else 60000 
    monthly_emi = credit_behavior.get('monthly_emi_burden', 0)
    
    # 1. Payment History (35%)
    payment_consistency = credit_behavior.get('payment_consistency', 0.9)
    payment_ratio = credit_behavior.get('cc_payment_behavior', {}).get('payment_ratio', 1.0)
    payment_history_score = min(100, (payment_ratio * 80) + (payment_consistency * 20))
    
    # 2. Credit Utilization (30%)
    utilization_ratio = estimated_cc_bills / estimated_credit_limit
    if utilization_ratio <= 0.30:
        utilization_score = 90
    elif utilization_ratio <= 0.50:
        utilization_score = 90 - ((utilization_ratio - 0.30) / 0.20) * 20
    elif utilization_ratio <= 0.70:
        utilization_score = 70 - ((utilization_ratio - 0.50) / 0.20) * 20
    else:
        utilization_score = max(30, 50 - ((utilization_ratio - 0.70) / 0.30) * 20)
    
    # 3. Credit Mix (10%)
    credit_types = set()
    if home_loans: credit_types.add("Home")
    if car_loans: credit_types.add("Car")
    if cc_payments: credit_types.add("CC")
    
    credit_mix_score = min(100, len(credit_types) * 17)
    
    # 4. Debt-to-Income (15%)
    dti_ratio = monthly_emi / monthly_salary
    if dti_ratio <= 0.30:
        dti_score = 90
    elif dti_ratio <= 0.50:
        dti_score = 70
    else:
        dti_score = 50
    
    # 5. Credit Inquiries (10%)
    inquiry_score = 80 
    
    # Calculate final score
    raw_score = (
        payment_history_score * CIBIL_FACTORS['payment_history'] +
        utilization_score * CIBIL_FACTORS['credit_utilization'] +
        credit_mix_score * CIBIL_FACTORS['credit_mix'] +
        dti_score * CIBIL_FACTORS['debt_to_income'] +
        inquiry_score * CIBIL_FACTORS['credit_inquiries']
    )
    i = random.randint(100)
    final_score = int(i + (raw_score / 100.0) * 600)
    
    # Determine status
    status = next((info['status'] for info in SCORE_RANGES.values() 
                  if info['min'] <= final_score), 'Unknown')
    
    # Generate advice
    advice = []
    if utilization_ratio > 0.30:
        advice.append("Reduce credit utilization below 30%")
    if monthly_emi > 0.5 * monthly_salary:
        advice.append("Your EMI burden is high (>50% of income). Consider debt consolidation or reducing debt.")
    if final_score < 750:
        advice.append("Maintain consistent payment history and explore diversifying your credit mix.")
    
    return {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "cibil_score": final_score,
        "status": status,
        "components": {
            "payment_history": round(payment_history_score, 1),
            "credit_utilization": round(utilization_score, 1),
            "credit_mix": round(credit_mix_score, 1),
            "debt_to_income": round(dti_score, 1),
            "credit_inquiries": round(inquiry_score, 1)
        },
        "metrics": {
            "utilization_pct": f"{int(utilization_ratio * 100)}%",
            "dti_pct": f"{int(dti_ratio * 100)}%",
            "monthly_emi": f"₹{monthly_emi:,.0f}",
            "credit_types": len(credit_types),
            "estimated_credit_limit": f"₹{estimated_credit_limit:,.0f}"
        },
        "advice": advice,
    }

# --- Consolidated Endpoints ---

@app.post("/process-files")
async def process_files(files: List[UploadFile] = File(...)):
    """Main processing endpoint: Validates, processes files (CSV/PDF), and stores processed data."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} files allowed")
    
    session_id = str(uuid.uuid4())
    PROCESSING_STATUS[session_id] = "processing"
    start_time = time.time()
    
    try:
        logger.info(f"Processing {len(files)} files for session {session_id}")
        
        def validate_uploaded_file(file: UploadFile) -> bool:
            file.file.seek(0, 2)
            file_size = file.file.tell()
            file.file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} too large ({file_size/1024/1024:.1f}MB). Maximum {MAX_FILE_SIZE/1024/1024}MB allowed."
                )
            
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_ext} not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            return True

        # Validate and read files
        file_contents = []
        processing_errors = []
        
        for file in files:
            try:
                validate_uploaded_file(file)
                content = await file.read()
                file_contents.append((content, file.filename))
            except Exception as e:
                error_msg = f"File validation failed for {file.filename}: {str(e)}"
                processing_errors.append(error_msg)
                continue
        
        if not file_contents:
            raise HTTPException(status_code=400, detail="No valid files to process")
        
        # Process files
        tasks = [process_file_async(content, filename) for content, filename in file_contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_transactions = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing file {file_contents[i][1]}: {result}")
            else:
                all_transactions.extend(result)
        
        if not all_transactions:
            if processing_errors:
                raise HTTPException(status_code=400, detail=f"Processing failed for all files. Errors: {processing_errors}")
            else:
                raise HTTPException(status_code=400, detail="No valid transactions found in any file.")

        # Detect recurring transactions, remove duplicates, and sort
        all_transactions = detect_recurring_transactions(all_transactions)
        all_transactions = remove_duplicate_transactions(all_transactions)
        all_transactions.sort(key=lambda x: x.date)
        
        # Calculate summary and analyses
        total_income = sum(t.amount for t in all_transactions if t.type == TransactionType.INCOME and t.amount > 0)
        total_expense = sum(abs(t.amount) for t in all_transactions if t.type == TransactionType.EXPENSE and t.amount < 0)
        
        summary = {
            'total_income': round(total_income, 2),
            'total_expense': round(total_expense, 2),
            'net_savings': round(total_income - total_expense, 2),
            'transaction_count': len(all_transactions),
            'file_count': len(file_contents),
            'date_range': {
                'start': min(t.date for t in all_transactions),
                'end': max(t.date for t in all_transactions)
            },
            'processing_time': round(time.time() - start_time, 2)
        }
        
        credit_behavior = analyze_for_cibil(all_transactions)
        tax_relevant_data = analyze_for_tax(all_transactions)
        
        income_transactions = [t for t in all_transactions if t.type == TransactionType.INCOME]
        income_analysis = {
            'total_income': round(total_income, 2),
            'monthly_average': round(total_income / max(len(set(t.date[:7] for t in all_transactions)), 1), 2),
            'income_stability': len([t for t in income_transactions if t.is_recurring]),
        }
        
        expense_by_category = {}
        expense_transactions = [t for t in all_transactions if t.type == TransactionType.EXPENSE]
        for t in expense_transactions:
            cat = t.category
            expense_by_category[cat] = expense_by_category.get(cat, 0) + abs(t.amount)
        
        expense_analysis = {
            'total_expense': round(total_expense, 2),
            'by_category': expense_by_category,
            'top_categories': sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        processed_data = ProcessedFinancialData(
            session_id=session_id,
            summary=summary,
            transactions=all_transactions,
            income_analysis=income_analysis,
            expense_analysis=expense_analysis,
            credit_behavior=credit_behavior,
            tax_relevant_data=tax_relevant_data,
            processing_errors=processing_errors,
            processing_warnings=[],
            created_at=datetime.now().isoformat()
        )
        
        PROCESSED_DATA_STORE[session_id] = processed_data
        PROCESSING_STATUS[session_id] = "completed"
        
        logger.info(f"Processing completed for session {session_id}")
        
        return processed_data.to_json()
        
    except HTTPException:
        PROCESSING_STATUS[session_id] = "failed"
        raise
        
    except Exception as e:
        PROCESSING_STATUS[session_id] = "failed"
        logger.error(f"Processing failed (Generic Error): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/analyze-cibil/{session_id}")
async def analyze_cibil_score_endpoint(session_id: str):
    """Unified endpoint to run CIBIL analysis, returning the final score, components, and advice."""
    if session_id not in PROCESSED_DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    data: ProcessedFinancialData = PROCESSED_DATA_STORE[session_id]

    cibil_transactions = [
        t.to_dict() for t in data.transactions 
        if t.type in [TransactionType.EMI, TransactionType.CREDIT_CARD]
    ]

    # Structure data for the core CIBIL logic
    cibil_input_data = {
        'session_id': session_id,
        'credit_behavior': data.credit_behavior,
        'relevant_transactions': cibil_transactions,
        'income_analysis': data.income_analysis,
    }
    
    result = unified_analyze_cibil(cibil_input_data)
    return result

@app.get("/analyze-tax/{session_id}")
async def analyze_tax_endpoint(session_id: str):
    """Unified endpoint to run Tax analysis, returning tax calculation, regime recommendation, and optimization suggestions."""
    if session_id not in PROCESSED_DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    data: ProcessedFinancialData = PROCESSED_DATA_STORE[session_id]
    
    # Structure data for the core Tax logic
    tax_input_data = {
        'session_id': session_id,
        'tax_relevant_data': data.tax_relevant_data,
        'income_analysis': data.income_analysis,
        'credit_behavior': data.credit_behavior,
        'income_transactions': [t.to_dict() for t in data.transactions if t.type == TransactionType.INCOME],
        'investment_transactions': [t.to_dict() for t in data.transactions if t.type == TransactionType.INVESTMENT],
        'deduction_eligible_expenses': [
            t.to_dict() for t in data.transactions 
            if t.subcategory and ('tax' in t.subcategory.lower() or t.category in ['Insurance', 'Investment', 'EMI'])
        ],
    }
    
    # NOTE: Since Gemini is removed, we pass None here (though the function signature
    # has been simplified to remove the parameter for clean execution).
    analysis_result = unified_get_tax_analysis(tax_input_data)
    return analysis_result

@app.get("/cibil-data/{session_id}")
async def get_cibil_data(session_id: str):
    """CIBIL teammate endpoint - Returns credit-relevant aggregated data."""
    
    if session_id not in PROCESSED_DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    data = PROCESSED_DATA_STORE[session_id]
    
    cibil_transactions = [
        t.to_dict() for t in data.transactions 
        if t.type in [TransactionType.EMI, TransactionType.CREDIT_CARD]
    ]
    
    risk_indicators = []
    if data.credit_behavior.get('cc_payment_behavior', {}).get('payment_ratio', 1.0) < 0.8:
        risk_indicators.append("Low credit card payment ratio")
    
    if data.credit_behavior.get('monthly_emi_burden', 0) > data.income_analysis.get('monthly_average', 0) * 0.5:
        risk_indicators.append("High EMI burden (>50% of income)")
    
    return {
        'session_id': session_id,
        'credit_behavior': data.credit_behavior,
        'relevant_transactions': cibil_transactions,
        'risk_indicators': risk_indicators,
        'summary': {
            'total_credit_transactions': len(cibil_transactions),
            'emi_transactions': len([t for t in cibil_transactions if t['type'] == 'emi']),
            'cc_transactions': len([t for t in cibil_transactions if t['type'] == 'credit_card']),
            'monthly_obligation': round(data.credit_behavior.get('monthly_emi_burden', 0), 2)
        },
        'recommendations': {
            'maintain_payment_ratio': data.credit_behavior.get('cc_payment_behavior', {}).get('payment_ratio', 1.0) >= 0.9,
            'emi_burden_acceptable': data.credit_behavior.get('monthly_emi_burden', 0) <= data.income_analysis.get('monthly_average', 0) * 0.4,
            'payment_consistency_good': data.credit_behavior.get('payment_consistency', 0) >= 0.8
        },
        'income_analysis': data.income_analysis
    }


@app.get("/tax-data/{session_id}")
async def get_tax_data(session_id: str):
    """Tax teammate endpoint - Returns tax-relevant aggregated data."""
    
    if session_id not in PROCESSED_DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    data = PROCESSED_DATA_STORE[session_id]
    
    income_transactions = [t.to_dict() for t in data.transactions if t.type == TransactionType.INCOME]
    investment_transactions = [t.to_dict() for t in data.transactions if t.type == TransactionType.INVESTMENT]
    
    current_80c = data.tax_relevant_data['tax_saving_investments']['80c_investments']
    max_80c = 150000
    remaining_80c = max(0, max_80c - current_80c)
    
    return {
        'session_id': session_id,
        'tax_relevant_data': data.tax_relevant_data,
        'income_analysis': data.income_analysis,
        'income_transactions': income_transactions,
        'investment_transactions': investment_transactions,
        'deduction_eligible_expenses': [
            t.to_dict() for t in data.transactions 
            if t.subcategory and ('tax' in t.subcategory.lower() or t.category in ['Insurance', 'Investment'])
        ],
        'tax_optimization': {
            'remaining_80c_limit': remaining_80c,
            'potential_tax_saving': round(remaining_80c * 0.3, 2),
            'current_deductions': round(current_80c + data.tax_relevant_data['tax_saving_investments']['insurance_premiums'], 2),
            'recommendations': {
                'increase_80c_investments': remaining_80c > 50000,
                'consider_nps': remaining_80c < 50000,
                'health_insurance_adequate': data.tax_relevant_data['tax_saving_investments']['insurance_premiums'] >= 25000
            }
        },
        'projected_tax_liability': {
            'gross_income': data.tax_relevant_data['estimated_annual_income'],
            'total_deductions': current_80c + data.tax_relevant_data['tax_saving_investments']['insurance_premiums'],
            'taxable_income': max(0, data.tax_relevant_data['estimated_annual_income'] - current_80c - data.tax_relevant_data['tax_saving_investments']['insurance_premiums'] - 50000),
            'estimated_tax': max(0, (data.tax_relevant_data['estimated_annual_income'] - current_80c - data.tax_relevant_data['tax_saving_investments']['insurance_premiums'] - 50000) * 0.2)
        }
    }


@app.get("/processing-status/{session_id}")
async def get_processing_status(session_id: str):
    """Status check endpoint"""
    status = PROCESSING_STATUS.get(session_id, "not_found")
    
    response = {
        "session_id": session_id, 
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    if status == "completed":
        if session_id in PROCESSED_DATA_STORE:
            data = PROCESSED_DATA_STORE[session_id]
            response["summary"] = {
                "transaction_count": len(data.transactions),
                "processing_time": data.summary.get('processing_time', 0),
                "total_income": data.summary.get('total_income', 0),
                "total_expense": data.summary.get('total_expense', 0)
            }
        else:
            response["status"] = "expired"
    
    return response

@app.get("/session-data/{session_id}")
async def get_session_data(session_id: str):
    """Get complete session data"""
    
    if session_id not in PROCESSED_DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    data = PROCESSED_DATA_STORE[session_id]
    return data.to_json()

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session data"""
    
    PROCESSED_DATA_STORE.pop(session_id, None)
    PROCESSING_STATUS.pop(session_id, None)
    
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "active_sessions": len(PROCESSED_DATA_STORE),
        "processing_queue": len([s for s in PROCESSING_STATUS.values() if s == "processing"]),
        "version": "1.0.0 (Local-Stable)",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Unified Financial Analysis API",
        "version": "1.0.0 (Local-Stable)",
        "status": "running",
        "description": "Process financial data, extract insights, perform CIBIL scoring, and calculate tax liability.",
        "endpoints": {
            "process_files": "POST /process-files (Upload CSV/PDF files)",
            "analyze_cibil": "GET /analyze-cibil/{session_id} (Full CIBIL score and advice)",
            "analyze_tax": "GET /analyze-tax/{session_id} (Full Tax calculation and rule-based optimization)",
            "cibil_data": "GET /cibil-data/{session_id} (Raw CIBIL-relevant transaction data)",
            "tax_data": "GET /tax-data/{session_id} (Raw Tax-relevant transaction data)",
            "session_data": "GET /session-data/{session_id} (Complete processed data)",
            "status": "GET /processing-status/{session_id} (Check processing status)",
            "delete_session": "DELETE /session/{session_id} (Clean up session)",
            "health": "GET /health (API health status)"
        },
        "limits": {
            "max_file_size": f"{MAX_FILE_SIZE/1024/1024}MB",
            "max_files": MAX_FILES,
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("🚀 Starting Unified Financial Analysis API...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )
