## Mini RAG: Financial Q&A Assistant

A lightweight Retrieval-Augmented Generation (RAG) system built with FastAPI that answers mutual fund-related queries using both:

  📄 Textual FAQs (e.g., "What is SIP?")

  📊 Structured fund data (e.g., "Top funds by Sharpe ratio")

  🔧 Features

1. Answer general finance FAQs: "What is CAGR?", "Explain index funds"

2. Retrieve top-performing mutual funds: "Top funds by Sharpe ratio", "Lowest volatility funds"

3. Dual Retrieval Modes:

  - Semantic Search (Sentence Transformers + FAISS)

  - Lexical Search (TF-IDF + cosine similarity)

4. API built using FastAPI

5. Easy configuration via data.json

6. Supports basic HTML UI for interaction

## 📁 Folder Structure
project/
├── data/               # contains faqs.csv and funds.csv
├── faiss_index.pkl     # stores FAISS index (auto-generated)
├── data.json           # config file for paths
├── main.py             # FastAPI app
├── fromtend.html       # optional frontend (basic UI)
├── requirements.txt    # dependencies
└── README.md           # this file

## 📥 Installation & Setup
### Step 1: Clone or Download
git clone <your_repo_url>
cd <project_folder>


Or unzip the folder if downloaded as a ZIP.

### Step 2: (Optional but Recommended) Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Add Data Files

Place your data files into the data/ folder:

data/
├── faqs.csv      # Columns: 'question', 'answer'
└── funds.csv     # Columns: 'fund_name', 'CAGR 3yr', 'Sharpe Ratio', 'Volatility', etc.

### Step 5: Configure data.json

Make sure your config file looks like this:

{
  "faq_path": "data/faqs.csv",
  "fund_path": "data/funds.csv",
  "embeddings_path": "faiss_index.pkl"
}


Controls dataset paths and where embeddings are cached.

### Step 6: Run the Server
python main.py


Server will start at:

http://localhost:8000

📬 How to Query

Send a POST request to:

http://localhost:8000/query

Sample JSON Body:
{
  "query": "What is an index fund?",
  "mode": "semantic"
}


Returns a structured response:

{
  "answer": "...",
  "sources": [...]
}

💡 Example Queries

"What is CAGR?"

"Explain index funds"

"Top mutual funds by Sharpe ratio"

"Funds with lowest volatility"

🌐 Optional HTML Frontend

Open fromtend.html in a browser to use a basic interface.

CORS is enabled for all origins by default. You can restrict it in main.py if needed.
