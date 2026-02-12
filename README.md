# GUARDIAN Transaction Monitoring System

Multi-Agent Transaction Monitoring & Access Control System with RAG (Retrieval-Augmented Generation).

## 🚀 Production Deployment

### Prerequisites

- Python 3.10+
- OpenAI API Key

### Quick Start

1. **Clone and Setup**
```bash
cd guardian_system
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
# Copy example env file
copy .env.example .env  # Windows
# or
cp .env.example .env    # Linux/Mac

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

4. **Start Server**
```bash
cd backend
python main.py
```

Server will start at: `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs`
- **Web Interface**: `http://localhost:8000`

## 📁 Project Structure

```
guardian_system/
├── backend/              # Backend Python application
│   ├── agents/          # Multi-agent system (Monitor, Evaluation, Coordinator)
│   ├── api/             # API endpoints
│   ├── config/          # Configuration settings
│   ├── models/          # Database models and schemas
│   ├── services/        # Core services (LLM, embeddings, vector store)
│   ├── utils/           # Helper utilities
│   └── main.py          # Application entry point
├── frontend/            # Frontend web interface
│   └── static/          # HTML, CSS, JavaScript files
├── data/                # Data storage
│   ├── uploads/         # Transaction CSVs and policy PDFs
│   └── chroma_db/       # Vector database
├── .env                 # Environment variables (create from .env.example)
├── .gitignore          # Git ignore rules
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔑 Key Features

- **Real-time Transaction Evaluation** - ALLOW/CHALLENGE/DENY decisions
- **Multi-Agent Architecture** - 3 main agents with 8 sub-agents
- **RAG-based Analysis** - Behavioral patterns and policy compliance
- **Continuous Monitoring** - Auto-loads new CSV and PDF files
- **Adaptive Learning** - Improves from feedback
- **RESTful API** - Easy integration with existing systems

## 📊 API Usage

### Evaluate Transaction
```bash
POST /api/v1/evaluate
{
  "user_id": "Alice",
  "amt": 1250.50,
  "merchant": "Amazon",
  "category": "shopping_net",
  "city": "Seattle",
  "state": "WA"
}
```

### Upload Transaction Data
```bash
POST /api/v1/transactions/upload
# Upload CSV file with transaction history
```

### Upload Policy Document
```bash
POST /api/v1/policies
# Upload PDF policy document
```

## 🏗️ Architecture

### Multi-Agent Pipeline

1. **Monitor Agent** - Continuously watches for new data files
   - Capture Sub-agent: Normalizes transactions
   - Context Sub-agent: Retrieves history
   - Feature Sub-agent: Extracts features

2. **Evaluation Agent** - Analyzes on-demand (parallel)
   - Behavioral Sub-agent: RAG-based anomaly detection
   - Policy Sub-agent: RAG-based compliance checking

3. **Coordinator Agent** - Makes final decisions
   - Fusion Sub-agent: Combines scores
   - Decision Sub-agent: Applies thresholds
   - Learning Sub-agent: Adapts from feedback

## 🔧 Configuration

Edit `.env` file to configure:

- **OpenAI API Key** - Required for LLM analysis
- **Server Settings** - Host, port, debug mode
- **Database Path** - SQLite database location
- **Vector Store** - ChromaDB settings
- **Agent Parameters** - Weights, thresholds, K-values

## 🛡️ Security

- Store `.env` file securely (never commit to git)
- Use environment-specific API keys
- Configure CORS for production
- Implement authentication (not included by default)

## 📝 License

Copyright © 2026. All rights reserved.

## 🤝 Support

For issues or questions, please check the API documentation at `/docs` when the server is running.
