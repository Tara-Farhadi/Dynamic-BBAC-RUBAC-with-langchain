# Guardian System - Complete Agent Architecture & Relationships

## Academic Documentation: Multi-Agent System Design

---

## 1. SYSTEM OVERVIEW: THREE-AGENT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GUARDIAN FRAUD DETECTION SYSTEM                   │
│                    Multi-Agent RAG-Based Architecture                │
└─────────────────────────────────────────────────────────────────────┘

                        HIERARCHICAL STRUCTURE
                        ═══════════════════════

LAYER 1: PERCEPTION
┌─────────────────────────────────────────────────────────────────┐
│                     MONITOR AGENT                               │
│                   (Data Collection Layer)                       │
│                                                                 │
│  Role: Capture, contextualize, and extract features             │
│  Trigger: File watcher (background) + API request (on-demand)   │
│  Output: Enriched transaction with features & embeddings        │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ SUB-AGENTS (3):                                           │ │
│  │                                                           │ │
│  │  1. Capture Sub-Agent                                     │ │
│  │     • Normalize transaction data                          │ │
│  │     • Generate transaction_id                             │ │
│  │     • Extract temporal features                           │ │
│  │     • Validate data integrity                             │ │
│  │                                                           │ │
│  │  2. Context Sub-Agent                                     │ │
│  │     • Query user profile from database                    │ │
│  │     • Retrieve transaction history                        │ │
│  │     • Check vector store for user data                    │ │
│  │     • Compile user behavioral baseline                    │ │
│  │                                                           │ │
│  │  3. Feature Sub-Agent                                     │ │
│  │     • Calculate amount statistics (zscore, ratio)         │ │
│  │     • Analyze geographic patterns                         │ │
│  │     • Extract merchant insights                           │ │
│  │     • Compute velocity metrics                            │ │
│  │     • Generate embedding (768-dim vector)                 │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Data Flow
                              ▼
LAYER 2: ANALYSIS
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION AGENT                             │
│                   (RAG-Based Analysis Layer)                    │
│                                                                 │
│  Role: Dual RAG pipelines for behavioral & policy analysis      │
│  Trigger: Receives Monitor Agent output                         │
│  Output: Behavioral assessment + Policy assessment              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ SUB-AGENTS (2) - PARALLEL EXECUTION:                      │ │
│  │                                                           │ │
│  │  1. Behavioral Sub-Agent (RAG Pipeline 1)                 │ │
│  │     • Query: User transaction vector store                │ │
│  │     • Retrieval: K=5 similar historical transactions      │ │
│  │     • Analysis: Statistical anomaly detection             │ │
│  │     • LLM: Contextual reasoning & explanation             │ │
│  │     • Output: Anomaly score (0-1) + confidence            │ │
│  │     • Scoring: 70% statistical + 30% LLM                  │ │
│  │                                                           │ │
│  │  2. Policy Sub-Agent (RAG Pipeline 2)                     │ │
│  │     • Query: Organizational & Regulatory policy stores    │ │
│  │     • Retrieval: K=3 chunks per policy type (6 total)     │ │
│  │     • Analysis: Compliance validation                     │ │
│  │     • LLM: Violation detection & severity                 │ │
│  │     • Output: Policy score (0-1) + violations list        │ │
│  │     • Fusion: Regulatory precedence (≥0.8 override)       │ │
│  │                                                           │ │
│  │  Note: Both sub-agents run simultaneously via             │ │
│  │        asyncio.gather() for optimal performance           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Data Flow
                              ▼
LAYER 3: DECISION
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATOR AGENT                            │
│              (Decision Fusion & Adaptive Learning)              │
│                                                                 │
│  Role: Fuse scores, make final decision, learn from feedback    │
│  Trigger: Receives Evaluation Agent output                      │
│  Output: Final decision (ALLOW/CHALLENGE/DENY) + explanation    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ SUB-AGENTS (3) - SEQUENTIAL EXECUTION:                    │ │
│  │                                                           │ │
│  │  1. Fusion Sub-Agent                                      │ │
│  │     • Input: Behavioral score + Policy score              │ │
│  │     • Process: Weighted combination (adaptive)            │ │
│  │     • Override: Regulatory check (≥0.9 auto-deny)         │ │
│  │     • Output: Fused risk score (0-1) + confidence         │ │
│  │     • Weights: w_behavioral=0.6, w_policy=0.4 (adaptive)  │ │
│  │                                                           │ │
│  │  2. Decision Sub-Agent                                    │ │
│  │     • Input: Fused score                                  │ │
│  │     • Process: Three-tier threshold logic                 │ │
│  │       - F < 0.4 → ALLOW (seamless)                        │ │
│  │       - 0.4 ≤ F < 0.7 → CHALLENGE (verify)                │ │
│  │       - F ≥ 0.7 → DENY (block)                            │ │
│  │     • Output: Decision + reason                           │ │
│  │     • Thresholds: Adaptive via feedback                   │ │
│  │                                                           │ │
│  │  3. Learning Sub-Agent                                    │ │
│  │     • Process: Log decision to database (async)           │ │
│  │     • Storage: Full evidence + parameters used            │ │
│  │     • Feedback: Accept ground truth (fraud/legitimate)    │ │
│  │     • Adaptation: Update weights/thresholds on errors     │ │
│  │     • Metrics: Track TP/TN/FP/FN, Precision, Recall, F1   │ │
│  │     • Output: Confirmation + metrics                      │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. AGENT RELATIONSHIP MATRIX

### 2.1 Inter-Agent Communication

```
┌──────────────────────────────────────────────────────────────────┐
│                  AGENT COMMUNICATION FLOW                        │
└──────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Raw Transaction│
                    │  (CSV/API)      │
                    └────────┬────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │        MONITOR AGENT                   │
        │  process_transaction(raw_txn)          │
        └────────────────┬───────────────────────┘
                         │
                         │ Output: MonitorAgentOutput
                         │ {
                         │   enriched_transaction: T',
                         │   user_context: C,
                         │   extracted_features: F,
                         │   embedding: E ∈ ℝ⁷⁶⁸,
                         │   initial_risk: R₀
                         │ }
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │      EVALUATION AGENT                  │
        │  process(monitor_output)               │
        └────────────────┬───────────────────────┘
                         │
                         │ Output: EvaluationAgentOutput
                         │ {
                         │   behavioral_assessment: {
                         │     anomaly_score: A,
                         │     confidence: c_b,
                         │     explanation: E_b,
                         │     similar_transactions: S,
                         │     deviation_factors: D
                         │   },
                         │   policy_assessment: {
                         │     policy_score: P,
                         │     confidence: c_p,
                         │     explanation: E_p,
                         │     violations: V,
                         │     org_score: P_o,
                         │     reg_score: P_r
                         │   }
                         │ }
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     COORDINATOR AGENT                  │
        │  process(evaluation_output)            │
        └────────────────┬───────────────────────┘
                         │
                         │ Output: CoordinatorAgentOutput
                         │ {
                         │   decision: D ∈ {ALLOW, CHALLENGE, DENY},
                         │   fused_score: F,
                         │   confidence: C,
                         │   explanation: E,
                         │   evidence: Ev,
                         │   processing_time_ms: t
                         │ }
                         │
                         ▼
                    ┌─────────────────┐
                    │  User Response  │
                    │  (Allow/Block)  │
                    └─────────────────┘
                         │
                         │ (Later)
                         ▼
                    ┌─────────────────┐
                    │  Feedback Loop  │
                    │  Ground Truth   │
                    └────────┬────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  COORDINATOR AGENT                     │
        │  submit_feedback(txn_id, outcome)      │
        │  → Learning Sub-Agent                  │
        │  → Update adaptive parameters          │
        └────────────────────────────────────────┘
```

---

## 3. SUB-AGENT INTERACTION PATTERNS

### 3.1 Monitor Agent: Sequential Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│         MONITOR AGENT: SEQUENTIAL SUB-AGENT EXECUTION            │
└──────────────────────────────────────────────────────────────────┘

Input: raw_transaction
  │
  ├─→ Capture Sub-Agent (Step 1)
  │   • Normalize data
  │   • Generate ID
  │   • Extract temporal features
  │   Output: normalized_transaction
  │
  ├─→ Context Sub-Agent (Step 2)
  │   • Query user profile (DB)
  │   • Retrieve transaction history (DB)
  │   • Check vector store
  │   Input: user_id from Step 1
  │   Output: user_context
  │
  └─→ Feature Sub-Agent (Step 3)
      • Calculate statistics
      • Analyze patterns
      • Generate embedding
      Input: normalized_transaction + user_context
      Output: features + embedding

PATTERN: Sequential (step-by-step)
REASON: Each step depends on previous output
TIMING: ~50ms total (10ms + 20ms + 20ms)
```

### 3.2 Evaluation Agent: Parallel Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│        EVALUATION AGENT: PARALLEL SUB-AGENT EXECUTION            │
└──────────────────────────────────────────────────────────────────┘

Input: monitor_output
  │
  ├────────────────────────┬────────────────────────┐
  │                        │                        │
  ▼                        ▼                        │
Behavioral                Policy                    │
Sub-Agent                 Sub-Agent                 │
(Thread 1)                (Thread 2)                │
  │                        │                        │
  │ • Query user txns      │ • Query org policies   │
  │ • K=5 neighbors        │ • K=3 chunks           │
  │ • Statistical calc     │ • Query reg policies   │
  │ • LLM analysis         │ • K=3 chunks           │
  │ • Hybrid scoring       │ • LLM compliance       │
  │                        │ • Regulatory override  │
  │                        │                        │
  │ (150ms)                │ (200ms)                │
  │                        │                        │
  └────────────┬───────────┘                        │
               │                                    │
               │ asyncio.gather() waits for both    │
               │                                    │
               ▼                                    │
        Output: {behavioral_assessment,             │
                 policy_assessment}                 │

PATTERN: Parallel (simultaneous execution)
REASON: Independent RAG pipelines, no dependencies
TIMING: ~200ms (max of both, not sum)
OPTIMIZATION: 43% faster than sequential (350ms → 200ms)
```

### 3.3 Coordinator Agent: Sequential Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│       COORDINATOR AGENT: SEQUENTIAL SUB-AGENT EXECUTION          │
└──────────────────────────────────────────────────────────────────┘

Input: evaluation_output
  │
  ├─→ Fusion Sub-Agent (Step 1)
  │   • Check regulatory override
  │   • Weighted score combination
  │   • Calculate confidence
  │   Input: behavioral_score + policy_score
  │   Output: fused_score (5ms)
  │
  ├─→ Decision Sub-Agent (Step 2)
  │   • Apply threshold logic
  │   • Determine ALLOW/CHALLENGE/DENY
  │   Input: fused_score
  │   Output: decision + reason (5ms)
  │
  └─→ Learning Sub-Agent (Step 3)
      • Log decision to DB (async, non-blocking)
      • Store evidence
      • Prepare for feedback
      Input: complete decision package
      Output: logged confirmation (10ms, async)

PATTERN: Sequential (step-by-step)
REASON: Logical dependency (fusion → decision → logging)
TIMING: ~50ms total (5ms + 5ms + 30ms LLM explanation + 10ms async)
```

---

## 4. DATA DEPENDENCIES & FLOW

### 4.1 Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    END-TO-END DATA FLOW                          │
└──────────────────────────────────────────────────────────────────┘

RAW TRANSACTION
{
  user_id: "U123",
  amt: 500.0,
  merchant: "Amazon",
  city: "Seattle",
  state: "WA",
  trans_date_trans_time: "2026-02-12T10:30:00"
}
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ MONITOR AGENT                                                 │
│                                                               │
│ Capture → Context → Feature                                  │
│                                                               │
│ Adds:                                                         │
│ • transaction_id: "txn_abc123"                                │
│ • temporal_features: {hour: 10, day: 2, is_weekend: false}   │
│ • user_context: {avg_amt: 250, max_amt: 800, history: 45}    │
│ • extracted_features: {amount_zscore: 1.2, is_new_city: F}   │
│ • embedding: [0.123, -0.456, ..., 0.789] (768 dims)          │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
ENRICHED TRANSACTION
{
  ...original_fields,
  transaction_id: "txn_abc123",
  temporal_features: {...},
  user_context: {
    profile: {...},
    has_history: true,
    transaction_count: 45
  },
  extracted_features: {
    amount: {zscore: 1.2, ratio_to_avg: 2.0},
    geographic: {is_new_city: false, is_new_state: false},
    merchant: {is_new_merchant: false},
    velocity: {score: 0.3}
  },
  embedding: [768-dim vector],
  initial_risk: 0.4
}
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ EVALUATION AGENT                                              │
│                                                               │
│ Behavioral (parallel) || Policy (parallel)                    │
│                                                               │
│ Behavioral adds:                                              │
│ • Query user vector DB → 5 similar transactions               │
│ • Calculate statistical anomalies                             │
│ • LLM contextual analysis                                     │
│ • Output: {anomaly_score: 0.45, confidence: 0.75, ...}        │
│                                                               │
│ Policy adds:                                                  │
│ • Query org & reg policy DBs → 6 policy chunks                │
│ • LLM compliance validation                                   │
│ • Output: {policy_score: 0.3, org: 0.2, reg: 0.1, ...}       │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
EVALUATION ASSESSMENTS
{
  behavioral_assessment: {
    anomaly_score: 0.45,
    confidence: 0.75,
    explanation: "Amount slightly higher than typical...",
    similar_transactions: [
      {similarity: 0.85, amount: 450, merchant: "Amazon"},
      {similarity: 0.82, amount: 520, merchant: "Walmart"},
      ...
    ],
    deviation_factors: ["amount_zscore: 1.2"],
    statistical_analysis: {...}
  },
  policy_assessment: {
    policy_score: 0.3,
    confidence: 0.8,
    explanation: "Complies with org policy. No reg violations.",
    organizational_score: 0.2,
    regulatory_score: 0.1,
    violations: [],
    retrieved_policies: [
      {source: "CompanyPolicy.pdf", excerpt: "Transactions under $1000..."},
      ...
    ]
  }
}
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ COORDINATOR AGENT                                             │
│                                                               │
│ Fusion → Decision → Learning                                  │
│                                                               │
│ Fusion:                                                       │
│ • F = (0.45 × 0.6) + (0.3 × 0.4) = 0.27 + 0.12 = 0.39        │
│                                                               │
│ Decision:                                                     │
│ • F = 0.39 < threshold_low (0.4)                              │
│ • Decision = ALLOW                                            │
│                                                               │
│ Learning:                                                     │
│ • Log to DecisionLog table                                    │
│ • Store full evidence                                         │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
FINAL OUTPUT
{
  success: true,
  decision: "ALLOW",
  fused_score: 0.39,
  confidence: 0.76,
  behavioral_score: 0.45,
  policy_score: 0.3,
  explanation: "Transaction approved. Amount is within normal range 
                for this user (risk: 0.39). Similar past transactions 
                show consistent spending pattern. No policy violations 
                detected.",
  evidence: {
    behavioral_rag: {
      similar_transactions: [5 transactions],
      deviations: ["amount_zscore: 1.2"]
    },
    policy_rag: {
      retrieved_policies: [6 policy chunks],
      violations: []
    }
  },
  transaction_id: "txn_abc123",
  processing_time_ms: 287,
  weights_used: {behavioral_weight: 0.6, policy_weight: 0.4},
  thresholds_used: {threshold_low: 0.4, threshold_high: 0.7}
}
```

---

## 5. SHARED RESOURCES & DEPENDENCIES

### 5.1 Database Dependencies

```
┌──────────────────────────────────────────────────────────────────┐
│                  DATABASE INTERACTIONS BY AGENT                  │
└──────────────────────────────────────────────────────────────────┘

POSTGRESQL DATABASE:
════════════════════

Tables Used by Each Agent:

MONITOR AGENT:
├─ UserProfile (READ)
│  └─ Context Sub-Agent: Load user baseline
│
├─ Transaction (READ)
│  └─ Context Sub-Agent: Get transaction history
│
└─ Transaction (WRITE)
   └─ Feature Sub-Agent: Store new transaction

EVALUATION AGENT:
└─ (None - uses Vector Store only)

COORDINATOR AGENT:
├─ DecisionLog (WRITE)
│  └─ Learning Sub-Agent: Log decision
│
├─ Feedback (READ/WRITE)
│  └─ Learning Sub-Agent: Store & query feedback
│
├─ AdaptiveParameters (READ/WRITE)
│  ├─ Fusion Sub-Agent: Load weights
│  ├─ Decision Sub-Agent: Load thresholds
│  └─ Learning Sub-Agent: Update on feedback
│
└─ SystemMetrics (WRITE)
   └─ Learning Sub-Agent: Store performance metrics
```

### 5.2 Vector Store Dependencies

```
┌──────────────────────────────────────────────────────────────────┐
│                VECTOR STORE INTERACTIONS BY AGENT                │
└──────────────────────────────────────────────────────────────────┘

CHROMADB VECTOR STORE:
══════════════════════

Collections Used by Each Agent:

MONITOR AGENT:
└─ user_transactions_{user_id} (READ/WRITE)
   └─ Feature Sub-Agent: Store new transaction embedding

EVALUATION AGENT:
├─ user_transactions_{user_id} (READ)
│  └─ Behavioral Sub-Agent: Query K=5 similar transactions
│
├─ policies_organizational (READ)
│  └─ Policy Sub-Agent: Query K=3 org policy chunks
│
└─ policies_regulatory (READ)
   └─ Policy Sub-Agent: Query K=3 reg policy chunks

COORDINATOR AGENT:
└─ (None - no direct vector store access)
```

### 5.3 LLM Service Dependencies

```
┌──────────────────────────────────────────────────────────────────┐
│                    LLM API CALLS BY AGENT                        │
└──────────────────────────────────────────────────────────────────┘

OPENAI API (GPT-4):
═══════════════════

MONITOR AGENT:
└─ (None - no LLM calls)

EVALUATION AGENT:
├─ Behavioral Sub-Agent:
│  └─ llm_client.analyze_behavioral_anomaly_async()
│     • Input: Transaction + similar txns + baseline
│     • Output: Anomaly score + explanation
│     • Model: GPT-4
│     • Tokens: ~1500 avg
│
└─ Policy Sub-Agent:
   ├─ llm_client.analyze_policy_compliance_async(org)
   │  • Input: Transaction + org policy chunks
   │  • Output: Compliance score + violations
   │  • Model: GPT-4
   │  • Tokens: ~1200 avg
   │
   └─ llm_client.analyze_policy_compliance_async(reg)
      • Input: Transaction + reg policy chunks
      • Output: Compliance score + violations
      • Model: GPT-4
      • Tokens: ~1200 avg

COORDINATOR AGENT:
└─ Fusion Sub-Agent: (None)
└─ Decision Sub-Agent: (None)
└─ Learning Sub-Agent: (None)
└─ Main Agent:
   └─ llm_client.generate_decision_explanation()
      • Input: Transaction + assessments + decision
      • Output: Human-readable explanation
      • Model: GPT-4
      • Tokens: ~800 avg

TOTAL LLM CALLS PER TRANSACTION: 4
TOTAL TOKENS: ~4700 avg
PARALLELIZATION: 3 calls in parallel (Eval Agent), 1 sequential (Coord)
```

---

## 6. EXECUTION TIMING & PARALLELIZATION

### 6.1 Complete Timing Breakdown

```
┌──────────────────────────────────────────────────────────────────┐
│               END-TO-END EXECUTION TIMELINE                      │
└──────────────────────────────────────────────────────────────────┘

TIME (ms)  │ AGENT / SUB-AGENT                        │ TYPE
───────────┼──────────────────────────────────────────┼──────────
0          │ START                                    │
           │                                          │
0-10       │ Monitor: Capture Sub-Agent               │ Sequential
           │   • Data normalization                   │
           │   • ID generation                        │
           │                                          │
10-30      │ Monitor: Context Sub-Agent               │ Sequential
           │   • DB query (user profile)              │
           │   • Vector store check                   │
           │                                          │
30-50      │ Monitor: Feature Sub-Agent               │ Sequential
           │   • Statistical calculations             │
           │   • Embedding generation                 │
           │                                          │
50         │ Monitor Output Complete                  │
           │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
50-200     │ Evaluation: Behavioral Sub-Agent         │ ║ Parallel
           │   • Vector search (50ms)                 │ ║
           │   • Statistical analysis (30ms)          │ ║
           │   • LLM call (70ms)                      │ ║
           │                                          │ ║
50-250     │ Evaluation: Policy Sub-Agent             │ ║ Parallel
           │   • Vector search org (40ms)             │ ║
           │   • Vector search reg (40ms)             │ ║
           │   • LLM call org (60ms)                  │ ║
           │   • LLM call reg (60ms)                  │ ║
           │   • Score fusion (10ms)                  │ ║
           │                                          │
250        │ Evaluation Output Complete               │
           │   (max of 200ms and 250ms = 250ms)       │
           │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
250-255    │ Coordinator: Fusion Sub-Agent            │ Sequential
           │   • Weighted combination                 │
           │   • Regulatory override check            │
           │                                          │
255-260    │ Coordinator: Decision Sub-Agent          │ Sequential
           │   • Threshold logic                      │
           │   • Decision determination               │
           │                                          │
260-290    │ Coordinator: LLM Explanation             │ Sequential
           │   • Generate human-readable text         │
           │                                          │
290-300    │ Coordinator: Learning Sub-Agent          │ Async
           │   • Log to database (non-blocking)       │
           │                                          │
300        │ OUTPUT RETURNED TO USER                  │
           │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
300+       │ Learning: Async logging completes        │ Background
           │   • Write to DecisionLog                 │
           │   • Store evidence                       │
───────────┴──────────────────────────────────────────┴──────────

TOTAL LATENCY: ~300ms
CRITICAL PATH: Monitor (50ms) → Eval-Policy (200ms) → Coord (50ms)
PARALLELIZATION BENEFIT: 350ms → 300ms (14% improvement)
```

---

## 7. AGENT AUTONOMY & INVOCATION MODES

### 7.1 Invocation Patterns

```
┌──────────────────────────────────────────────────────────────────┐
│                    AGENT INVOCATION MODES                        │
└──────────────────────────────────────────────────────────────────┘

MONITOR AGENT:
══════════════

Mode 1: BACKGROUND (Autonomous)
├─ Trigger: File Watcher (every 30 seconds)
├─ Scans: data/uploads/transactions/*.csv
├─ Action: Auto-process new files
├─ Continues: To Evaluation & Coordinator automatically
└─ Use Case: Batch processing of uploaded transactions

Mode 2: ON-DEMAND (API-triggered)
├─ Trigger: API endpoint /api/process_transaction
├─ Input: Single transaction JSON
├─ Action: Process immediately
├─ Continues: To Evaluation & Coordinator automatically
└─ Use Case: Real-time transaction processing


EVALUATION AGENT:
═════════════════

Mode 1: PIPELINE (Automatic)
├─ Trigger: Receives Monitor Agent output
├─ Action: Dual RAG analysis (behavioral + policy)
├─ Continues: To Coordinator automatically
└─ Use Case: Normal transaction flow

Mode 2: STANDALONE (Manual)
├─ Trigger: API endpoint /api/evaluate
├─ Input: Enriched transaction
├─ Action: Analysis only, no decision
├─ Continues: Returns analysis, stops
└─ Use Case: Testing, debugging, analysis


COORDINATOR AGENT:
══════════════════

Mode 1: PIPELINE (Automatic)
├─ Trigger: Receives Evaluation Agent output
├─ Action: Fusion → Decision → Logging
├─ Continues: Returns to user/API
└─ Use Case: Normal transaction flow

Mode 2: FEEDBACK (Manual)
├─ Trigger: API endpoint /api/feedback
├─ Input: transaction_id + actual_outcome
├─ Action: Process feedback, update parameters
├─ Continues: Reloads adaptive parameters
└─ Use Case: Continuous learning
```

---

## 8. CROSS-CUTTING CONCERNS

### 8.1 Error Handling Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING STRATEGY                       │
└──────────────────────────────────────────────────────────────────┘

MONITOR AGENT:
├─ Capture Sub-Agent Error:
│  └─ Fallback: Skip transaction, log error, continue
│
├─ Context Sub-Agent Error (DB down):
│  └─ Fallback: Assume no history, continue with defaults
│
└─ Feature Sub-Agent Error:
   └─ Fallback: Use basic features, generate zero embedding

EVALUATION AGENT:
├─ Behavioral Sub-Agent Error:
│  └─ Fallback: Default anomaly_score=0.5, confidence=0.3
│
└─ Policy Sub-Agent Error:
   └─ Fallback: Default policy_score=0.0 (assume compliant)

COORDINATOR AGENT:
├─ Fusion Sub-Agent Error:
│  └─ Fallback: Use simple average
│
├─ Decision Sub-Agent Error:
│  └─ Fallback: CHALLENGE (safe default)
│
└─ Learning Sub-Agent Error:
   └─ Fallback: Log to file, continue without DB logging

PHILOSOPHY: Graceful degradation - system continues with reduced 
            capabilities rather than complete failure
```

### 8.2 Logging & Observability

```
┌──────────────────────────────────────────────────────────────────┐
│                  LOGGING INSTRUMENTATION                         │
└──────────────────────────────────────────────────────────────────┘

ALL AGENTS LOG:
├─ Initialization
├─ Sub-agent execution start/end
├─ Errors and exceptions
├─ Performance metrics (timing)
└─ Input/output shapes

MONITOR AGENT LOGS:
├─ File watcher events (scan start/end)
├─ New transactions detected
├─ Normalization warnings
├─ Missing user profiles
└─ Vector store writes

EVALUATION AGENT LOGS:
├─ RAG query results (K neighbors found)
├─ Vector similarity scores
├─ LLM call timing
├─ Anomaly score calculations
└─ Policy violations detected

COORDINATOR AGENT LOGS:
├─ Fusion calculations (weights used)
├─ Decision logic path taken
├─ Feedback received
├─ Parameter updates
└─ System metrics calculations

CENTRALIZED LOGGING:
All logs aggregated with:
├─ Timestamp
├─ Agent/Sub-agent name
├─ Transaction ID (correlation)
├─ Log level (DEBUG/INFO/WARN/ERROR)
└─ Structured JSON format
```

---

## 9. SCALABILITY & DEPLOYMENT

### 9.1 Horizontal Scaling Strategy

```
┌──────────────────────────────────────────────────────────────────┐
│                    SCALABILITY ARCHITECTURE                      │
└──────────────────────────────────────────────────────────────────┘

MONITOR AGENT:
├─ Scaling: Multiple instances for file watching
├─ Coordination: Distributed file locks
├─ State: Stateless (reads from DB/Vector store)
└─ Bottleneck: DB queries (Context Sub-Agent)

EVALUATION AGENT:
├─ Scaling: Infinite horizontal scaling
├─ Coordination: None needed (stateless)
├─ State: Stateless (RAG pipelines independent)
├─ Bottleneck: LLM API rate limits
└─ Optimization: Batch LLM calls, cache similar queries

COORDINATOR AGENT:
├─ Scaling: Horizontal with shared DB
├─ Coordination: DB transactions for feedback
├─ State: Adaptive parameters in DB (shared)
├─ Bottleneck: Feedback processing (serialized)
└─ Optimization: Eventual consistency for parameter updates

DEPLOYMENT MODEL:
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Monitor     │  │  Monitor     │  │  Monitor     │        │
│  │  Instance 1  │  │  Instance 2  │  │  Instance 3  │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐        │
│  │ Evaluation   │  │ Evaluation   │  │ Evaluation   │        │
│  │ Instance 1   │  │ Instance 2   │  │ Instance 3   │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│  ┌──────────────┐  ┌──────┴───────┐  ┌──────────────┐        │
│  │ Coordinator  │  │ Coordinator  │  │ Coordinator  │        │
│  │ Instance 1   │  │ Instance 2   │  │ Instance 3   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                │
│  ─────────────────────────────────────────────────────────────│
│                                                                │
│  ┌────────────────────────────────────────────────────────┐   │
│  │            SHARED INFRASTRUCTURE                       │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ PostgreSQL │  │  ChromaDB    │  │  OpenAI API  │   │   │
│  │  │  Database  │  │ Vector Store │  │  (LLM)       │   │   │
│  │  └────────────┘  └──────────────┘  └──────────────┘   │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

---

## 10. KEY ARCHITECTURAL PRINCIPLES

### 10.1 Design Patterns

```
┌──────────────────────────────────────────────────────────────────┐
│                  ARCHITECTURAL PATTERNS USED                     │
└──────────────────────────────────────────────────────────────────┘

1. AGENT PATTERN
   ═════════════
   • Autonomous agents with clear responsibilities
   • Each agent has distinct purpose (perception, analysis, decision)
   • Can operate independently or in pipeline

2. SUB-AGENT PATTERN (Strategy Pattern)
   ════════════════════════════════════
   • Complex tasks decomposed into specialized sub-agents
   • Each sub-agent focuses on single responsibility
   • Easy to test, maintain, and extend

3. PIPELINE PATTERN
   ════════════════
   • Sequential data transformation
   • Each stage enriches data for next stage
   • Clear data contracts between stages

4. PARALLEL PROCESSING (Fork-Join)
   ═══════════════════════════════
   • Evaluation Agent forks into 2 parallel RAG pipelines
   • Join results after both complete
   • Optimizes I/O-bound operations

5. RAG PATTERN (Retrieval-Augmented Generation)
   ════════════════════════════════════════════
   • Query vector store for relevant context
   • Augment LLM prompt with retrieved data
   • Generate contextual, grounded responses

6. ADAPTIVE LEARNING (Reinforcement Learning)
   ══════════════════════════════════════════
   • Feedback loop updates system parameters
   • Reward/penalty signals guide optimization
   • Continuous improvement over time

7. GRACEFUL DEGRADATION
   ════════════════════
   • Fallback mechanisms at every stage
   • System continues with reduced capabilities
   • Never complete failure on single error

8. SEPARATION OF CONCERNS
   ════════════════════════
   • Perception (Monitor) separate from Analysis (Evaluation)
   • Analysis separate from Decision (Coordinator)
   • Each agent has single, well-defined purpose
```

---

## 11. SUMMARY: AGENT RELATIONSHIPS

### 11.1 Relationship Matrix

```
┌──────────────────────────────────────────────────────────────────┐
│              AGENT-TO-AGENT RELATIONSHIP SUMMARY                 │
└──────────────────────────────────────────────────────────────────┘

                    Monitor  │  Evaluation  │  Coordinator
────────────────────────────────────────────────────────────────────
Monitor         │      -     │   Produces   │      -
                │            │  input for   │
────────────────────────────────────────────────────────────────────
Evaluation      │  Consumes  │      -       │   Produces
                │  output    │              │  input for
────────────────────────────────────────────────────────────────────
Coordinator     │      -     │  Consumes    │      -
                │            │  output      │
────────────────────────────────────────────────────────────────────

Feedback Loop   │      -     │      -       │  Self-update
(Later)         │            │              │  (Learning)
────────────────────────────────────────────────────────────────────


SUB-AGENT RELATIONSHIPS:
════════════════════════

MONITOR (Sequential Dependencies):
  Capture → Context → Feature
  (Each depends on previous)

EVALUATION (Independent Parallel):
  Behavioral || Policy
  (No dependencies, run simultaneously)

COORDINATOR (Sequential Dependencies):
  Fusion → Decision → Learning
  (Each depends on previous)


EXTERNAL DEPENDENCIES:
══════════════════════

All Agents:
├─ PostgreSQL Database (shared)
├─ ChromaDB Vector Store (shared)
├─ OpenAI LLM API (shared, rate-limited)
└─ Configuration Settings (shared)

Monitor:
├─ File system (CSV files)
└─ User profiles (DB)

Evaluation:
├─ Transaction vectors (Vector DB)
└─ Policy vectors (Vector DB)

Coordinator:
├─ Adaptive parameters (DB)
└─ Decision logs (DB)
```

---

## 12. ACADEMIC CONTRIBUTION SUMMARY

```
┌──────────────────────────────────────────────────────────────────┐
│           NOVEL ARCHITECTURAL CONTRIBUTIONS                      │
└──────────────────────────────────────────────────────────────────┘

1. HYBRID PARALLEL-SEQUENTIAL ARCHITECTURE
   ═══════════════════════════════════════
   • Monitor: Sequential (dependency chain)
   • Evaluation: Parallel (independent RAG pipelines)
   • Coordinator: Sequential (logical flow)
   • Optimization: Match pattern to data dependencies

2. DUAL RAG PIPELINES WITH DIFFERENT OBJECTIVES
   ════════════════════════════════════════════
   • Behavioral RAG: Intra-user anomaly detection
   • Policy RAG: Compliance validation
   • Independent vector stores, models, and queries
   • Enables domain-specific optimization

3. ADAPTIVE MULTI-AGENT SYSTEM
   ════════════════════════════
   • Parameters evolve based on feedback
   • Weights and thresholds self-optimize
   • Continuous learning without retraining models
   • Fast adaptation to concept drift

4. REGULATORY PRECEDENCE MECHANISM
   ═══════════════════════════════
   • Hard overrides ensure legal compliance
   • Separate scoring for org vs regulatory policies
   • Weighted fusion with regulatory boost
   • Auditable decision trail

5. EXPLAINABLE AI THROUGH RAG CITATIONS
   ═══════════════════════════════════
   • Every decision backed by retrieved evidence
   • Similar transaction citations
   • Policy document excerpts
   • Natural language explanations
   • Full transparency for stakeholders

6. THREE-TIER DECISION FRAMEWORK
   ═══════════════════════════════
   • ALLOW: Seamless (low friction)
   • CHALLENGE: Verify (balanced)
   • DENY: Block (high security)
   • Optimizes user experience vs security tradeoff
```

---

## END OF DOCUMENTATION

This document provides a complete view of the Guardian System's multi-agent architecture, showing how all agents and sub-agents relate to each other, their dependencies, execution patterns, and design principles.
