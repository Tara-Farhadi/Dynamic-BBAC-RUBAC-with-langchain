# Monitor Agent - Academic Flowchart Documentation

## For Academic Paper Publication

---

## 1. HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        MONITOR AGENT                             │
│                    (Perception Layer)                            │
└─────────────────────────────────────────────────────────────────┘

INPUT: Raw Transaction T = {user_id, amt, merchant, city, state, ...}
                              ↓
        ┌─────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
┌───────────────────┐              ┌──────────────────────────────┐
│  BACKGROUND       │              │   ON-DEMAND PROCESSING        │
│  MONITORING       │              │   (Per Transaction Request)   │
│  (Continuous)     │              │                               │
└───────────────────┘              └──────────────────────────────┘
        │                                          │
        │                          ┌───────────────┴───────────────┐
        │                          │                               │
        ▼                          ▼                               ▼
┌───────────────┐      ┌──────────────────┐          ┌──────────────────┐
│   File        │      │   Capture        │          │   Context        │
│   Watcher     │      │   Sub-Agent      │          │   Sub-Agent      │
│               │      │                  │          │                  │
│ • Scan every  │      │ • Normalize data │          │ • Query user     │
│   30 seconds  │      │ • Generate ID    │          │   profile        │
│ • Detect new  │      │ • Extract time   │          │ • Get history    │
│   CSV files   │      │   features       │          │ • Check vector   │
│ • Load & index│      │                  │          │   store          │
└───────┬───────┘      └────────┬─────────┘          └────────┬─────────┘
        │                       │                              │
        │                       └──────────┬───────────────────┘
        │                                  │
        ▼                                  ▼
┌───────────────┐              ┌──────────────────┐
│   Load CSV    │              │   Feature        │
│   File        │              │   Sub-Agent      │
│               │              │                  │
│ • Parse CSV   │              │ • Amount stats   │
│ • Store in DB │              │ • Location flags │
│ • Generate    │              │ • Merchant flags │
│   embeddings  │              │ • Velocity calc  │
│ • Create      │              │ • Generate       │
│   profile     │              │   embedding      │
└───────┬───────┘              └────────┬─────────┘
        │                               │
        └───────────┬───────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   OUTPUT PACKAGE      │
        │                       │
        │ • Enriched Txn T'     │
        │ • User Context C      │
        │ • Features F          │
        │ • Embedding E ∈ ℝ⁷⁶⁸ │
        │ • Risk Score R₀       │
        └───────────┬───────────┘
                    │
                    ▼
        TO EVALUATION AGENT →
```

---

## 2. SUB-AGENT DETAILED FLOWCHARTS

### 2.1 Capture Sub-Agent

```
┌─────────────────────────────────────────────────────┐
│              CAPTURE SUB-AGENT                       │
│         (Data Normalization Module)                  │
└─────────────────────────────────────────────────────┘

START
  │
  ▼
┌──────────────────────────┐
│ Receive Raw Transaction  │
│ T = {user_id, amt, ...}  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Normalize Fields       │
│                          │
│ • amt: string → float    │
│ • merchant: lowercase    │
│ • state: uppercase       │
│ • Handle missing values  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Generate Transaction ID  │
│                          │
│ txn_id = hash(          │
│   user_id + timestamp   │
│ )                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Extract Temporal Features│
│                          │
│ • hour: 0-23             │
│ • day_of_week: 0-6       │
│ • is_weekend: bool       │
│ • period: morning/...    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Create Enriched Txn T'   │
│                          │
│ T' = T ∪ {              │
│   txn_id,                │
│   temporal_features,     │
│   timestamp              │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
          RETURN T'

OUTPUT:
  Enriched Transaction T'
  Complexity: O(1)
```

---

### 2.2 Context Sub-Agent

```
┌─────────────────────────────────────────────────────┐
│              CONTEXT SUB-AGENT                       │
│       (Historical Context Retrieval)                 │
└─────────────────────────────────────────────────────┘

START
  │
  ▼
┌──────────────────────────┐
│ Receive user_id          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Query User Profile     │
│   from Database          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Build Profile Object P   │
│                          │
│ P = {                    │
│   avg_amount: μ          │
│   std_amount: σ          │
│   max_amount: M          │
│   common_merchants: {m}  │
│   common_categories: {c} │
│   typical_hours: {h}     │
│   typical_cities: {l}    │
│   risk_level: r          │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Query Recent Transactions│
│                          │
│ H = {T₁, T₂, ..., Tₙ}   │
│                          │
│ Filters:                 │
│ • Last 90 days           │
│ • Limit: 100 txns        │
│ • Order: DESC by date    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Check Vector Store Count │
│                          │
│ count = |E_user|         │
│                          │
│ (Number of embeddings)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Build Context Object C   │
│                          │
│ C = {                    │
│   user_profile: P,       │
│   recent_txns: H,        │
│   vector_count: count,   │
│   has_history: bool      │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
          RETURN C

OUTPUT:
  User Context C
  Complexity: O(n) where n = |H|
```

---

### 2.3 Feature Sub-Agent

```
┌─────────────────────────────────────────────────────┐
│              FEATURE SUB-AGENT                       │
│     (Statistical Feature Extraction)                 │
└─────────────────────────────────────────────────────┘

START
  │
  ▼
┌──────────────────────────┐
│ Receive T' and C         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Extract Amount Features  │
│                          │
│ F_amt = {                │
│   z_score: (amt-μ)/σ     │
│   ratio_to_max: amt/M    │
│   ratio_to_avg: amt/μ    │
│   deviation: |amt-μ|     │
│   is_high: z > 2         │
│   is_low: z < -2         │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Extract Location Features│
│                          │
│ F_loc = {                │
│   is_new_city: c ∉ {l}   │
│   is_new_state: s ∉ {s}  │
│   is_international: bool │
│   distance_from_home: d  │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Extract Merchant Features│
│                          │
│ F_merch = {              │
│   is_new_merchant:       │
│     m ∉ {m}              │
│   is_new_category:       │
│     cat ∉ {c}            │
│   merchant_frequency: f  │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Calculate Velocity       │
│                          │
│ F_vel = {                │
│   time_since_last: Δt    │
│   txn_per_hour: count    │
│   is_rapid: Δt < 300s    │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Calculate Combined Risk  │
│                          │
│ R₀ = w₁·z_score +        │
│      w₂·loc_risk +       │
│      w₃·merch_risk +     │
│      w₄·vel_risk         │
│                          │
│ where Σwᵢ = 1            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Generate Embedding       │
│                          │
│ E = Encoder(T')          │
│                          │
│ Using sentence-          │
│ transformers:            │
│ E ∈ ℝ⁷⁶⁸                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Compile Feature Vector   │
│                          │
│ F = {                    │
│   temporal: F_time,      │
│   amount: F_amt,         │
│   location: F_loc,       │
│   merchant: F_merch,     │
│   velocity: F_vel,       │
│   combined: R₀           │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
      RETURN {F, E, R₀}

OUTPUT:
  Features F, Embedding E, Risk Score R₀
  Complexity: O(d) where d = embedding_dim
```

---

## 3. BACKGROUND MONITORING PROCESS

```
┌─────────────────────────────────────────────────────┐
│           BACKGROUND FILE WATCHER                    │
│           (Continuous Monitoring)                    │
└─────────────────────────────────────────────────────┘

START (System Initialization)
  │
  ▼
┌──────────────────────────┐
│ Create Uploads Directory │
│ if not exists            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Load All Existing        │
│ CSV Files                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Start Background Thread  │
│ (Daemon Mode)            │
└────────────┬─────────────┘
             │
             ▼
        ┌────────┐
        │ LOOP   │←─────────────┐
        └────┬───┘              │
             │                  │
             ▼                  │
┌──────────────────────────┐   │
│ Scan Directory           │   │
│ Get all *.csv files      │   │
└────────────┬─────────────┘   │
             │                  │
             ▼                  │
┌──────────────────────────┐   │
│ For Each File:           │   │
│                          │   │
│ 1. Calculate MD5 Hash    │   │
│ 2. Compare with Stored   │   │
└────────────┬─────────────┘   │
             │                  │
             ▼                  │
         ┌───────┐              │
         │ New   │              │
         │ File? │──No──┐       │
         └───┬───┘      │       │
             │          │       │
            Yes         ▼       │
             │    ┌───────────┐ │
             │    │ Changed   │ │
             │    │ File?     │─┼─No─┐
             │    └─────┬─────┘ │    │
             │          │        │    │
             │         Yes       │    │
             └──────────┴────────┘    │
                        │             │
                        ▼             │
           ┌──────────────────────┐  │
           │ Load CSV File        │  │
           │                      │  │
           │ 1. Parse Txns        │  │
           │ 2. Store in DB       │  │
           │ 3. Generate          │  │
           │    Embeddings        │  │
           │ 4. Add to Vector DB  │  │
           │ 5. Update Profile    │  │
           │ 6. Update Hash       │  │
           └──────────┬───────────┘  │
                      │              │
                      └──────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ Sleep 30 seconds     │
           └──────────┬───────────┘
                      │
                      └──────────────┘

CONTINUOUS PROCESS (Runs Forever)
```

---

## 4. MAIN PROCESSING PIPELINE

```
┌─────────────────────────────────────────────────────┐
│         TRANSACTION PROCESSING PIPELINE              │
│         (On-Demand Execution)                        │
└─────────────────────────────────────────────────────┘

START (API Request Received)
  │
  ▼
┌──────────────────────────┐
│ Receive Raw Transaction  │
│ from API Endpoint        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ STEP 1: CAPTURE          │
│                          │
│ Call Capture Sub-Agent   │
│                          │
│ Input: T                 │
│ Output: T'               │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Check Success            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ STEP 2: CONTEXT          │
│                          │
│ Call Context Sub-Agent   │
│                          │
│ Input: T'.user_id        │
│ Output: C                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ STEP 3: FEATURE          │
│                          │
│ Call Feature Sub-Agent   │
│                          │
│ Input: T', C             │
│ Output: F, E, R₀         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Check Success            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Package Output           │
│                          │
│ Output = {               │
│   success: true,         │
│   transaction_id: id,    │
│   user_id: uid,          │
│   enriched_txn: T',      │
│   user_context: C,       │
│   features: F,           │
│   embedding: E,          │
│   prelim_risk: R₀        │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Return to Main Pipeline  │
│                          │
│ → Send to Evaluation     │
│   Agent for RAG Analysis │
└──────────────────────────┘

END
```

---

## 5. FORMAL NOTATION FOR ACADEMIC PAPER

### 5.1 Function Definitions

**Monitor Agent Function:**
```
M: T → (T', C, F, E, R₀)

Where:
  T ∈ T_space    : Raw transaction
  T' ∈ T'_space  : Enriched transaction
  C ∈ C_space    : User context (profile + history)
  F ∈ ℝⁿ         : Feature vector
  E ∈ ℝ⁷⁶⁸       : Transaction embedding
  R₀ ∈ [0,1]     : Preliminary risk score
```

**Sub-Agent Functions:**
```
Capture Sub-Agent:  φ_c: T → T'
Context Sub-Agent:  φ_x: UserID → C
Feature Sub-Agent:  φ_f: (T', C) → (F, E, R₀)

Pipeline Composition:
  M(T) = φ_f(φ_c(T), φ_x(T.user_id))
```

### 5.2 Mathematical Formulations

**Feature Extraction:**
```
Z-Score:
  z = (amt - μ) / σ
  where μ = mean(historical_amounts)
        σ = std(historical_amounts)

Location Risk:
  loc_risk = α·I(city_new) + β·I(state_new) + γ·I(international)
  where I(·) is indicator function

Preliminary Risk Score:
  R₀ = Σᵢ₌₁ⁿ wᵢ·fᵢ
  where Σwᵢ = 1, fᵢ ∈ [0,1]
```

**Embedding Generation:**
```
E = Encoder(T')
where Encoder: T'_space → ℝ⁷⁶⁸
using sentence-transformers model
```

---

## 6. COMPONENT COMPLEXITY ANALYSIS

| Component | Input Size | Processing | Output Size | Time Complexity | Space Complexity |
|-----------|------------|------------|-------------|-----------------|------------------|
| **Capture Sub-Agent** | O(1) | Normalization | O(1) | O(1) | O(1) |
| **Context Sub-Agent** | O(1) | DB queries | O(n) | O(n + log m) | O(n) |
| **Feature Sub-Agent** | O(n) | Statistics + ML | O(d) | O(n + d) | O(d) |
| **File Watcher** | O(k) | I/O + Hashing | O(k·m) | O(k·m) | O(k·m) |
| **Overall Pipeline** | O(1) | Sequential | O(n+d) | O(n + d) | O(n + d) |

**Where:**
- n = number of historical transactions
- d = embedding dimension (768)
- k = number of CSV files
- m = average transactions per file

---

## 7. ALGORITHM PSEUDOCODE

### Algorithm 1: Monitor Agent Processing

```
ALGORITHM: MonitorAgent_Process
────────────────────────────────────────────────────
INPUT: T = (user_id, amt, merchant, city, state, ...)
OUTPUT: (T', C, F, E, R₀)

1:  procedure PROCESS(T)
2:    // Step 1: Capture and normalize
3:    T' ← CAPTURE_SUBAGENT(T)
4:      T'.normalized ← NORMALIZE_FIELDS(T)
5:      T'.txn_id ← GENERATE_ID(T.user_id, T.timestamp)
6:      T'.temporal ← EXTRACT_TEMPORAL(T.timestamp)
7:    
8:    // Step 2: Retrieve user context
9:    C ← CONTEXT_SUBAGENT(T'.user_id)
10:     C.profile ← DB.QUERY_PROFILE(T'.user_id)
11:     C.history ← DB.QUERY_HISTORY(T'.user_id, days=90)
12:     C.count ← VECTOR_STORE.COUNT(T'.user_id)
13:     C.has_history ← (C.count > 0)
14:   
15:   // Step 3: Extract features and embedding
16:   (F, E, R₀) ← FEATURE_SUBAGENT(T', C)
17:     F.amount ← COMPUTE_AMOUNT_FEATURES(T'.amt, C.profile)
18:     F.location ← COMPUTE_LOCATION_FEATURES(T'.city, C.history)
19:     F.merchant ← COMPUTE_MERCHANT_FEATURES(T'.merchant, C.profile)
20:     F.velocity ← COMPUTE_VELOCITY(T'.timestamp, C.history)
21:     E ← EMBEDDING_SERVICE.ENCODE(T')
22:     R₀ ← WEIGHTED_SUM(F)
23:   
24:   return (T', C, F, E, R₀)
25: end procedure

COMPLEXITY: O(n + d) where n = |C.history|, d = dim(E)
```

### Algorithm 2: Background Monitoring

```
ALGORITHM: BackgroundFileWatcher
────────────────────────────────────────────────────
INPUT: Directory path D, interval Δt
OUTPUT: Indexed transaction database

1:  procedure WATCH_FOLDER(D, Δt)
2:    indexed ← ∅
3:    while true do
4:      Files ← SCAN_DIRECTORY(D, "*.csv")
5:      
6:      for each file ∈ Files do
7:        if file.name = "README.csv" then
8:          continue
9:        end if
10:       
11:       hash_current ← MD5(file)
12:       
13:       if file ∉ indexed then
14:         // New file detected
15:         LOAD_AND_INDEX(file)
16:         indexed[file] ← hash_current
17:         
18:       else if indexed[file] ≠ hash_current then
19:         // File modified
20:         LOAD_AND_INDEX(file)
21:         indexed[file] ← hash_current
22:       end if
23:     end for
24:     
25:     SLEEP(Δt)  // Default: 30 seconds
26:   end while
27: end procedure
28:
29: procedure LOAD_AND_INDEX(file)
30:   Transactions ← PARSE_CSV(file)
31:   DB.STORE(Transactions)
32:   Embeddings ← GENERATE_EMBEDDINGS(Transactions)
33:   VECTOR_STORE.ADD(Embeddings)
34:   Profile ← CALCULATE_PROFILE(Transactions)
35:   DB.UPDATE_PROFILE(Profile)
36: end procedure

COMPLEXITY: O(k·m·d) per iteration
  where k = |Files|, m = avg txns per file, d = embedding dim
```

---

## 8. DATA STRUCTURES

### Transaction Object
```python
T = {
  user_id: string,
  amt: float,
  merchant: string,
  category: string,
  city: string,
  state: string,
  country: string,
  trans_date_trans_time: datetime,
  cc_num: string (hashed),
  zip: string,
  trans_num: string
}
```

### Enriched Transaction Object
```python
T' = T ∪ {
  transaction_id: string,
  temporal_features: {
    hour: int [0-23],
    day_of_week: int [0-6],
    is_weekend: bool,
    period: enum {morning, afternoon, evening, night}
  },
  capture_timestamp: datetime
}
```

### User Context Object
```python
C = {
  user_profile: {
    avg_amount: float,
    std_amount: float,
    max_amount: float,
    min_amount: float,
    common_merchants: list[string],
    common_categories: list[string],
    common_locations: list[{city, state, country}],
    typical_hours: list[int],
    risk_level: enum {low, medium, high}
  },
  recent_transactions: list[Transaction],
  vector_count: int,
  has_history: bool
}
```

### Feature Vector Object
```python
F = {
  temporal: {...},
  amount: {
    zscore: float,
    ratio_to_max: float,
    ratio_to_avg: float,
    deviation: float
  },
  location: {
    is_new_city: bool,
    is_new_state: bool,
    is_international: bool
  },
  merchant: {
    is_new_merchant: bool,
    is_new_category: bool
  },
  velocity: {
    seconds_since_last: int,
    txn_per_hour: float
  },
  combined: {
    preliminary_risk_score: float [0,1]
  }
}
```

### Embedding Vector
```python
E ∈ ℝ⁷⁶⁸  # Using sentence-transformers
```

---

## 9. SYSTEM METRICS

### Performance Characteristics
- **Latency per Transaction:** ~50-100ms
- **Throughput:** ~1000 transactions/second
- **Memory Footprint:** ~500MB (base) + O(n·d) for vectors
- **Disk I/O:** ~10MB/s (during bulk loading)

### Monitoring Metrics
- **File Scan Frequency:** 30 seconds
- **Average File Load Time:** 2-5 seconds per 1000 transactions
- **Embedding Generation:** ~10ms per transaction
- **Database Query Time:** ~5-10ms

---

## 10. REFERENCES FOR CITATION

**Key Technologies:**
1. Sentence Transformers: Reimers & Gurevych (2019)
2. Vector Similarity Search: Johnson et al. (2019) - FAISS
3. Transaction Fraud Detection: Carneiro et al. (2017)
4. Time-Series Feature Engineering: Christ et al. (2018)

**Relevant Papers:**
- Multi-agent systems for fraud detection
- RAG architectures for contextual analysis
- Behavioral biometrics in financial transactions
- Real-time anomaly detection in streaming data

---

## END OF DOCUMENT

**Document Purpose:** Academic flowchart and algorithm documentation for Monitor Agent component of the GUARDIAN transaction monitoring system.

**Usage:** This document can be used to create flowcharts in tools like:
- draw.io (diagrams.net)
- Lucidchart
- Microsoft Visio
- LaTeX TikZ
- PlantUML

**Format:** Markdown with ASCII art diagrams, mathematical notation, and pseudocode suitable for academic publications.
