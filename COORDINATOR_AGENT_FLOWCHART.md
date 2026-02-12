# Coordinator Agent - Academic Flowchart Documentation

## For Academic Paper Publication

---

## 1. HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COORDINATOR AGENT                                │
│         (Decision Fusion & Adaptive Learning Layer)                  │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Evaluation Agent Output = {Behavioral, Policy Assessments}
                              ↓
        ┌──────────────────────────────────────────────────────┐
        │    SEQUENTIAL PROCESSING PIPELINE                    │
        │    Three Sub-Agents in Series                        │
        └──────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  FUSION SUB-AGENT   │
                    │  (Score Combining)  │
                    │                     │
                    │  • Weighted blend   │
                    │  • Regulatory check │
                    │  • Confidence calc  │
                    │                     │
                    │  Adaptive Weights:  │
                    │  • w_b = 0.6        │
                    │  • w_p = 0.4        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ DECISION SUB-AGENT  │
                    │ (Threshold Logic)   │
                    │                     │
                    │  Rules:             │
                    │  • score < 0.4:     │
                    │    → ALLOW          │
                    │  • score >= 0.7:    │
                    │    → DENY           │
                    │  • 0.4-0.7:         │
                    │    → CHALLENGE      │
                    │                     │
                    │  Adaptive:          │
                    │  • threshold_low    │
                    │  • threshold_high   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ LEARNING SUB-AGENT  │
                    │ (Logging & Adapt)   │
                    │                     │
                    │  • Log decision     │
                    │  • Store evidence   │
                    │  • Await feedback   │
                    │  • Update params    │
                    │  • Track metrics    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FINAL OUTPUT      │
                    │                     │
                    │ • Decision: D       │
                    │ • Fused Score: F    │
                    │ • Confidence: C     │
                    │ • Explanation: E    │
                    │ • Evidence: Ev      │
                    └──────────┬──────────┘
                               │
                               ▼
                    TO USER / SYSTEM →
                               │
                               │ (Later)
                               ▼
                    ┌─────────────────────┐
                    │  FEEDBACK LOOP      │
                    │                     │
                    │  Ground truth from: │
                    │  • Human review     │
                    │  • Fraud detection  │
                    │  • Customer dispute │
                    │                     │
                    │  Updates:           │
                    │  • Weights (w_b,w_p)│
                    │  • Thresholds (θ)   │
                    └─────────────────────┘
```

---

## 2. FUSION SUB-AGENT DETAILED FLOWCHART

### 2.1 Score Fusion with Regulatory Override

```
┌─────────────────────────────────────────────────────────────────────┐
│              FUSION SUB-AGENT (Score Combination)                    │
│              Weighted Blending with Override Logic                   │
└─────────────────────────────────────────────────────────────────────┘

START: execute(input_data)
  │
  ▼
┌────────────────────────────────────┐
│ Extract Input Scores               │
│                                    │
│ behavioral_assessment = {          │
│   anomaly_score: A ∈ [0,1],        │
│   confidence: c_b ∈ [0,1],         │
│   explanation: E_b,                │
│   similar_transactions: S,         │
│   deviation_factors: D             │
│ }                                  │
│                                    │
│ policy_assessment = {              │
│   policy_score: P ∈ [0,1],         │
│   confidence: c_p ∈ [0,1],         │
│   organizational_score: P_o,       │
│   regulatory_score: P_r,           │
│   violations: V[],                 │
│   retrieved_policies: Pol[]        │
│ }                                  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Check Regulatory Override          │
│                                    │
│ IF P_r >= 0.9:                     │
│   CRITICAL REGULATORY VIOLATION    │
└────────┬───────────────────────┬───┘
         │                       │
    True │                       │ False
         │                       │
         ▼                       │
┌─────────────────────┐          │
│ REGULATORY OVERRIDE │          │
│                     │          │
│ Return:             │          │
│ {                   │          │
│   fused_score: P_r, │          │
│   confidence: 0.95, │          │
│   behavioral_contrib│          │
│     ution: 0.0,     │          │
│   policy_contrib    │          │
│     ution: P_r,     │          │
│   override_reason:  │          │
│     'regulatory_    │          │
│     violation'      │          │
│ }                   │          │
│                     │          │
│ SKIP TO OUTPUT →    │          │
└─────────────────────┘          │
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ Load Current Weights         │
                    │                              │
                    │ w_b = behavioral_weight      │
                    │ w_p = policy_weight          │
                    │                              │
                    │ Defaults:                    │
                    │ • w_b = 0.6 (60%)            │
                    │ • w_p = 0.4 (40%)            │
                    │                              │
                    │ Note: Weights are adaptive   │
                    │ and updated via feedback     │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ Normalize Weights            │
                    │                              │
                    │ total = w_b + w_p            │
                    │                              │
                    │ w_b_norm = w_b / total       │
                    │ w_p_norm = w_p / total       │
                    │                              │
                    │ Ensures: w_b_norm + w_p_norm │
                    │          = 1.0               │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ Calculate Fused Score        │
                    │                              │
                    │ F = (A × w_b_norm) +         │
                    │     (P × w_p_norm)           │
                    │                              │
                    │ Example:                     │
                    │ A = 0.7, P = 0.5             │
                    │ w_b_norm = 0.6               │
                    │ w_p_norm = 0.4               │
                    │                              │
                    │ F = (0.7 × 0.6) +            │
                    │     (0.5 × 0.4)              │
                    │   = 0.42 + 0.20              │
                    │   = 0.62                     │
                    │                              │
                    │ F = min(1.0, F)              │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ Calculate Fused Confidence   │
                    │                              │
                    │ C = (c_b × w_b_norm) +       │
                    │     (c_p × w_p_norm)         │
                    │                              │
                    │ Weighted by contribution of  │
                    │ each assessment              │
                    │                              │
                    │ C = min(1.0, C)              │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ Calculate Contributions      │
                    │                              │
                    │ behavioral_contribution =    │
                    │   A × w_b_norm               │
                    │                              │
                    │ policy_contribution =        │
                    │   P × w_p_norm               │
                    │                              │
                    │ Purpose: Show which factor   │
                    │ contributed more to final    │
                    │ risk score                   │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │ FUSION OUTPUT                │
                    │                              │
                    │ Return:                      │
                    │ {                            │
                    │   success: True,             │
                    │   fused_score: F,            │
                    │   confidence: C,             │
                    │   behavioral_contribution,   │
                    │   policy_contribution,       │
                    │   override_reason: None,     │
                    │   weights_used: {            │
                    │     behavioral_weight: w_b,  │
                    │     policy_weight: w_p       │
                    │   }                          │
                    │ }                            │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                        TO DECISION SUB-AGENT →
```

---

## 3. DECISION SUB-AGENT DETAILED FLOWCHART

### 3.1 Threshold-Based Decision Logic

```
┌─────────────────────────────────────────────────────────────────────┐
│            DECISION SUB-AGENT (Threshold Logic)                      │
│            Three-Tier Decision Framework                             │
└─────────────────────────────────────────────────────────────────────┘

START: execute(input_data)
  │
  ▼
┌────────────────────────────────────┐
│ Extract Input                      │
│                                    │
│ • fused_score: F ∈ [0, 1]          │
│ • confidence: C ∈ [0, 1]           │
│ • override_reason: OR              │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Load Current Thresholds            │
│                                    │
│ • θ_low (threshold_low)            │
│ • θ_high (threshold_high)          │
│                                    │
│ Defaults:                          │
│ • θ_low = 0.4                      │
│ • θ_high = 0.7                     │
│                                    │
│ Note: Adaptive via feedback        │
└────────────┬───────────────────────┘
             │
             ▼
        ┌─────────┐
        │ Override?│
        └────┬───┬─┘
             │   │
         Yes │   │ No
             │   │
             ▼   │
┌──────────────────────┐             │
│ REGULATORY OVERRIDE  │             │
│                      │             │
│ Decision: DENY       │             │
│                      │             │
│ Reason: "Regulatory  │             │
│ violation detected - │             │
│ automatic denial"    │             │
│                      │             │
│ Bypass threshold     │             │
│ checks entirely      │             │
└──────────┬───────────┘             │
           │                         │
           │                         │
           └─────────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  THRESHOLD DECISION  │
              │      TREE            │
              └──────────┬───────────┘
                         │
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ F < θ_low?   │  │ F >= θ_high? │  │ θ_low <= F   │
│              │  │              │  │  < θ_high?   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │ True            │ True            │ True
       ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   ALLOW      │  │     DENY     │  │  CHALLENGE   │
│              │  │              │  │              │
│ Decision: D  │  │ Decision: D  │  │ Decision: D  │
│   = "ALLOW"  │  │   = "DENY"   │  │   = "CHAL-   │
│              │  │              │  │    LENGE"    │
│              │  │              │  │              │
│ Reason:      │  │ Reason:      │  │ Reason:      │
│ "Risk {F}    │  │ "Risk {F}    │  │ "Risk {F}    │
│ below        │  │ exceeds      │  │ in challenge │
│ threshold    │  │ threshold    │  │ range        │
│ ({θ_low})"   │  │ ({θ_high})"  │  │ ({θ_low}-    │
│              │  │              │  │  {θ_high})"  │
│              │  │              │  │              │
│ Action:      │  │ Action:      │  │ Action:      │
│ • Approve    │  │ • Block txn  │  │ • Request    │
│ • No further │  │ • Alert user │  │   2FA        │
│   action     │  │ • Log alert  │  │ • SMS code   │
│              │  │              │  │ • Biometric  │
│              │  │              │  │ • Review     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
                ┌────────────────────┐
                │ DECISION OUTPUT    │
                │                    │
                │ Return:            │
                │ {                  │
                │   success: True,   │
                │   decision: D,     │
                │   decision_reason, │
                │   thresholds_used: │
                │   {                │
                │     threshold_low, │
                │     threshold_high │
                │   }                │
                │ }                  │
                └─────────┬──────────┘
                          │
                          ▼
                TO COORDINATOR PROCESS →


DECISION SEMANTICS:
═══════════════════

ALLOW:
• Transaction proceeds without intervention
• User experience: Seamless, no delay
• Risk: Low (F < 0.4)
• Monitoring: Passive logging

CHALLENGE:
• Transaction requires additional verification
• User experience: 2FA, SMS code, biometric scan
• Risk: Moderate (0.4 ≤ F < 0.7)
• Monitoring: Active verification required
• Outcome: User proves identity → ALLOW
           User fails → DENY

DENY:
• Transaction blocked immediately
• User experience: Declined, notification sent
• Risk: High (F ≥ 0.7)
• Monitoring: Alert sent to fraud team
• Outcome: Manual review or permanent block
```

---

## 4. LEARNING SUB-AGENT DETAILED FLOWCHART

### 4.1 Decision Logging Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│          LEARNING SUB-AGENT (Logging Component)                      │
│          Immediate Decision Recording                                │
└─────────────────────────────────────────────────────────────────────┘

START: execute(input_data)
  │
  ▼
┌────────────────────────────────────┐
│ Receive Decision Package           │
│                                    │
│ Contains:                          │
│ • transaction_id                   │
│ • decision (ALLOW/CHALLENGE/DENY)  │
│ • fused_score                      │
│ • confidence                       │
│ • behavioral_score                 │
│ • policy_score                     │
│ • explanation                      │
│ • evidence (RAG citations)         │
│ • monitor_output (features)        │
│ • evaluation_output (assessments)  │
│ • coordinator_output (fusion)      │
│ • weights_used (w_b, w_p)          │
│ • thresholds_used (θ_low, θ_high)  │
│ • processing_time_ms               │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Create DecisionLog Record          │
│                                    │
│ DecisionLog(                       │
│   transaction_id,                  │
│   decision,                        │
│   fused_score,                     │
│   confidence,                      │
│   behavioral_score,                │
│   policy_score,                    │
│   explanation,                     │
│   evidence,                        │
│   monitor_output,                  │
│   evaluation_output,               │
│   coordinator_output,              │
│   behavioral_weight,               │
│   policy_weight,                   │
│   processing_time_ms,              │
│   timestamp=NOW()                  │
│ )                                  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Persist to Database                │
│                                    │
│ session.add(decision_log)          │
│ session.commit()                   │
│                                    │
│ Purpose:                           │
│ • Audit trail                      │
│ • Explainability                   │
│ • Future learning                  │
│ • Performance analysis             │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Return Success                     │
│                                    │
│ {                                  │
│   success: True,                   │
│   logged: True                     │
│ }                                  │
└────────────────────────────────────┘

Note: Logging is non-blocking (asyncio.create_task)
      - Main process continues without waiting
      - Doesn't delay user response
```

---

### 4.2 Feedback Processing & Adaptive Learning Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│          LEARNING SUB-AGENT (Feedback Component)                     │
│          Adaptive Parameter Update via Feedback Loop                 │
└─────────────────────────────────────────────────────────────────────┘

START: process_feedback(txn_id, actual_outcome, notes)
  │
  │ Triggered by:
  │ • Human analyst review
  │ • Customer dispute
  │ • Confirmed fraud report
  │ • Chargeback notification
  │
  ▼
┌────────────────────────────────────┐
│ Retrieve DecisionLog               │
│                                    │
│ decision_log = query(              │
│   DecisionLog                      │
│ ).filter(                          │
│   transaction_id == txn_id         │
│ ).first()                          │
│                                    │
│ IF NOT found:                      │
│   RETURN error                     │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Extract Original Decision          │
│                                    │
│ original_decision = decision_log   │
│   .decision                        │
│                                    │
│ Possible values:                   │
│ • ALLOW                            │
│ • CHALLENGE                        │
│ • DENY                             │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Parse Actual Outcome               │
│                                    │
│ actual_outcome:                    │
│ • "fraud" (True Positive target)   │
│ • "legitimate" (True Negative)     │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Evaluate Decision Correctness      │
│                                    │
│ IF actual_outcome == "fraud":      │
│   was_correct = decision in        │
│     ["DENY", "CHALLENGE"]          │
│                                    │
│ ELSE: # legitimate                 │
│   was_correct = decision in        │
│     ["ALLOW", "CHALLENGE"]         │
│                                    │
│ Rationale:                         │
│ • Fraud: Should block/verify       │
│ • Legit: Should allow/verify       │
│ • CHALLENGE is cautious (correct)  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Calculate Reward/Penalty           │
│                                    │
│ CONFUSION MATRIX LOGIC:            │
│                                    │
│ IF was_correct:                    │
│   reward = +1.0                    │
│   (True Positive or True Negative) │
│                                    │
│ ELSE IF actual="fraud" AND         │
│         decision="ALLOW":          │
│   reward = -10.0                   │
│   (FALSE NEGATIVE - missed fraud!) │
│                                    │
│ ELSE IF actual="legitimate" AND    │
│         decision="DENY":           │
│   reward = -2.0                    │
│   (FALSE POSITIVE - wrong block)   │
│                                    │
│ ELSE:                              │
│   reward = -1.0                    │
│   (Other error)                    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Create Feedback Record             │
│                                    │
│ Feedback(                          │
│   decision_log_id,                 │
│   transaction_id,                  │
│   actual_outcome,                  │
│   notes,                           │
│   reward,                          │
│   was_correct,                     │
│   timestamp=NOW()                  │
│ )                                  │
│                                    │
│ session.add(feedback)              │
└────────────┬───────────────────────┘
             │
             ▼
        ┌─────────┐
        │ Correct?│
        └────┬───┬┘
             │   │
         Yes │   │ No
             │   │
             │   └────────────────────────────┐
             │                                │
             ▼                                ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│ No Parameter Update  │    │ ADAPTIVE PARAMETER UPDATE       │
│                      │    │                                 │
│ System performing    │    │ Error detected - adjust system  │
│ correctly            │    └────────────┬────────────────────┘
│                      │                 │
│ Just log for metrics │                 ▼
└──────────────────────┘    ┌─────────────────────────────────┐
             │              │ Load Current Parameters         │
             │              │                                 │
             │              │ params = query(                 │
             │              │   AdaptiveParameters            │
             │              │ ).filter(is_active).first()     │
             │              │                                 │
             │              │ IF NOT exists:                  │
             │              │   CREATE with defaults          │
             │              └────────────┬────────────────────┘
             │                           │
             │                           ▼
             │              ┌─────────────────────────────────┐
             │              │ Determine Update Strategy       │
             │              │                                 │
             │              │ learning_rate = α = 0.02        │
             │              └────────┬───────────────┬────────┘
             │                       │               │
             │              ┌────────┴──────┐        │
             │              │               │        │
             │              ▼               ▼        │
             │    ┌──────────────┐  ┌──────────────┐│
             │    │ FALSE NEG    │  │ FALSE POS    ││
             │    │ (Missed      │  │ (Wrong       ││
             │    │  Fraud)      │  │  Denial)     ││
             │    └──────┬───────┘  └──────┬───────┘│
             │           │                 │        │
             │           ▼                 ▼        │
             │  ┌─────────────────────────────────┐ │
             │  │ UPDATE LOGIC:                   │ │
             │  │                                 │ │
             │  │ FALSE NEGATIVE:                 │ │
             │  │ ───────────────                 │ │
             │  │ Increase sensitivity to fraud   │ │
             │  │                                 │ │
             │  │ behavioral_weight = min(0.8,    │ │
             │  │   current + α)                  │ │
             │  │                                 │ │
             │  │ threshold_low = max(0.1,        │ │
             │  │   current - α/2)                │ │
             │  │                                 │ │
             │  │ Effect: More aggressive         │ │
             │  │                                 │ │
             │  │                                 │ │
             │  │ FALSE POSITIVE:                 │ │
             │  │ ───────────────                 │ │
             │  │ Relax to reduce false alarms    │ │
             │  │                                 │ │
             │  │ threshold_high = min(0.9,       │ │
             │  │   current + α/2)                │ │
             │  │                                 │ │
             │  │ Effect: Less aggressive         │ │
             │  └────────────┬────────────────────┘ │
             │               │                      │
             │               ▼                      │
             │  ┌─────────────────────────────────┐ │
             │  │ Update Database                 │ │
             │  │                                 │ │
             │  │ params.behavioral_weight = ...  │ │
             │  │ params.policy_weight = ...      │ │
             │  │ params.threshold_low = ...      │ │
             │  │ params.threshold_high = ...     │ │
             │  │ params.total_updates += 1       │ │
             │  │ params.last_update = NOW()      │ │
             │  │ params.update_reason = reason   │ │
             │  │                                 │ │
             │  │ session.commit()                │ │
             │  └────────────┬────────────────────┘ │
             │               │                      │
             │               │                      │
             └───────────────┴──────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │ Update Transaction Ground    │
                │ Truth                        │
                │                              │
                │ transaction.is_fraud =       │
                │   (actual_outcome=="fraud")  │
                │                              │
                │ Enables supervised learning  │
                └────────────┬─────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │ FEEDBACK OUTPUT              │
                │                              │
                │ Return:                      │
                │ {                            │
                │   success: True,             │
                │   was_correct,               │
                │   reward,                    │
                │   parameters_updated: bool,  │
                │   original_decision,         │
                │   actual_outcome             │
                │ }                            │
                └────────────┬─────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │ Reload Parameters            │
                │                              │
                │ coordinator_agent.           │
                │   initialize()               │
                │                              │
                │ • Fusion sub-agent gets      │
                │   updated weights            │
                │ • Decision sub-agent gets    │
                │   updated thresholds         │
                │                              │
                │ Changes take effect for next │
                │ transaction immediately      │
                └──────────────────────────────┘
```

---

### 4.3 System Metrics Calculation

```
┌─────────────────────────────────────────────────────────────────────┐
│          LEARNING SUB-AGENT (Metrics Component)                      │
│          Performance Analysis via Confusion Matrix                   │
└─────────────────────────────────────────────────────────────────────┘

START: get_metrics()
  │
  ▼
┌────────────────────────────────────┐
│ Query All Feedback Records         │
│                                    │
│ feedbacks = query(Feedback).all()  │
│                                    │
│ IF empty:                          │
│   RETURN no_data                   │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Initialize Confusion Matrix        │
│                                    │
│ TP = 0  # True Positives           │
│ TN = 0  # True Negatives           │
│ FP = 0  # False Positives          │
│ FN = 0  # False Negatives          │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Iterate Through Feedback           │
│                                    │
│ FOR each feedback:                 │
│   decision_log = get_decision()    │
│   decision = decision_log.decision │
│   outcome = feedback.actual_outcome│
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Classify Into Confusion Matrix     │
│                                    │
│ IF outcome == "fraud":             │
│   IF decision in ["DENY",          │
│                    "CHALLENGE"]:   │
│     TP += 1  ✓ Caught fraud        │
│   ELSE:                            │
│     FN += 1  ✗ Missed fraud        │
│                                    │
│ ELSE: # legitimate                 │
│   IF decision == "ALLOW":          │
│     TN += 1  ✓ Correct allow       │
│   ELSE:                            │
│     FP += 1  ✗ Wrong denial        │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ Calculate Performance Metrics      │
│                                    │
│ PRECISION (PPV):                   │
│ ────────────────                   │
│ P = TP / (TP + FP)                 │
│                                    │
│ Interpretation: Of all flagged     │
│ transactions, what % were truly    │
│ fraudulent?                        │
│                                    │
│                                    │
│ RECALL (Sensitivity, TPR):         │
│ ──────────────────────             │
│ R = TP / (TP + FN)                 │
│                                    │
│ Interpretation: Of all fraudulent  │
│ transactions, what % did we catch? │
│                                    │
│                                    │
│ F1 SCORE (Harmonic Mean):          │
│ ──────────────────────             │
│ F1 = 2PR / (P + R)                 │
│    = 2TP / (2TP + FP + FN)         │
│                                    │
│ Interpretation: Balanced measure   │
│ of precision and recall            │
│                                    │
│                                    │
│ FALSE POSITIVE RATE:               │
│ ────────────────────               │
│ FPR = FP / (FP + TN)               │
│                                    │
│ Interpretation: Of legitimate txns,│
│ what % did we wrongly deny?        │
│                                    │
│                                    │
│ FALSE NEGATIVE RATE:               │
│ ────────────────────               │
│ FNR = FN / (FN + TP)               │
│                                    │
│ Interpretation: Of fraudulent txns,│
│ what % did we miss?                │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ METRICS OUTPUT                     │
│                                    │
│ Return:                            │
│ {                                  │
│   total_feedback: N,               │
│   true_positives: TP,              │
│   true_negatives: TN,              │
│   false_positives: FP,             │
│   false_negatives: FN,             │
│   precision: P,                    │
│   recall: R,                       │
│   f1_score: F1,                    │
│   false_positive_rate: FPR,        │
│   false_negative_rate: FNR         │
│ }                                  │
│                                    │
│ Used for:                          │
│ • System performance monitoring    │
│ • Stakeholder reporting            │
│ • A/B testing parameter changes    │
│ • Regulatory compliance            │
└────────────────────────────────────┘
```

---

## 5. COORDINATOR AGENT COMPLETE PROCESSING PIPELINE

### 5.1 End-to-End Sequential Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│              COORDINATOR AGENT: process()                            │
│              Sequential Processing of 3 Sub-Agents + Explanation     │
└─────────────────────────────────────────────────────────────────────┘

START: process(input_data)
  │
  │ Input from Evaluation Agent:
  │ • enriched_transaction
  │ • behavioral_assessment {A, c_b, E_b, S, D}
  │ • policy_assessment {P, c_p, E_p, V, P_o, P_r}
  │
  ▼
┌────────────────────────────────────┐
│ Record Start Time                  │
│                                    │
│ start_time = NOW()                 │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 1: FUSION                     │
│                                    │
│ fusion_result = await              │
│   fusion_subagent.execute({        │
│     behavioral_assessment,         │
│     policy_assessment              │
│   })                               │
│                                    │
│ Output:                            │
│ • fused_score: F                   │
│ • confidence: C                    │
│ • behavioral_contribution          │
│ • policy_contribution              │
│ • override_reason (if any)         │
│ • weights_used {w_b, w_p}          │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 2: DECISION                   │
│                                    │
│ decision_result = await            │
│   decision_subagent.execute({      │
│     fused_score: F,                │
│     confidence: C,                 │
│     override_reason                │
│   })                               │
│                                    │
│ Output:                            │
│ • decision: D (ALLOW/CHAL/DENY)    │
│ • decision_reason                  │
│ • thresholds_used {θ_low, θ_high}  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 3: GENERATE EXPLANATION       │
│                                    │
│ explanation = await                │
│   _generate_explanation(           │
│     transaction,                   │
│     behavioral,                    │
│     policy,                        │
│     decision: D,                   │
│     fused_score: F                 │
│   )                                │
│                                    │
│ Uses LLM to create human-readable  │
│ explanation combining all factors  │
│                                    │
│ Fallback: Rule-based if LLM fails  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 4: COMPILE EVIDENCE           │
│                                    │
│ evidence = {                       │
│   behavioral_rag: {                │
│     similar_transactions: S,       │
│     deviations: D,                 │
│     statistical_analysis           │
│   },                               │
│   policy_rag: {                    │
│     retrieved_policies: Pol[],     │
│     violations: V[]                │
│   }                                │
│ }                                  │
│                                    │
│ Purpose: Explainability & audit    │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 5: CALCULATE PROCESSING TIME  │
│                                    │
│ processing_time_ms =               │
│   (NOW() - start_time) × 1000      │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 6: ASSEMBLE FINAL OUTPUT      │
│                                    │
│ output = {                         │
│   success: True,                   │
│   decision: D,                     │
│   fused_score: F,                  │
│   confidence: C,                   │
│   behavioral_score: A,             │
│   policy_score: P,                 │
│   explanation: E,                  │
│   evidence: Ev,                    │
│   transaction_id: txn_id,          │
│   processing_time_ms,              │
│   weights_used,                    │
│   thresholds_used,                 │
│   behavioral_assessment,           │
│   policy_assessment                │
│ }                                  │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ STEP 7: ASYNC LOGGING              │
│                                    │
│ asyncio.create_task(               │
│   learning_subagent.execute(       │
│     output + additional_context    │
│   )                                │
│ )                                  │
│                                    │
│ Non-blocking: Logs in background   │
│ Main process continues immediately │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ RETURN OUTPUT                      │
│                                    │
│ Return to caller (API/System)      │
└────────────────────────────────────┘
```

---

## 6. COMPLETE SYSTEM INTEGRATION DIAGRAM

### 6.1 Monitor → Evaluation → Coordinator Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                COMPLETE GUARDIAN SYSTEM FLOW                         │
│                3-Agent Architecture with RAG & Adaptive Learning     │
└─────────────────────────────────────────────────────────────────────┘

USER TRANSACTION
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    MONITOR AGENT                              │
│                    (Perception Layer)                         │
│                                                               │
│  Sub-Agents:                                                  │
│  • Capture Sub-Agent      (normalize data)                    │
│  • Context Sub-Agent       (query profile)                    │
│  • Feature Sub-Agent       (extract features)                 │
│                                                               │
│  Output: {T', C, F, E, R₀}                                    │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    EVALUATION AGENT                           │
│                    (RAG-Based Analysis)                       │
│                                                               │
│  Parallel Sub-Agents (asyncio.gather):                        │
│                                                               │
│  ┌──────────────────────┐    ┌──────────────────────┐        │
│  │ Behavioral Sub-Agent │    │ Policy Sub-Agent      │        │
│  │ (RAG for Anomaly)    │    │ (RAG for Compliance)  │        │
│  │                      │    │                       │        │
│  │ • Query user txns    │    │ • Query org policies  │        │
│  │ • K=5 neighbors      │    │ • Query reg policies  │        │
│  │ • Statistical calc   │    │ • K=3 chunks each     │        │
│  │ • LLM analysis       │    │ • LLM compliance      │        │
│  │ • Hybrid score       │    │ • Regulatory override │        │
│  └──────────┬───────────┘    └──────────┬───────────┘        │
│             │                           │                    │
│             └───────────┬───────────────┘                    │
│                         │                                    │
│  Output: {Behavioral Assessment, Policy Assessment}          │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    COORDINATOR AGENT                          │
│                    (Decision Fusion & Learning)               │
│                                                               │
│  Sequential Sub-Agents:                                       │
│                                                               │
│  ┌────────────────────────────────────────────────┐          │
│  │ 1. FUSION SUB-AGENT                            │          │
│  │    • Weighted blend (w_b=0.6, w_p=0.4)         │          │
│  │    • Regulatory override check (≥0.9)          │          │
│  │    • Confidence calculation                    │          │
│  │    Output: F, C, contributions                 │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────┐          │
│  │ 2. DECISION SUB-AGENT                          │          │
│  │    • Threshold logic                           │          │
│  │      - F < 0.4 → ALLOW                         │          │
│  │      - F ≥ 0.7 → DENY                          │          │
│  │      - 0.4-0.7 → CHALLENGE                     │          │
│  │    • Override handling                         │          │
│  │    Output: D, reason, thresholds               │          │
│  └────────────────────┬───────────────────────────┘          │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────┐          │
│  │ 3. LEARNING SUB-AGENT                          │          │
│  │    • Log decision to DB (async)                │          │
│  │    • Store full evidence                       │          │
│  │    • Await feedback                            │          │
│  │    Output: Logged confirmation                 │          │
│  └────────────────────────────────────────────────┘          │
│                                                               │
│  Final Output: {D, F, C, E, Ev, processing_time}             │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │  USER RESPONSE  │
                  │                 │
                  │  • ALLOW        │
                  │  • CHALLENGE    │
                  │  • DENY         │
                  └─────────────────┘
                            │
                            │ (Later: Ground Truth)
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    FEEDBACK LOOP                              │
│                    (Adaptive Learning)                        │
│                                                               │
│  Trigger: Human review / Fraud confirmation                   │
│                                                               │
│  ┌──────────────────────────────────────────────┐            │
│  │ Learning Sub-Agent: process_feedback()       │            │
│  │                                              │            │
│  │ • Compare decision vs actual_outcome         │            │
│  │ • Calculate reward/penalty                   │            │
│  │ • Update confusion matrix                    │            │
│  │                                              │            │
│  │ IF error detected:                           │            │
│  │   • Adjust weights (w_b, w_p)                │            │
│  │   • Adjust thresholds (θ_low, θ_high)        │            │
│  │   • Persist to AdaptiveParameters table      │            │
│  │   • Reload into Fusion & Decision agents     │            │
│  └──────────────────────────────────────────────┘            │
│                                                               │
│  Parameters updated → Next transaction uses new settings      │
└───────────────────────────────────────────────────────────────┘


TIMING BREAKDOWN (Example):
═══════════════════════════

Monitor Agent:        ~50ms
  • Capture:           10ms
  • Context:           20ms (DB query)
  • Feature:           20ms

Evaluation Agent:    ~200ms (parallel)
  • Behavioral:       150ms (vector search + LLM)
  • Policy:           200ms (vector search + LLM)
  • Parallel max:     200ms

Coordinator Agent:    ~50ms
  • Fusion:            5ms (arithmetic)
  • Decision:          5ms (threshold logic)
  • Explanation:      30ms (LLM)
  • Logging:          10ms (async, non-blocking)

TOTAL LATENCY:       ~300ms
```

---

## 7. ADAPTIVE LEARNING MECHANISM

### 7.1 Parameter Evolution Over Time

```
┌─────────────────────────────────────────────────────────────────────┐
│              ADAPTIVE PARAMETER EVOLUTION                            │
│              Continuous Learning from Feedback                       │
└─────────────────────────────────────────────────────────────────────┘

INITIAL STATE (t=0):
════════════════════
behavioral_weight = 0.6
policy_weight = 0.4
threshold_low = 0.4
threshold_high = 0.7


FEEDBACK SCENARIO 1: FALSE NEGATIVE (Missed Fraud)
═══════════════════════════════════════════════════

Transaction:
• Decision: ALLOW (F=0.35)
• Actual: FRAUD ✗

Feedback Processing:
• reward = -10.0 (severe penalty)
• was_correct = False

Parameter Update:
• behavioral_weight += α (0.02)
  0.6 → 0.62
  
• threshold_low -= α/2 (0.01)
  0.4 → 0.39
  
Effect: System becomes more sensitive to fraud signals


FEEDBACK SCENARIO 2: FALSE POSITIVE (Wrong Denial)
═══════════════════════════════════════════════════

Transaction:
• Decision: DENY (F=0.75)
• Actual: LEGITIMATE ✗

Feedback Processing:
• reward = -2.0 (moderate penalty)
• was_correct = False

Parameter Update:
• threshold_high += α/2 (0.01)
  0.7 → 0.71
  
Effect: System becomes less aggressive, reduces false alarms


FEEDBACK SCENARIO 3: TRUE POSITIVE (Correct Detection)
═══════════════════════════════════════════════════════

Transaction:
• Decision: DENY (F=0.82)
• Actual: FRAUD ✓

Feedback Processing:
• reward = +1.0
• was_correct = True

Parameter Update:
• NO CHANGE (system performing correctly)

Effect: Reinforces current strategy


PARAMETER TRAJECTORY (Example):
════════════════════════════════

Time  | behavioral_w | policy_w | θ_low | θ_high | F1_score
──────┼──────────────┼──────────┼───────┼────────┼─────────
t=0   | 0.60         | 0.40     | 0.40  | 0.70   | N/A
t=1   | 0.62         | 0.38     | 0.39  | 0.70   | 0.75
t=2   | 0.62         | 0.38     | 0.39  | 0.71   | 0.78
t=3   | 0.64         | 0.36     | 0.38  | 0.71   | 0.82
t=4   | 0.64         | 0.36     | 0.38  | 0.72   | 0.84
...

Convergence: Parameters stabilize as system learns optimal balance
```

---

## 8. DECISION FUSION FORMULAS

### 8.1 Mathematical Definitions

```
FUSION SUB-AGENT:
═════════════════

Inputs:
  A = anomaly_score ∈ [0, 1]      (from Behavioral)
  P = policy_score ∈ [0, 1]       (from Policy)
  w_b = behavioral_weight         (adaptive)
  w_p = policy_weight             (adaptive)
  P_r = regulatory_score ∈ [0, 1] (from Policy)

Regulatory Override:
  IF P_r ≥ 0.9:
    F = P_r
    RETURN (skip fusion)

Normalized Weights:
  w_b_norm = w_b / (w_b + w_p)
  w_p_norm = w_p / (w_b + w_p)
  
  Constraint: w_b_norm + w_p_norm = 1.0

Fused Score:
  F = (A × w_b_norm) + (P × w_p_norm)
  F = min(1.0, F)

Fused Confidence:
  c_b = behavioral_confidence ∈ [0, 1]
  c_p = policy_confidence ∈ [0, 1]
  
  C = (c_b × w_b_norm) + (c_p × w_p_norm)
  C = min(1.0, C)

Contributions (for explainability):
  contribution_behavioral = A × w_b_norm
  contribution_policy = P × w_p_norm
  
  Verification: F = contribution_behavioral + contribution_policy


DECISION SUB-AGENT:
═══════════════════

Inputs:
  F = fused_score ∈ [0, 1]
  θ_low = threshold_low     (adaptive, default 0.4)
  θ_high = threshold_high   (adaptive, default 0.7)

Decision Function:
  D(F) = {
           ALLOW      if F < θ_low
           DENY       if F ≥ θ_high
           CHALLENGE  if θ_low ≤ F < θ_high
         }

Threshold Constraints:
  0.0 ≤ θ_low < θ_high ≤ 1.0
  θ_low ∈ [0.1, 0.5]     (practical bounds)
  θ_high ∈ [0.6, 0.9]    (practical bounds)


LEARNING SUB-AGENT:
═══════════════════

Reward Function:
  R(D, outcome) = {
                    +1.0   if correct
                    -2.0   if FP (wrong DENY)
                    -10.0  if FN (wrong ALLOW)
                    -1.0   otherwise
                  }

Parameter Update (Gradient Descent):
  α = learning_rate = 0.02
  
  False Negative Update:
    w_b ← min(0.8, w_b + α)
    θ_low ← max(0.1, θ_low - α/2)
  
  False Positive Update:
    θ_high ← min(0.9, θ_high + α/2)

Metrics:
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1 = 2PR / (P + R)
  FPR = FP / (FP + TN)
  FNR = FN / (FN + TP)
```

---

## 9. PSEUDOCODE FOR COORDINATOR AGENT

```python
# COORDINATOR AGENT - COMPLETE PSEUDOCODE
# ========================================

class CoordinatorAgent:
    """
    Orchestrates decision fusion, threshold logic, and adaptive learning
    """
    
    def __init__(self):
        self.fusion_subagent = FusionSubAgent()
        self.decision_subagent = DecisionSubAgent()
        self.learning_subagent = LearningSubAgent()
    
    def initialize(self):
        """Load adaptive parameters from database"""
        params = db.query(AdaptiveParameters).filter(is_active=True).first()
        
        if params:
            self.fusion_subagent.update_weights(
                params.behavioral_weight,
                params.policy_weight
            )
            self.decision_subagent.update_thresholds(
                params.threshold_low,
                params.threshold_high
            )
    
    async def process(self, input_data):
        """
        Main processing pipeline: Fusion → Decision → Learning
        """
        start_time = now()
        
        behavioral = input_data['behavioral_assessment']
        policy = input_data['policy_assessment']
        transaction = input_data['enriched_transaction']
        
        # STEP 1: Fuse scores
        fusion_result = await self.fusion_subagent.execute({
            'behavioral_assessment': behavioral,
            'policy_assessment': policy
        })
        
        # STEP 2: Make decision
        decision_result = await self.decision_subagent.execute({
            'fused_score': fusion_result['fused_score'],
            'confidence': fusion_result['confidence'],
            'override_reason': fusion_result['override_reason']
        })
        
        # STEP 3: Generate explanation (LLM)
        explanation = await llm_client.generate_decision_explanation(
            transaction, behavioral, policy,
            decision_result['decision'],
            fusion_result['fused_score']
        )
        
        # STEP 4: Compile evidence
        evidence = {
            'behavioral_rag': behavioral['similar_transactions'],
            'policy_rag': policy['retrieved_policies']
        }
        
        # STEP 5: Calculate processing time
        processing_time_ms = (now() - start_time) * 1000
        
        # STEP 6: Assemble output
        output = {
            'success': True,
            'decision': decision_result['decision'],
            'fused_score': fusion_result['fused_score'],
            'confidence': fusion_result['confidence'],
            'explanation': explanation,
            'evidence': evidence,
            'processing_time_ms': processing_time_ms,
            'weights_used': fusion_result['weights_used'],
            'thresholds_used': decision_result['thresholds_used']
        }
        
        # STEP 7: Async logging (non-blocking)
        asyncio.create_task(
            self.learning_subagent.execute(output)
        )
        
        return output
    
    async def submit_feedback(self, txn_id, actual_outcome, notes=None):
        """
        Process feedback and trigger adaptive learning
        """
        result = await self.learning_subagent.process_feedback(
            txn_id, actual_outcome, notes
        )
        
        # Reload parameters if updated
        if result['parameters_updated']:
            self.initialize()
        
        return result


class FusionSubAgent:
    """Score fusion with weighted blend"""
    
    def __init__(self):
        self.w_b = 0.6  # behavioral_weight (adaptive)
        self.w_p = 0.4  # policy_weight (adaptive)
    
    async def execute(self, input_data):
        A = input_data['behavioral_assessment']['anomaly_score']
        P = input_data['policy_assessment']['policy_score']
        P_r = input_data['policy_assessment']['regulatory_score']
        
        # Regulatory override
        if P_r >= 0.9:
            return {
                'fused_score': P_r,
                'confidence': 0.95,
                'override_reason': 'regulatory_violation'
            }
        
        # Normalize weights
        total = self.w_b + self.w_p
        w_b_norm = self.w_b / total
        w_p_norm = self.w_p / total
        
        # Fuse
        F = (A * w_b_norm) + (P * w_p_norm)
        F = min(1.0, F)
        
        return {
            'fused_score': F,
            'confidence': calculate_confidence(A, P, w_b_norm, w_p_norm),
            'override_reason': None
        }


class DecisionSubAgent:
    """Threshold-based decision logic"""
    
    def __init__(self):
        self.theta_low = 0.4   # threshold_low (adaptive)
        self.theta_high = 0.7  # threshold_high (adaptive)
    
    async def execute(self, input_data):
        F = input_data['fused_score']
        override = input_data['override_reason']
        
        if override == 'regulatory_violation':
            return {'decision': 'DENY', 'reason': 'Regulatory violation'}
        
        if F < self.theta_low:
            return {'decision': 'ALLOW', 'reason': f'Low risk ({F:.2f})'}
        elif F >= self.theta_high:
            return {'decision': 'DENY', 'reason': f'High risk ({F:.2f})'}
        else:
            return {'decision': 'CHALLENGE', 'reason': f'Moderate risk ({F:.2f})'}


class LearningSubAgent:
    """Logging and adaptive learning"""
    
    async def execute(self, input_data):
        """Log decision to database"""
        decision_log = DecisionLog(**input_data)
        db.save(decision_log)
        return {'success': True}
    
    async def process_feedback(self, txn_id, actual_outcome, notes):
        """Process feedback and update parameters"""
        decision_log = db.query(DecisionLog).filter(
            transaction_id=txn_id
        ).first()
        
        original_decision = decision_log.decision
        
        # Evaluate correctness
        was_correct = self._evaluate(original_decision, actual_outcome)
        
        # Calculate reward
        reward = self._reward(original_decision, actual_outcome, was_correct)
        
        # Save feedback
        feedback = Feedback(
            decision_log_id=decision_log.id,
            actual_outcome=actual_outcome,
            reward=reward,
            was_correct=was_correct
        )
        db.save(feedback)
        
        # Update parameters if error
        params_updated = False
        if not was_correct:
            params_updated = self._update_parameters(
                original_decision, actual_outcome
            )
        
        return {
            'was_correct': was_correct,
            'reward': reward,
            'parameters_updated': params_updated
        }
    
    def _update_parameters(self, decision, actual_outcome):
        """Adaptive parameter update"""
        params = db.query(AdaptiveParameters).filter(is_active=True).first()
        
        alpha = 0.02  # learning_rate
        
        if actual_outcome == 'fraud' and decision == 'ALLOW':
            # FALSE NEGATIVE: Increase sensitivity
            params.behavioral_weight = min(0.8, params.behavioral_weight + alpha)
            params.threshold_low = max(0.1, params.threshold_low - alpha/2)
        
        elif actual_outcome == 'legitimate' and decision == 'DENY':
            # FALSE POSITIVE: Relax thresholds
            params.threshold_high = min(0.9, params.threshold_high + alpha/2)
        
        db.save(params)
        return True
```

---

## 10. KEY INSIGHTS FOR ACADEMIC PAPER

1. **Three-Tier Decision Architecture**: ALLOW (seamless) → CHALLENGE (verify) → DENY (block) provides graduated risk response

2. **Adaptive Fusion Weights**: System learns optimal balance between behavioral (anomaly) and policy (compliance) signals via gradient descent on feedback

3. **Regulatory Precedence**: Hard override (P_r ≥ 0.9) ensures absolute compliance with legal requirements, bypassing all other logic

4. **Asymmetric Penalties**: False negatives (missed fraud) penalized 5× more than false positives (-10 vs -2) reflects business priorities

5. **Continuous Learning**: Feedback loop updates parameters in real-time, enabling system to adapt to evolving fraud patterns

6. **Explainable Decisions**: Every decision includes:
   - Numerical scores (F, A, P)
   - Natural language explanation (LLM-generated)
   - Evidence citations (RAG retrievals)
   - Parameter transparency (weights, thresholds)

7. **Performance Tracking**: Confusion matrix metrics (Precision, Recall, F1, FPR, FNR) enable data-driven optimization

8. **Non-blocking Logging**: Async decision recording ensures sub-10ms user latency despite comprehensive audit trail

---

## END OF FLOWCHART
