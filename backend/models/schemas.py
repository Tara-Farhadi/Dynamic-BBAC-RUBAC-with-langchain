"""Pydantic models for API request/response validation."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ============================================================
# Enums
# ============================================================

class DecisionType(str, Enum):
    """Transaction decision types."""
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


class FeedbackOutcome(str, Enum):
    """Feedback outcome types."""
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


class PolicyType(str, Enum):
    """Policy document types."""
    ORGANIZATIONAL = "organizational"
    REGULATORY = "regulatory"


# ============================================================
# Transaction Models
# ============================================================

class TransactionRequest(BaseModel):
    """Request model for transaction evaluation."""
    user_id: str = Field(..., description="Unique user identifier")
    cc_num: Optional[str] = Field(None, description="Credit card number (will be masked)")
    trans_date_trans_time: datetime = Field(..., description="Transaction timestamp")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., gt=0, description="Transaction amount")
    gender: Optional[str] = Field(None, description="User gender")
    street: Optional[str] = Field(None, description="Street address")
    city: str = Field(..., description="City of transaction")
    state: str = Field(..., description="State/province code")
    zip: Optional[str] = Field(None, description="ZIP/postal code")
    country: str = Field(default="US", description="Country code")
    trans_num: Optional[str] = Field(None, description="Unique transaction number")
    
    @field_validator('cc_num')
    @classmethod
    def mask_credit_card(cls, v: Optional[str]) -> Optional[str]:
        """Mask credit card number for security."""
        if v and len(v) > 4:
            return '*' * (len(v) - 4) + v[-4:]
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_042",
                "cc_num": "4532015112830366",
                "trans_date_trans_time": "2024-02-15T02:30:00",
                "merchant": "Electronics_Store_Moscow",
                "category": "shopping_net",
                "amt": 8500.00,
                "city": "Moscow",
                "state": "MOW",
                "country": "RU"
            }
        }


class BehavioralEvidence(BaseModel):
    """Evidence from behavioral RAG analysis."""
    similar_transactions: List[Dict[str, Any]] = Field(default_factory=list)
    similarity_scores: List[float] = Field(default_factory=list)
    deviations: List[str] = Field(default_factory=list)
    statistical_features: Optional[Dict[str, Any]] = None


class PolicyEvidence(BaseModel):
    """Evidence from policy RAG analysis."""
    retrieved_policies: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    organizational_score: Optional[float] = None
    regulatory_score: Optional[float] = None


class EvidenceDetail(BaseModel):
    """Combined evidence from both RAG pipelines."""
    behavioral_rag: BehavioralEvidence
    policy_rag: PolicyEvidence


class TransactionDecision(BaseModel):
    """Response model for transaction evaluation."""
    decision: DecisionType = Field(..., description="Decision: ALLOW, CHALLENGE, or DENY")
    fused_score: Optional[float] = Field(None, ge=0, le=1, description="Combined risk score")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in decision")
    behavioral_score: Optional[float] = Field(None, ge=0, le=1, description="Behavioral anomaly score")
    policy_score: Optional[float] = Field(None, ge=0, le=1, description="Policy compliance score")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    evidence: EvidenceDetail = Field(..., description="Detailed evidence")
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = Field(None, description="Processing time in ms")
    behavioral_assessment: Optional[Dict[str, Any]] = Field(None, description="Full behavioral analysis")
    policy_assessment: Optional[Dict[str, Any]] = Field(None, description="Full policy analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "decision": "DENY",
                "fused_score": 0.94,
                "confidence": 0.87,
                "behavioral_score": 0.88,
                "policy_score": 1.0,
                "explanation": "High-risk transaction detected...",
                "evidence": {
                    "behavioral_rag": {
                        "similar_transactions": [],
                        "deviations": ["location", "amount", "time"]
                    },
                    "policy_rag": {
                        "retrieved_policies": ["sanctions_list.pdf"],
                        "violations": ["Sanctioned country"]
                    }
                },
                "transaction_id": "txn_20240215_023000_user042",
                "timestamp": "2024-02-15T02:30:00Z"
            }
        }


# ============================================================
# Feedback Models
# ============================================================

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    transaction_id: str = Field(..., description="Transaction ID to provide feedback for")
    actual_outcome: FeedbackOutcome = Field(..., description="Actual outcome: fraud or legitimate")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_20240215_023000_user042",
                "actual_outcome": "fraud",
                "notes": "Confirmed fraudulent by card holder"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_recorded: bool
    parameters_updated: bool
    reward_applied: float
    message: str


# ============================================================
# User Models
# ============================================================

class UserProfileResponse(BaseModel):
    """Response model for user profile."""
    user_id: str
    total_transactions: int
    avg_amount: float
    std_amount: float
    max_amount: float
    min_amount: float
    common_merchants: List[str]
    common_categories: List[str]
    common_locations: List[Dict[str, str]]
    typical_hours: List[int]
    risk_level: str
    first_transaction_date: Optional[datetime]
    last_transaction_date: Optional[datetime]


class TransactionUploadResponse(BaseModel):
    """Response model for transaction history upload."""
    user_id: str
    transactions_loaded: int
    embeddings_created: int
    profile_updated: bool
    status: str
    message: Optional[str] = None


# ============================================================
# Policy Models
# ============================================================

class PolicyUploadResponse(BaseModel):
    """Response model for policy document upload."""
    filename: str
    policy_type: PolicyType
    chunks_created: int
    embeddings_indexed: int
    status: str
    message: Optional[str] = None


# ============================================================
# Health & Status Models
# ============================================================

class HealthStatus(BaseModel):
    """Response model for health check."""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "All systems operational",
                "details": {
                    "database": "connected",
                    "vector_store": "connected",
                    "llm": "available"
                }
            }
        }


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""
    total_decisions: int
    allow_count: int
    challenge_count: int
    deny_count: int
    total_users: Optional[int] = 0
    total_transactions: Optional[int] = 0
    policy_chunks: Optional[int] = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    current_weights: Dict[str, float] = {}
    current_thresholds: Dict[str, float] = {}


# ============================================================
# Agent Communication Models
# ============================================================

class AgentMessage(BaseModel):
    """Base model for inter-agent communication."""
    agent_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MonitorAgentOutput(BaseModel):
    """Output from Monitor Agent."""
    transaction_id: str
    enriched_transaction: Dict[str, Any]
    extracted_features: Dict[str, Any]
    embedding: Optional[List[float]] = None
    user_context: Optional[Dict[str, Any]] = None


class BehavioralAssessment(BaseModel):
    """Output from Behavioral Sub-agent (RAG)."""
    anomaly_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    explanation: str
    similar_transactions: List[Dict[str, Any]]
    deviation_factors: List[str]
    statistical_analysis: Optional[Dict[str, Any]] = None


class PolicyAssessment(BaseModel):
    """Output from Policy Sub-agent (RAG)."""
    policy_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    explanation: str
    organizational_score: float
    regulatory_score: float
    violations: List[str]
    retrieved_policies: List[Dict[str, Any]]


class EvaluationAgentOutput(BaseModel):
    """Combined output from Evaluation Agent."""
    behavioral_assessment: BehavioralAssessment
    policy_assessment: PolicyAssessment


class CoordinatorDecision(BaseModel):
    """Output from Coordinator Agent."""
    decision: DecisionType
    fused_score: float
    confidence: float
    explanation: str
    behavioral_contribution: float
    policy_contribution: float
    weights_used: Dict[str, float]
