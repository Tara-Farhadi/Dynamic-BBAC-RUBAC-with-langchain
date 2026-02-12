"""Database models for GUARDIAN transaction monitoring system."""
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, DateTime, Boolean, Text, Integer, 
    ForeignKey, JSON, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class DecisionType(enum.Enum):
    """Transaction decision types."""
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"


class FeedbackOutcome(enum.Enum):
    """Feedback outcome types."""
    FRAUD = "fraud"
    LEGITIMATE = "legitimate"


class Transaction(Base):
    """Transaction record model - stores all evaluated transactions."""
    
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    cc_num = Column(String(20))  # Masked for security
    trans_date_trans_time = Column(DateTime, nullable=False)
    merchant = Column(String(200))
    category = Column(String(100))
    amt = Column(Float, nullable=False)
    gender = Column(String(10))
    street = Column(String(200))
    city = Column(String(100))
    state = Column(String(50))
    zip = Column(String(20))
    country = Column(String(50), default="US")
    trans_num = Column(String(100))
    is_fraud = Column(Boolean, default=None)  # Ground truth if available
    
    # Geographic coordinates (essential for RAG-based behavioral analysis)
    lat = Column(Float)  # User's home latitude
    long = Column(Float)  # User's home longitude
    merch_lat = Column(Float)  # Merchant latitude
    merch_long = Column(Float)  # Merchant longitude
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to decision
    decision = relationship("DecisionLog", back_populates="transaction", uselist=False)
    
    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, user={self.user_id}, amt={self.amt})>"


class UserProfile(Base):
    """User behavioral profile summary."""
    
    __tablename__ = "user_profiles"
    
    user_id = Column(String(50), primary_key=True)
    total_transactions = Column(Integer, default=0)
    avg_amount = Column(Float, default=0.0)
    std_amount = Column(Float, default=0.0)
    max_amount = Column(Float, default=0.0)
    min_amount = Column(Float, default=0.0)
    common_merchants = Column(JSON)  # List of frequently used merchants
    common_categories = Column(JSON)  # List of common transaction categories
    common_locations = Column(JSON)  # List of common locations (city, state)
    typical_hours = Column(JSON)  # Typical transaction hours
    risk_level = Column(String(20), default="unknown")  # low, medium, high
    
    # Geographic behavioral baseline
    home_lat = Column(Float)  # User's home latitude
    home_long = Column(Float)  # User's home longitude
    avg_shopping_distance = Column(Float)  # Average distance from home (miles)
    max_shopping_distance = Column(Float)  # Maximum distance from home (miles)
    
    # Timestamps
    first_transaction_date = Column(DateTime)
    last_transaction_date = Column(DateTime)
    profile_updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id}, risk_level={self.risk_level})>"


class DecisionLog(Base):
    """Log of all decisions made by the system."""
    
    __tablename__ = "decision_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(100), ForeignKey("transactions.transaction_id"), unique=True)
    
    # Decision details
    decision = Column(String(20), nullable=False)  # ALLOW, CHALLENGE, DENY
    fused_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    behavioral_score = Column(Float)
    policy_score = Column(Float)
    
    # Explanations
    explanation = Column(Text)
    evidence = Column(JSON)  # Detailed evidence from RAG
    
    # Agent outputs (for debugging/audit)
    monitor_output = Column(JSON)
    evaluation_output = Column(JSON)
    coordinator_output = Column(JSON)
    
    # Weights used at decision time
    behavioral_weight = Column(Float)
    policy_weight = Column(Float)
    
    # Timestamps
    decision_timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time_ms = Column(Float)  # Time taken to process
    
    # Relationship
    transaction = relationship("Transaction", back_populates="decision")
    feedback = relationship("Feedback", back_populates="decision_log", uselist=False)
    
    def __repr__(self):
        return f"<DecisionLog(txn={self.transaction_id}, decision={self.decision})>"


class Feedback(Base):
    """Feedback on decisions for adaptive learning."""
    
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_log_id = Column(Integer, ForeignKey("decision_logs.id"), unique=True)
    transaction_id = Column(String(100), nullable=False, index=True)
    
    # Feedback details
    actual_outcome = Column(String(20), nullable=False)  # fraud, legitimate
    notes = Column(Text)
    
    # Calculated values
    reward = Column(Float)  # Reward/penalty applied
    was_correct = Column(Boolean)  # Whether decision was correct
    
    # Timestamps
    feedback_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    decision_log = relationship("DecisionLog", back_populates="feedback")
    
    def __repr__(self):
        return f"<Feedback(txn={self.transaction_id}, outcome={self.actual_outcome})>"


class PolicyDocument(Base):
    """Metadata for uploaded policy documents."""
    
    __tablename__ = "policy_documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(200), nullable=False)
    policy_type = Column(String(50), nullable=False)  # organizational, regulatory
    
    # Document info
    file_path = Column(String(500))
    file_size = Column(Integer)
    total_chunks = Column(Integer)
    
    # Processing status
    is_indexed = Column(Boolean, default=False)
    index_date = Column(DateTime)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PolicyDocument(filename={self.filename}, type={self.policy_type})>"


class SystemMetrics(Base):
    """System performance and accuracy metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_date = Column(DateTime, default=datetime.utcnow)
    
    # Decision counts
    total_decisions = Column(Integer, default=0)
    allow_count = Column(Integer, default=0)
    challenge_count = Column(Integer, default=0)
    deny_count = Column(Integer, default=0)
    
    # Accuracy metrics (calculated from feedback)
    true_positives = Column(Integer, default=0)  # Correctly identified fraud
    true_negatives = Column(Integer, default=0)  # Correctly allowed legitimate
    false_positives = Column(Integer, default=0)  # Wrongly denied legitimate
    false_negatives = Column(Integer, default=0)  # Wrongly allowed fraud
    
    # Calculated metrics
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    false_positive_rate = Column(Float)
    false_negative_rate = Column(Float)
    
    # Current weights
    behavioral_weight = Column(Float)
    policy_weight = Column(Float)
    threshold_low = Column(Float)
    threshold_high = Column(Float)
    
    def __repr__(self):
        return f"<SystemMetrics(date={self.metric_date}, f1={self.f1_score})>"


class AdaptiveParameters(Base):
    """Store adaptive learning parameters."""
    
    __tablename__ = "adaptive_parameters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Current weights
    behavioral_weight = Column(Float, default=0.5)
    policy_weight = Column(Float, default=0.5)
    
    # Current thresholds
    threshold_low = Column(Float, default=0.3)
    threshold_high = Column(Float, default=0.7)
    
    # Learning metadata
    total_updates = Column(Integer, default=0)
    last_update = Column(DateTime, default=datetime.utcnow)
    update_reason = Column(Text)
    
    # Version for tracking
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<AdaptiveParameters(v={self.version}, bw={self.behavioral_weight}, pw={self.policy_weight})>"
