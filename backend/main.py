"""
GUARDIAN Transaction Monitoring System - FastAPI Application

Main entry point for the multi-agent transaction monitoring and access control system.
Implements the GUARDIAN (Generative User Assessment and Runtime Decision Integration
Architecture Network) framework.
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from config.settings import settings
from models.db_manager import db_manager, get_db
from models.database import (
    Transaction, UserProfile, DecisionLog,
    PolicyDocument, AdaptiveParameters
)
from models.schemas import (
    TransactionRequest, TransactionDecision,
    FeedbackRequest, FeedbackResponse,
    UserProfileResponse, TransactionUploadResponse,
    PolicyUploadResponse, HealthStatus, SystemMetricsResponse,
    PolicyType, EvidenceDetail, BehavioralEvidence, PolicyEvidence
)
from agents.monitor_agent import monitor_agent
from agents.evaluation_agent import evaluation_agent
from agents.coordinator_agent import coordinator_agent
from services.vector_store import vector_store
from services.embedding import embedding_service
from utils.helpers import (
    parse_csv_transactions, calculate_user_profile_stats,
    generate_transaction_id
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("guardian")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting GUARDIAN Transaction Monitoring System...")
    logger.info("=" * 60)

    try:
        # Initialize database
        db_manager.init_db()
        logger.info("✓ Database initialized")

        # Initialize agents
        monitor_agent.initialize()
        logger.info("✓ Monitor Agent ready")

        evaluation_agent.initialize()
        logger.info("✓ Evaluation Agent ready")

        coordinator_agent.initialize()
        logger.info("✓ Coordinator Agent ready")

        # Check vector store
        stats = vector_store.get_stats()
        logger.info(
            f"✓ Vector store: {
                stats['policy_documents']} policy chunks indexed")

        logger.info("=" * 60)
        logger.info("✓ GUARDIAN System ready!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GUARDIAN System...")
    db_manager.close()


# Create FastAPI application
app = FastAPI(
    title="GUARDIAN Transaction Monitor",
    description="""
    Multi-Agent Transaction Monitoring & Access Control System

    Implements the GUARDIAN (Generative User Assessment and Runtime Decision
    Integration Architecture Network) framework for real-time transaction evaluation
    using RAG (Retrieval-Augmented Generation).

    ## Features
    - Real-time transaction evaluation with ALLOW/CHALLENGE/DENY decisions
    - Behavioral pattern analysis using RAG over transaction history
    - Policy compliance checking using RAG over policy documents
    - Adaptive learning from feedback
    - Multi-agent architecture with independent, on-demand agents

    ## Agents
    - **Monitor Agent**: Perception and feature extraction
    - **Evaluation Agent**: Behavioral and policy RAG analysis
    - **Coordinator Agent**: Decision fusion and learning
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================
# Middleware for request logging
# ============================================================

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"

    # Add request ID to state
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"- {response.status_code} ({process_time:.2f}ms)"
    )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    return response


# ============================================================
# Core Endpoints
# ============================================================

@app.post(
    "/api/v1/evaluate",
    response_model=TransactionDecision,
    tags=["Transactions"],
    summary="Evaluate a transaction",
    description="Evaluate a transaction through the multi-agent RAG pipeline"
)
async def evaluate_transaction(
    request: TransactionRequest,
    db: Session = Depends(get_db)
) -> TransactionDecision:
    """
    Evaluate a transaction request through the complete multi-agent pipeline.

    The transaction flows through:
    1. Monitor Agent - Capture, Context, Feature extraction
    2. Evaluation Agent - Behavioral RAG and Policy RAG (parallel)
    3. Coordinator Agent - Fusion, Decision, Learning

    Returns a decision (ALLOW, CHALLENGE, or DENY) with evidence.
    """
    start_time = time.time()

    try:
        # Normalize user_id to title case (capitalize first letter) for
        # consistency
        request.user_id = request.user_id.strip().title()

        # Convert request to dict
        transaction_data = request.model_dump()

        # Step 0: Load user data if not already loaded (lazy loading)
        logger.info(f"Checking if data exists for user {request.user_id}")
        load_result = monitor_agent.load_user_if_needed(request.user_id)
        if load_result.get('success'):
            if load_result.get('already_loaded'):
                logger.debug(
                    f"User {
                        request.user_id} data already loaded ({
                        load_result.get('count')} transactions)")
            else:
                logger.info(
                    f"Loaded {
                        load_result.get(
                            'transactions_count',
                            0)} transactions for user {
                        request.user_id}")
        else:
            logger.warning(
                f"Could not load data for user {
                    request.user_id}: {
                    load_result.get('error')}")

        # Step 1: Monitor Agent processing
        logger.info(f"Processing transaction for user {request.user_id}")
        monitor_result = await monitor_agent.invoke(transaction_data)

        if not monitor_result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=f"Monitor Agent error: {monitor_result.get('error')}"
            )

        # Step 2: Evaluation Agent processing (parallel RAG)
        evaluation_input = {
            'enriched_transaction': monitor_result['enriched_transaction'],
            'user_id': monitor_result['user_id'],
            'user_context': monitor_result['user_context'],
            'extracted_features': monitor_result['extracted_features'],
            'embedding': monitor_result['embedding']
        }

        evaluation_result = await evaluation_agent.invoke(evaluation_input)

        if not evaluation_result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation Agent error: {
                    evaluation_result.get('error')}")

        # Step 3: Coordinator Agent decision
        coordinator_input = {
            'transaction_id': monitor_result['transaction_id'],
            'enriched_transaction': monitor_result['enriched_transaction'],
            'behavioral_assessment': evaluation_result['behavioral_assessment'],
            'policy_assessment': evaluation_result['policy_assessment'],
            'monitor_output': monitor_result,
            'evaluation_output': evaluation_result}

        coordinator_result = await coordinator_agent.invoke(coordinator_input)

        if not coordinator_result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Coordinator Agent error: {
                    coordinator_result.get('error')}")

        # Store transaction in database
        try:
            transaction_record = Transaction(
                transaction_id=monitor_result['transaction_id'],
                user_id=request.user_id,
                cc_num=request.cc_num,
                trans_date_trans_time=request.trans_date_trans_time,
                merchant=request.merchant,
                category=request.category,
                amt=request.amt,
                city=request.city,
                state=request.state,
                zip=request.zip,
                country=request.country,
                trans_num=request.trans_num
            )
            db.add(transaction_record)
            db.commit()
        except Exception as e:
            logger.warning(f"Could not store transaction: {e}")
            db.rollback()

        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000

        # Build response
        behavioral_assessment = evaluation_result['behavioral_assessment']
        policy_assessment = evaluation_result['policy_assessment']

        evidence = EvidenceDetail(
            behavioral_rag=BehavioralEvidence(
                similar_transactions=behavioral_assessment.get('similar_transactions', []),
                similarity_scores=[t.get('similarity', 0) for t in behavioral_assessment.get('similar_transactions', [])],
                deviations=behavioral_assessment.get('deviation_factors', []),
                statistical_features=behavioral_assessment.get('statistical_analysis')
            ),
            policy_rag=PolicyEvidence(
                retrieved_policies=[
                    f"{p.get('source', '')}: {p.get('excerpt', '')[:100]}"
                    for p in policy_assessment.get('retrieved_policies', [])
                ],
                violations=policy_assessment.get('violations', []),
                organizational_score=policy_assessment.get('organizational_score'),
                regulatory_score=policy_assessment.get('regulatory_score')
            )
        )

        return TransactionDecision(
            decision=coordinator_result['decision'],
            fused_score=coordinator_result.get('fused_score'),
            confidence=coordinator_result.get('confidence'),
            behavioral_score=coordinator_result.get('behavioral_score'),
            policy_score=coordinator_result.get('policy_score'),
            explanation=coordinator_result.get('explanation'),
            evidence=evidence,
            transaction_id=monitor_result['transaction_id'],
            timestamp=datetime.utcnow(),
            processing_time_ms=round(total_time_ms, 2),
            behavioral_assessment=behavioral_assessment,
            policy_assessment=policy_assessment
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/users/{user_id}/transactions",
    response_model=TransactionUploadResponse,
    tags=["Users"],
    summary="Upload user transaction history",
    description="Upload a CSV file containing historical transactions for a user"
)
async def upload_user_transactions(
    user_id: str,
    file: UploadFile = File(..., description="CSV file with transaction history"),
    db: Session = Depends(get_db)
) -> TransactionUploadResponse:
    """
    Upload historical transactions for a user from a CSV file.

    The system will:
    1. Parse and validate the CSV data
    2. Store transactions in the relational database
    3. Generate embeddings for each transaction
    4. Index embeddings in the behavioral vector database
    5. Update the user's profile statistics
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are accepted")

        # Read file content
        content = await file.read()

        # Parse transactions
        transactions = parse_csv_transactions(content, user_id)

        if not transactions:
            raise HTTPException(status_code=400,
                                detail="No valid transactions found in CSV")

        logger.info(
            f"Processing {
                len(transactions)} transactions for user {user_id}")

        # Store transactions in database
        transactions_stored = 0
        for txn in transactions:
            try:
                # Generate transaction ID
                timestamp = txn.get('trans_date_trans_time', datetime.utcnow())
                if not isinstance(timestamp, datetime):
                    timestamp = datetime.utcnow()

                txn_id = generate_transaction_id(user_id, timestamp)

                # Create transaction record
                transaction_record = Transaction(
                    transaction_id=txn_id,
                    user_id=user_id,
                    cc_num=txn.get('cc_num'),
                    trans_date_trans_time=timestamp,
                    merchant=txn.get('merchant'),
                    category=txn.get('category'),
                    amt=float(txn.get('amt', 0)),
                    city=txn.get('city'),
                    state=txn.get('state'),
                    zip=str(txn.get('zip', '')),
                    country=txn.get('country', 'US'),
                    trans_num=txn.get('trans_num'),
                    is_fraud=txn.get('is_fraud')
                )
                db.add(transaction_record)
                transactions_stored += 1
            except Exception as e:
                logger.warning(f"Could not store transaction: {e}")
                continue

        db.commit()

        # Generate embeddings and index in vector store
        transaction_ids = []
        transaction_dicts = []

        for i, txn in enumerate(transactions):
            txn_id = f"{user_id}_hist_{i}_{int(time.time())}"
            transaction_ids.append(txn_id)
            transaction_dicts.append(txn)

        embeddings_created = vector_store.add_transactions_batch(
            user_id=user_id,
            transactions=transaction_dicts,
            transaction_ids=transaction_ids
        )

        # Update user profile
        profile_stats = calculate_user_profile_stats(transactions)

        # Check if profile exists
        existing_profile = db.query(UserProfile).filter(
            UserProfile.user_id == user_id
        ).first()

        if existing_profile:
            # Update existing profile
            existing_profile.total_transactions = profile_stats['total_transactions']
            existing_profile.avg_amount = profile_stats['avg_amount']
            existing_profile.std_amount = profile_stats['std_amount']
            existing_profile.max_amount = profile_stats['max_amount']
            existing_profile.min_amount = profile_stats['min_amount']
            existing_profile.common_merchants = profile_stats['common_merchants']
            existing_profile.common_categories = profile_stats['common_categories']
            existing_profile.common_locations = profile_stats['common_locations']
            existing_profile.typical_hours = profile_stats['typical_hours']
            existing_profile.first_transaction_date = profile_stats['first_transaction_date']
            existing_profile.last_transaction_date = profile_stats['last_transaction_date']
            existing_profile.profile_updated_at = datetime.utcnow()
            existing_profile.risk_level = 'low'  # Default, will be updated with more data
        else:
            # Create new profile
            new_profile = UserProfile(
                user_id=user_id,
                total_transactions=profile_stats['total_transactions'],
                avg_amount=profile_stats['avg_amount'],
                std_amount=profile_stats['std_amount'],
                max_amount=profile_stats['max_amount'],
                min_amount=profile_stats['min_amount'],
                common_merchants=profile_stats['common_merchants'],
                common_categories=profile_stats['common_categories'],
                common_locations=profile_stats['common_locations'],
                typical_hours=profile_stats['typical_hours'],
                first_transaction_date=profile_stats['first_transaction_date'],
                last_transaction_date=profile_stats['last_transaction_date'],
                risk_level='low'
            )
            db.add(new_profile)

        db.commit()

        return TransactionUploadResponse(
            user_id=user_id,
            transactions_loaded=transactions_stored,
            embeddings_created=embeddings_created,
            profile_updated=True,
            status="success",
            message=f"Successfully loaded {transactions_stored} transactions and created {embeddings_created} embeddings"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction upload error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/policies",
    response_model=PolicyUploadResponse,
    tags=["Policies"],
    summary="Upload policy document",
    description="Upload a PDF policy document for RAG-based compliance checking"
)
async def upload_policy(file: UploadFile = File(...,
                                                description="PDF policy document"),
                        policy_type: PolicyType = Form(...,
                                                       description="Policy type: organizational or regulatory"),
                        db: Session = Depends(get_db)) -> PolicyUploadResponse:
    """
    Upload a policy document (PDF) for RAG-based compliance checking.

    The system will:
    1. Extract text from the PDF
    2. Chunk the content into ~400 token segments
    3. Generate embeddings for each chunk
    4. Index in the policy vector database

    Supports both organizational and regulatory policy types.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are accepted")

        # Save file temporarily
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # Load and index the PDF
            chunks_created = vector_store.load_policy_pdf(
                pdf_path=tmp_path,
                policy_type=policy_type.value
            )

            if chunks_created == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from PDF"
                )

            # Store metadata in database
            policy_doc = PolicyDocument(
                filename=file.filename,
                policy_type=policy_type.value,
                file_size=len(content),
                total_chunks=chunks_created,
                is_indexed=True,
                index_date=datetime.utcnow()
            )
            db.add(policy_doc)
            db.commit()

            logger.info(
                f"Indexed policy document: {
                    file.filename} ({chunks_created} chunks)")

            return PolicyUploadResponse(
                filename=file.filename,
                policy_type=policy_type,
                chunks_created=chunks_created,
                embeddings_indexed=chunks_created,
                status="success",
                message=f"Successfully indexed {chunks_created} chunks from {
                    file.filename}")

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy upload error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Submit feedback on a decision",
    description="Provide feedback on a transaction decision for adaptive learning"
)
async def submit_feedback(
    request: FeedbackRequest
) -> FeedbackResponse:
    """
    Submit feedback on a previous transaction decision.

    The system will:
    1. Record the feedback
    2. Calculate reward/penalty based on correctness
    3. Update adaptive parameters if needed
    4. Track metrics for system improvement
    """
    try:
        result = await coordinator_agent.submit_feedback(
            transaction_id=request.transaction_id,
            actual_outcome=request.actual_outcome.value,
            notes=request.notes
        )

        if not result.get('success'):
            raise HTTPException(
                status_code=404 if 'not found' in str(
                    result.get(
                        'error', '')).lower() else 500, detail=result.get(
                    'error', 'Feedback processing failed'))

        return FeedbackResponse(
            feedback_recorded=True, parameters_updated=result.get(
                'parameters_updated', False), reward_applied=result.get(
                'reward', 0), message=f"Feedback recorded. Decision was {
                'correct' if result['was_correct'] else 'incorrect'}.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/users/{user_id}/profile",
    response_model=UserProfileResponse,
    tags=["Users"],
    summary="Get user profile",
    description="Retrieve behavioral profile summary for a user"
)
async def get_user_profile(
    user_id: str,
    db: Session = Depends(get_db)
) -> UserProfileResponse:
    """
    Get the behavioral profile for a user.

    Returns transaction statistics, common patterns, and risk level.
    """
    profile = db.query(UserProfile).filter(
        UserProfile.user_id == user_id
    ).first()

    if not profile:
        raise HTTPException(status_code=404,
                            detail=f"Profile not found for user {user_id}")

    return UserProfileResponse(
        user_id=profile.user_id,
        total_transactions=profile.total_transactions,
        avg_amount=profile.avg_amount,
        std_amount=profile.std_amount,
        max_amount=profile.max_amount,
        min_amount=profile.min_amount,
        common_merchants=profile.common_merchants or [],
        common_categories=profile.common_categories or [],
        common_locations=profile.common_locations or [],
        typical_hours=profile.typical_hours or [],
        risk_level=profile.risk_level,
        first_transaction_date=profile.first_transaction_date,
        last_transaction_date=profile.last_transaction_date
    )


@app.get(
    "/api/v1/health",
    response_model=HealthStatus,
    tags=["System"],
    summary="Health check",
    description="Check system health and component status"
)
async def health_check() -> HealthStatus:
    """
    Health check endpoint for monitoring system status.

    Returns status of database, vector store, and agent readiness.
    """
    try:
        details = {}

        # Check database
        try:
            from sqlalchemy import text
            with db_manager.session_scope() as session:
                session.execute(text("SELECT 1"))
            details['database'] = 'connected'
        except Exception as e:
            details['database'] = f'error: {str(e)}'

        # Check vector store
        try:
            stats = vector_store.get_stats()
            details['vector_store'] = 'connected'
            details['policy_documents'] = stats['policy_documents']
        except Exception as e:
            details['vector_store'] = f'error: {str(e)}'

        # Check agents
        details['monitor_agent'] = 'ready' if monitor_agent.is_ready else 'not ready'
        details['evaluation_agent'] = 'ready' if evaluation_agent.is_ready else 'not ready'
        details['coordinator_agent'] = 'ready' if coordinator_agent.is_ready else 'not ready'

        # Check LLM
        details['llm_configured'] = settings.openai_api_key is not None

        # Determine overall status
        all_healthy = (
            details['database'] == 'connected' and
            details['vector_store'] == 'connected' and
            monitor_agent.is_ready and
            evaluation_agent.is_ready and
            coordinator_agent.is_ready
        )

        return HealthStatus(
            status='healthy' if all_healthy else 'degraded',
            message='All systems operational' if all_healthy else 'Some components have issues',
            details=details)

    except Exception as e:
        return HealthStatus(
            status='unhealthy',
            message=f'Health check failed: {str(e)}',
            details={'error': str(e)}
        )


@app.get(
    "/api/v1/metrics",
    response_model=SystemMetricsResponse,
    tags=["System"],
    summary="Get system metrics",
    description="Retrieve system performance and accuracy metrics"
)
async def get_metrics(
    db: Session = Depends(get_db)
) -> SystemMetricsResponse:
    """
    Get system performance metrics.

    Returns decision counts, accuracy metrics, and current parameters.
    """
    try:
        metrics = await coordinator_agent.get_system_metrics()

        # Get decision counts
        from sqlalchemy import func
        decision_counts = db.query(
            DecisionLog.decision,
            func.count(DecisionLog.id)
        ).group_by(DecisionLog.decision).all()

        counts = {'ALLOW': 0, 'CHALLENGE': 0, 'DENY': 0}
        total = 0
        for decision, count in decision_counts:
            counts[decision] = count
            total += count

        # Get user count
        total_users = db.query(UserProfile).count()

        # Get transaction count
        total_transactions = db.query(Transaction).count()

        # Get policy chunks from vector store
        vector_stats = vector_store.get_stats()
        policy_chunks = vector_stats.get('policy_documents', 0)

        return SystemMetricsResponse(
            total_decisions=total,
            allow_count=counts['ALLOW'],
            challenge_count=counts['CHALLENGE'],
            deny_count=counts['DENY'],
            total_users=total_users,
            total_transactions=total_transactions,
            policy_chunks=policy_chunks,
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            false_positive_rate=metrics.get('false_positive_rate'),
            false_negative_rate=metrics.get('false_negative_rate'),
            current_weights=metrics.get('current_weights', {}),
            current_thresholds=metrics.get('current_thresholds', {})
        )

    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["System"],
    summary="Root endpoint",
    description="Serve the GUARDIAN web interface"
)
async def root():
    """Root endpoint - serve the web interface."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "static")
    index_path = os.path.join(static_dir, "index.html")

    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
            <html>
                <body>
                    <h1>GUARDIAN Transaction Monitoring API</h1>
                    <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
                </body>
            </html>
        """)


# ============================================================
# Run with: uvicorn main:app --reload
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
