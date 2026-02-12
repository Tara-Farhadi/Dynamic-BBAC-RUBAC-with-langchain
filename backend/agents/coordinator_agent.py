"""
Coordinator Agent - Decision fusion, execution, and adaptive learning.

Sub-agents:
- Fusion Sub-agent: Combines behavioral and policy scores into unified risk score
- Decision Sub-agent: Applies threshold logic for ALLOW/CHALLENGE/DENY decisions
- Learning Sub-agent: Logs decisions and adapts parameters from feedback
"""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from agents.base_agent import BaseAgent, BaseSubAgent
from services.llm_client import llm_client
from models.db_manager import db_manager
from models.database import (
    Transaction, DecisionLog, Feedback, AdaptiveParameters,
    SystemMetrics, UserProfile
)
from config.settings import settings


class FusionSubAgent(BaseSubAgent):
    """
    Fusion Sub-agent: Combines behavioral and policy scores into unified risk score.

    Uses weighted combination with adaptive weights that can be updated
    through the learning feedback loop.
    """

    def __init__(self):
        super().__init__("fusion", "coordinator_agent")
        self._behavioral_weight = settings.behavioral_weight
        self._policy_weight = settings.policy_weight

    def update_weights(self, behavioral_weight: float, policy_weight: float):
        """Update fusion weights from adaptive learning."""
        self._behavioral_weight = behavioral_weight
        self._policy_weight = policy_weight

    def get_weights(self) -> Dict[str, float]:
        """Get current fusion weights."""
        return {
            'behavioral_weight': self._behavioral_weight,
            'policy_weight': self._policy_weight
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse behavioral and policy scores.

        Args:
            input_data: Contains behavioral_assessment and policy_assessment

        Returns:
            Fused score with contribution breakdown
        """
        behavioral = input_data.get('behavioral_assessment', {})
        policy = input_data.get('policy_assessment', {})

        behavioral_score = behavioral.get('anomaly_score', 0.5)
        policy_score = policy.get('policy_score', 0.0)

        behavioral_confidence = behavioral.get('confidence', 0.5)
        policy_confidence = policy.get('confidence', 0.5)

        self.logger.debug(
            f"Fusing scores: behavioral={behavioral_score}, policy={policy_score}")

        # Check for regulatory override
        regulatory_score = policy.get('regulatory_score', 0.0)
        if regulatory_score >= 0.9:
            self.logger.info("Regulatory override triggered")
            return {
                'success': True,
                'fused_score': regulatory_score,
                'confidence': 0.95,
                'behavioral_contribution': 0.0,
                'policy_contribution': regulatory_score,
                'override_reason': 'regulatory_violation',
                'weights_used': self.get_weights()
            }

        # Normalize weights to sum to 1
        total_weight = self._behavioral_weight + self._policy_weight
        norm_behavioral_weight = self._behavioral_weight / total_weight
        norm_policy_weight = self._policy_weight / total_weight

        # Calculate weighted fused score
        fused_score = (
            behavioral_score * norm_behavioral_weight +
            policy_score * norm_policy_weight
        )

        # Calculate confidence (weighted by contribution)
        fused_confidence = (
            behavioral_confidence * norm_behavioral_weight +
            policy_confidence * norm_policy_weight
        )

        # Cap at 1.0
        fused_score = min(1.0, fused_score)
        fused_confidence = min(1.0, fused_confidence)

        return {
            'success': True,
            'fused_score': round(
                fused_score,
                3),
            'confidence': round(
                fused_confidence,
                3),
            'behavioral_contribution': round(
                behavioral_score *
                norm_behavioral_weight,
                3),
            'policy_contribution': round(
                policy_score *
                norm_policy_weight,
                3),
            'override_reason': None,
            'weights_used': self.get_weights()}


class DecisionSubAgent(BaseSubAgent):
    """
    Decision Sub-agent: Applies threshold logic for final decision.

    Decision rules:
    - fused_score < threshold_low → ALLOW
    - fused_score >= threshold_high → DENY
    - Otherwise → CHALLENGE (requires additional verification)
    """

    def __init__(self):
        super().__init__("decision", "coordinator_agent")
        self._threshold_low = settings.threshold_low
        self._threshold_high = settings.threshold_high

    def update_thresholds(self, threshold_low: float, threshold_high: float):
        """Update decision thresholds from adaptive learning."""
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high

    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return {
            'threshold_low': self._threshold_low,
            'threshold_high': self._threshold_high
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final decision based on fused score.

        Args:
            input_data: Contains fused_score, confidence, and assessments

        Returns:
            Final decision with explanation
        """
        fused_score = input_data.get('fused_score', 0.5)
        confidence = input_data.get('confidence', 0.5)
        override_reason = input_data.get('override_reason')

        self.logger.debug(f"Making decision: fused_score={fused_score}")

        # Determine decision
        if override_reason == 'regulatory_violation':
            decision = 'DENY'
            decision_reason = 'Regulatory violation detected - automatic denial'
        elif fused_score < self._threshold_low:
            decision = 'ALLOW'
            decision_reason = f'Risk score ({
                fused_score:.2f}) below threshold ({
                self._threshold_low})'
        elif fused_score >= self._threshold_high:
            decision = 'DENY'
            decision_reason = f'Risk score ({
                fused_score:.2f}) exceeds threshold ({
                self._threshold_high})'
        else:
            decision = 'CHALLENGE'
            decision_reason = f'Risk score ({
                fused_score:.2f}) in challenge range ({
                self._threshold_low}-{
                self._threshold_high})'

        return {
            'success': True,
            'decision': decision,
            'decision_reason': decision_reason,
            'thresholds_used': self.get_thresholds()
        }


class LearningSubAgent(BaseSubAgent):
    """
    Learning Sub-agent: Logs decisions and adapts parameters from feedback.

    Implements adaptive learning:
    - Records all decisions for analysis
    - Processes feedback to calculate rewards
    - Updates weights and thresholds based on outcomes
    - Tracks system metrics (precision, recall, F1)
    """

    def __init__(self):
        super().__init__("learning", "coordinator_agent")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log decision to database for future learning.

        Args:
            input_data: Complete decision data

        Returns:
            Logging confirmation
        """
        try:
            with db_manager.session_scope() as session:
                # Create decision log
                decision_log = DecisionLog(
                    transaction_id=input_data.get('transaction_id'),
                    decision=input_data.get('decision'),
                    fused_score=input_data.get('fused_score'),
                    confidence=input_data.get('confidence'),
                    behavioral_score=input_data.get('behavioral_score'),
                    policy_score=input_data.get('policy_score'),
                    explanation=input_data.get('explanation'),
                    evidence=input_data.get('evidence'),
                    monitor_output=input_data.get('monitor_output'),
                    evaluation_output=input_data.get('evaluation_output'),
                    coordinator_output=input_data.get('coordinator_output'),
                    behavioral_weight=input_data.get('behavioral_weight'),
                    policy_weight=input_data.get('policy_weight'),
                    processing_time_ms=input_data.get('processing_time_ms')
                )
                session.add(decision_log)

            self.logger.debug(
                f"Decision logged: {
                    input_data.get('transaction_id')}")
            return {'success': True, 'logged': True}

        except Exception as e:
            self.logger.error(f"Failed to log decision: {e}")
            return {'success': False, 'error': str(e)}

    async def process_feedback(
        self,
        transaction_id: str,
        actual_outcome: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process feedback and update adaptive parameters.

        Args:
            transaction_id: Transaction to provide feedback for
            actual_outcome: 'fraud' or 'legitimate'
            notes: Optional notes

        Returns:
            Feedback processing result with reward/penalty
        """
        try:
            with db_manager.session_scope() as session:
                # Find the decision log
                decision_log = session.query(DecisionLog).filter(
                    DecisionLog.transaction_id == transaction_id
                ).first()

                if not decision_log:
                    return {'success': False, 'error': 'Decision not found'}

                original_decision = decision_log.decision

                # Determine if decision was correct
                was_correct = self._evaluate_decision(
                    original_decision, actual_outcome)

                # Calculate reward
                reward = self._calculate_reward(
                    original_decision, actual_outcome, was_correct)

                # Create feedback record
                feedback = Feedback(
                    decision_log_id=decision_log.id,
                    transaction_id=transaction_id,
                    actual_outcome=actual_outcome,
                    notes=notes,
                    reward=reward,
                    was_correct=was_correct
                )
                session.add(feedback)

                # Update adaptive parameters if needed
                params_updated = False
                if not was_correct:
                    params_updated = await self._update_parameters(
                        session, original_decision, actual_outcome, reward
                    )

                # Update transaction with ground truth
                transaction = session.query(Transaction).filter(
                    Transaction.transaction_id == transaction_id
                ).first()
                if transaction:
                    transaction.is_fraud = (actual_outcome == 'fraud')

                return {
                    'success': True,
                    'was_correct': was_correct,
                    'reward': reward,
                    'parameters_updated': params_updated,
                    'original_decision': original_decision,
                    'actual_outcome': actual_outcome
                }

        except Exception as e:
            self.logger.error(f"Feedback processing error: {e}")
            return {'success': False, 'error': str(e)}

    def _evaluate_decision(self, decision: str, actual_outcome: str) -> bool:
        """Determine if the decision was correct."""
        if actual_outcome == 'fraud':
            return decision in ['DENY', 'CHALLENGE']
        else:
            return decision in ['ALLOW', 'CHALLENGE']

    def _calculate_reward(
            self,
            decision: str,
            actual_outcome: str,
            was_correct: bool) -> float:
        """Calculate reward/penalty based on outcome."""
        if was_correct:
            return float(settings.reward_correct)

        if actual_outcome == 'fraud' and decision == 'ALLOW':
            return float(settings.penalty_false_negative)
        elif actual_outcome == 'legitimate' and decision == 'DENY':
            return float(settings.penalty_false_positive)
        else:
            return float(settings.penalty_false_positive / 2)

    async def _update_parameters(
        self,
        session,
        decision: str,
        actual_outcome: str,
        reward: float
    ) -> bool:
        """Update adaptive parameters based on feedback."""
        try:
            # Get current parameters
            params = session.query(AdaptiveParameters).filter(
                AdaptiveParameters.is_active
            ).first()

            if not params:
                # Create initial parameters
                params = AdaptiveParameters(
                    behavioral_weight=settings.behavioral_weight,
                    policy_weight=settings.policy_weight,
                    threshold_low=settings.threshold_low,
                    threshold_high=settings.threshold_high
                )
                session.add(params)

            learning_rate = settings.learning_rate

            # Adjust weights based on error type
            if actual_outcome == 'fraud' and decision == 'ALLOW':
                params.behavioral_weight = min(
                    0.8, params.behavioral_weight + learning_rate)
                params.threshold_low = max(
                    0.1, params.threshold_low - learning_rate / 2)
                params.update_reason = "Increased behavioral weight after false negative"

            elif actual_outcome == 'legitimate' and decision == 'DENY':
                params.threshold_high = min(
                    0.9, params.threshold_high + learning_rate / 2)
                params.update_reason = "Relaxed thresholds after false positive"

            params.total_updates += 1
            params.last_update = datetime.utcnow()

            return True

        except Exception as e:
            self.logger.error(f"Parameter update error: {e}")
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return current system metrics."""
        try:
            with db_manager.session_scope() as session:
                # Get all feedback records
                feedbacks = session.query(Feedback).all()

                if not feedbacks:
                    return {
                        'total_feedback': 0,
                        'precision': None,
                        'recall': None,
                        'f1_score': None
                    }

                tp = fp = tn = fn = 0

                for fb in feedbacks:
                    decision_log = session.query(DecisionLog).filter(
                        DecisionLog.id == fb.decision_log_id
                    ).first()

                    if not decision_log:
                        continue

                    decision = decision_log.decision
                    outcome = fb.actual_outcome

                    if outcome == 'fraud':
                        if decision in ['DENY', 'CHALLENGE']:
                            tp += 1  # Correctly identified fraud
                        else:
                            fn += 1  # Missed fraud
                    else:  # legitimate
                        if decision == 'ALLOW':
                            tn += 1  # Correctly allowed
                        else:
                            fp += 1  # Wrongly denied

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else None
                recall = tp / (tp + fn) if (tp + fn) > 0 else None
                f1 = 2 * precision * recall / \
                    (precision + recall) if precision and recall else None
                fpr = fp / (fp + tn) if (fp + tn) > 0 else None
                fnr = fn / (fn + tp) if (fn + tp) > 0 else None

                return {
                    'total_feedback': len(feedbacks),
                    'true_positives': tp,
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'precision': round(precision, 3) if precision else None,
                    'recall': round(recall, 3) if recall else None,
                    'f1_score': round(f1, 3) if f1 else None,
                    'false_positive_rate': round(fpr, 3) if fpr else None,
                    'false_negative_rate': round(fnr, 3) if fnr else None
                }

        except Exception as e:
            self.logger.error(f"Metrics calculation error: {e}")
            return {'error': str(e)}


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent: Decision fusion, execution, and adaptive learning.

    Operates autonomously and can be invoked on-demand.
    Contains three sub-agents:
    - Fusion Sub-agent: Combines scores
    - Decision Sub-agent: Makes final decision
    - Learning Sub-agent: Logs and learns from feedback
    """

    def __init__(self):
        super().__init__("coordinator_agent", "Coordinator Agent")
        self.fusion_subagent = FusionSubAgent()
        self.decision_subagent = DecisionSubAgent()
        self.learning_subagent = LearningSubAgent()

    def initialize(self) -> bool:
        """Initialize the Coordinator Agent and load adaptive parameters."""
        self.logger.info("Initializing Coordinator Agent")

        # Load adaptive parameters from database
        try:
            with db_manager.session_scope() as session:
                params = session.query(AdaptiveParameters).filter(
                    AdaptiveParameters.is_active
                ).first()

                if params:
                    self.fusion_subagent.update_weights(
                        params.behavioral_weight,
                        params.policy_weight
                    )
                    self.decision_subagent.update_thresholds(
                        params.threshold_low,
                        params.threshold_high
                    )
                    self.logger.info(
                        f"Loaded adaptive parameters v{
                            params.version}")
        except Exception as e:
            self.logger.warning(f"Could not load adaptive parameters: {e}")

        self.is_ready = True
        return True

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process evaluation results and make final decision.

        Args:
            input_data: Contains transaction info and evaluation assessments

        Returns:
            Final decision with full explanation
        """
        start_time = datetime.utcnow()

        transaction = input_data.get('enriched_transaction', {})
        behavioral = input_data.get('behavioral_assessment', {})
        policy = input_data.get('policy_assessment', {})

        # Step 1: Fuse scores
        fusion_result = await self.fusion_subagent.execute({
            'behavioral_assessment': behavioral,
            'policy_assessment': policy
        })

        # Step 2: Make decision
        decision_result = await self.decision_subagent.execute({
            'fused_score': fusion_result['fused_score'],
            'confidence': fusion_result['confidence'],
            'override_reason': fusion_result.get('override_reason')
        })

        # Step 3: Generate explanation
        explanation = await self._generate_explanation(
            transaction, behavioral, policy,
            decision_result['decision'], fusion_result['fused_score']
        )

        # Compile evidence
        evidence = {
            'behavioral_rag': {
                'similar_transactions': behavioral.get('similar_transactions', []),
                'deviations': behavioral.get('deviation_factors', []),
                'statistical_analysis': behavioral.get('statistical_analysis')
            },
            'policy_rag': {
                'retrieved_policies': [
                    p.get('source', '') + ': ' + p.get('excerpt', '')[:100]
                    for p in policy.get('retrieved_policies', [])
                ],
                'violations': policy.get('violations', [])
            }
        }

        # Calculate processing time
        processing_time_ms = (
            datetime.utcnow() - start_time).total_seconds() * 1000

        # Prepare output
        output = {
            'success': True,
            'decision': decision_result['decision'],
            'fused_score': fusion_result['fused_score'],
            'confidence': fusion_result['confidence'],
            'behavioral_score': behavioral.get('anomaly_score', 0.5),
            'policy_score': policy.get('policy_score', 0.0),
            'explanation': explanation,
            'evidence': evidence,
            'transaction_id': input_data.get('transaction_id'),
            'processing_time_ms': round(processing_time_ms, 2),
            'weights_used': fusion_result['weights_used'],
            'thresholds_used': decision_result['thresholds_used'],
            # Include full assessments for analysis
            'behavioral_assessment': behavioral,
            'policy_assessment': policy
        }

        # Step 4: Log decision (async, non-blocking)
        asyncio.create_task(self.learning_subagent.execute({
            **output,
            'monitor_output': input_data.get('monitor_output'),
            'evaluation_output': input_data.get('evaluation_output'),
            'coordinator_output': {
                'fusion': fusion_result,
                'decision': decision_result
            },
            'behavioral_weight': fusion_result['weights_used']['behavioral_weight'],
            'policy_weight': fusion_result['weights_used']['policy_weight']
        }))

        return output

    async def _generate_explanation(
        self,
        transaction: Dict[str, Any],
        behavioral: Dict[str, Any],
        policy: Dict[str, Any],
        decision: str,
        fused_score: float
    ) -> str:
        """Generate human-readable explanation of the decision."""
        try:
            explanation = llm_client.generate_decision_explanation(
                transaction=transaction,
                behavioral_result=behavioral,
                policy_result=policy,
                decision=decision,
                fused_score=fused_score
            )
            return explanation
        except Exception as e:
            # Fallback explanation
            parts = []

            if decision == 'DENY':
                parts.append(
                    f"High-risk transaction detected (risk score: {fused_score:.2f}).")
            elif decision == 'CHALLENGE':
                parts.append(
                    f"Moderate risk transaction (risk score: {
                        fused_score:.2f}) requires verification.")
            else:
                parts.append(
                    f"Transaction approved (risk score: {
                        fused_score:.2f}).")

            # Add behavioral details
            if behavioral.get('deviation_factors'):
                parts.append(
                    f"Behavioral concerns: {
                        ', '.join(
                            behavioral['deviation_factors'])}.")

            # Add policy details
            if policy.get('violations'):
                parts.append(
                    f"Policy violations: {', '.join(policy['violations'][:3])}.")

            return " ".join(parts)

    async def submit_feedback(
        self,
        transaction_id: str,
        actual_outcome: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit feedback for a transaction decision.

        Args:
            transaction_id: Transaction ID
            actual_outcome: 'fraud' or 'legitimate'
            notes: Optional notes

        Returns:
            Feedback processing result
        """
        result = await self.learning_subagent.process_feedback(
            transaction_id, actual_outcome, notes
        )

        # Reload parameters if they were updated
        if result.get('parameters_updated'):
            self.initialize()

        return result

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        metrics = await self.learning_subagent.get_metrics()
        metrics['current_weights'] = self.fusion_subagent.get_weights()
        metrics['current_thresholds'] = self.decision_subagent.get_thresholds()
        return metrics

    def decide(
        self,
        behavioral_assessment: Dict[str, Any],
        policy_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronous decision method for compatibility.

        Args:
            behavioral_assessment: From Evaluation Agent
            policy_assessment: From Evaluation Agent

        Returns:
            Decision output
        """
        import asyncio

        # Run async process in event loop
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.process({
                'behavioral_assessment': behavioral_assessment,
                'policy_assessment': policy_assessment
            }))
            return result
        finally:
            loop.close()


# Singleton instance for on-demand invocation
coordinator_agent = CoordinatorAgent()
