"""
Evaluation Agent - Behavioral analysis and policy compliance checking using RAG.

Sub-agents:
- Behavioral Sub-agent: Uses RAG to analyze behavioral patterns and detect anomalies
- Policy Sub-agent: Uses RAG to validate against organizational and regulatory policies
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from agents.base_agent import BaseAgent, BaseSubAgent
from services.embedding import embedding_service
from services.vector_store import vector_store
from services.llm_client import llm_client
from config.settings import settings


class BehavioralSubAgent(BaseSubAgent):
    """
    Behavioral Sub-agent: Uses RAG to establish behavioral baselines and detect anomalies.

    RAG Workflow:
    1. Generate embedding for current transaction
    2. Query vector DB for K similar historical transactions
    3. Retrieve transaction details for similar results
    4. Send to LLM for anomaly analysis
    5. Return anomaly score, confidence, and explanation
    """

    def __init__(self):
        super().__init__("behavioral", "evaluation_agent")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform behavioral RAG analysis.

        Args:
            input_data: Contains transaction, embedding, user_id, and context

        Returns:
            BehavioralAssessment with anomaly_score, confidence, explanation
        """
        transaction = input_data.get('enriched_transaction', {})
        embedding = input_data.get('embedding', [])
        user_id = input_data.get('user_id')
        user_context = input_data.get('user_context', {})
        features = input_data.get('extracted_features', {})

        self.logger.debug(
            f"Performing behavioral RAG analysis for user {user_id}")

        # Check if user has historical data
        has_history = user_context.get('has_history', False)

        if not has_history:
            # No history - flag as potentially risky but uncertain
            return {
                'success': True,
                'anomaly_score': 0.5,
                'confidence': 0.3,
                'explanation': "No historical data available for this user. Cannot establish behavioral baseline.",
                'similar_transactions': [],
                'deviation_factors': ['no_history'],
                'statistical_analysis': features}

        # Step 1: Query vector DB for similar transactions
        similar_results = vector_store.search_similar_transactions(
            user_id=user_id,
            query_embedding=embedding,
            n_results=settings.behavioral_k_results
        )

        # Extract similar transactions and similarity scores
        similar_transactions = []
        similarity_scores = []
        MIN_SIMILARITY_THRESHOLD = 0.5

        if similar_results.get(
                'documents') and similar_results['documents'][0]:
            documents = similar_results['documents'][0]
            metadatas = similar_results.get('metadatas', [[]])[0]
            distances = similar_results.get('distances', [[]])[0]

            for i, (doc, meta, dist) in enumerate(
                    zip(documents, metadatas, distances)):
                # Convert L2 distance to similarity score (0-1 range)
                similarity = max(0, 1 - (dist / 2))

                # Filter out low-quality matches
                if similarity < MIN_SIMILARITY_THRESHOLD:
                    continue

                similarity_scores.append(round(similarity, 3))

                similar_transactions.append({
                    'index': i,
                    'description': doc,
                    'metadata': meta,
                    'similarity': round(similarity, 3)
                })

        # Step 2: Prepare user's behavioral baseline for context
        user_profile = user_context.get('profile', {})
        behavioral_baseline = {
            'avg_amount': user_profile.get('avg_amount', 0),
            'max_amount': user_profile.get('max_amount', 0),
            'typical_hours': user_profile.get('typical_hours', []),
            'common_merchants': user_profile.get('common_merchants', []),
            'common_categories': user_profile.get('common_categories', []),
            'typical_cities': [loc.get('city', '') for loc in user_profile.get('common_locations', [])[:5]],
            'typical_states': list(set([loc.get('state', '') for loc in user_profile.get('common_locations', [])])),
            'avg_merchant_distance': user_profile.get('avg_shopping_distance', 0),
            'max_merchant_distance': user_profile.get('max_shopping_distance', 0)
        }

        # Step 3: Prepare current transaction for LLM analysis
        transaction_for_llm = {
            'user_id': user_id,
            'amount': transaction.get('amt'),
            'merchant': transaction.get('merchant'),
            'city': transaction.get('city'),
            'state': transaction.get('state'),
            'hour': transaction.get(
                'temporal_features',
                {}).get('hour'),
            'day_of_week': transaction.get(
                'temporal_features',
                {}).get('day_of_week')}

        # Add geographic comparisons
        statistical_analysis = {
            'amount_zscore': features.get(
                'amount',
                {}).get('amount_zscore'),
            'is_new_city': features.get(
                'geographic',
                {}).get(
                'is_new_city',
                False),
            'is_new_state': features.get(
                'geographic',
                    {}).get(
                        'is_new_state',
                        False),
            'is_new_merchant': features.get(
                'merchant',
                {}).get(
                'is_new_merchant',
                False),
            'is_new_category': features.get(
                'merchant',
                {}).get(
                'is_new_category',
                False)}

        # Calculate adaptive statistical anomaly factors
        curr_amt = transaction.get('amt', 0)
        avg_amt = behavioral_baseline.get('avg_amount', 0)
        max_amt = behavioral_baseline.get('max_amount', 0)
        curr_hour = transaction.get('temporal_features', {}).get('hour')
        typical_hours = behavioral_baseline.get('typical_hours', [])
        curr_city = transaction.get('city', '')
        typical_cities = behavioral_baseline.get('typical_cities', [])

        # Use statistical Z-score for adaptive thresholds
        amount_zscore = features.get('amount', {}).get('amount_zscore', 0)

        # Calculate anomaly components
        anomaly_factors = []

        # Amount anomaly analysis
        if avg_amt > 0:
            if curr_amt > max_amt:
                pct_over = ((curr_amt - max_amt) / max_amt) * 100
                if pct_over > 50:
                    anomaly_factors.append(
                        ('amount', 0.5, f"Amount ${
                            curr_amt:.2f} is {
                            pct_over:.0f}% above historical max ${
                            max_amt:.2f}"))
                else:
                    anomaly_factors.append(
                        ('amount',
                         0.3,
                         f"Amount ${
                             curr_amt:.2f} exceeds historical max ${
                             max_amt:.2f} by {
                             pct_over:.0f}%"))
            elif amount_zscore > 2.0:
                anomaly_factors.append(
                    ('amount', 0.35, f"Amount ${
                        curr_amt:.2f} is {
                        amount_zscore:.1f} std devs above average (Z-score: {
                        amount_zscore:.2f})"))
            elif amount_zscore > 1.5:
                anomaly_factors.append(
                    ('amount', 0.25, f"Amount ${curr_amt:.2f} is moderately high (Z-score: {amount_zscore:.2f})"))
            elif amount_zscore < -2.0:
                anomaly_factors.append(
                    ('amount', 0.15, f"Amount ${
                        curr_amt:.2f} is unusually low (Z-score: {
                        amount_zscore:.2f}) - possible fraud testing"))

        # Time-based anomaly
        if typical_hours and curr_hour is not None and curr_hour not in typical_hours:
            anomaly_factors.append(
                ('time', 0.2, f"Hour {curr_hour} not in typical hours {typical_hours}"))

        # Location-based anomaly
        if typical_cities and curr_city and curr_city not in typical_cities:
            anomaly_factors.append(
                ('location', 0.25, f"City '{curr_city}' not in typical cities {typical_cities[:3]}"))

        # Merchant anomaly
        if statistical_analysis.get('is_new_merchant'):
            anomaly_factors.append(
                ('merchant', 0.15, f"New merchant not seen before"))

        # Calculate base anomaly score from factors
        if anomaly_factors:
            base_anomaly = min(1.0, sum(f[1] for f in anomaly_factors))
        else:
            base_anomaly = 0.1  # Baseline low risk

        # Step 5: LLM analysis for explanation (with calculated score as
        # guidance)
        llm_result = await llm_client.analyze_behavioral_anomaly_async(
            current_transaction=transaction_for_llm,
            similar_transactions=similar_transactions,
            similarity_scores=similarity_scores,
            user_baseline=behavioral_baseline,
            calculated_anomaly_score=base_anomaly,
            anomaly_factors=[f[2] for f in anomaly_factors]
        )

        # Use calculated score, but let LLM adjust slightly based on context
        llm_anomaly = llm_result.get('anomaly_score', base_anomaly)
        # Blend: 70% calculated, 30% LLM
        final_anomaly = (base_anomaly * 0.7) + (llm_anomaly * 0.3)

        confidence = llm_result.get('confidence', 0.5)

        # Only adjust if no similar transactions found (edge case)
        if not similar_transactions:
            # Lower confidence, but keep LLM's score
            confidence = min(1.0, confidence * 0.7)

        return {
            'success': True,
            'anomaly_score': round(min(1.0, final_anomaly), 2),
            'confidence': round(confidence, 2),
            'explanation': llm_result.get('explanation', 'Behavioral analysis completed'),
            'similar_transactions': similar_transactions,
            'deviation_factors': [f[2] for f in anomaly_factors],
            'statistical_analysis': statistical_analysis,
            'calculated_base_anomaly': round(base_anomaly, 2)
        }


class PolicySubAgent(BaseSubAgent):
    """
    Policy Sub-agent: Uses RAG to validate transactions against policies.

    RAG Workflow:
    1. Create query from transaction context
    2. Query organizational policy collection
    3. Query regulatory policy collection
    4. Use LLM to evaluate compliance
    5. Fuse organizational and regulatory scores
    """

    def __init__(self):
        super().__init__("policy", "evaluation_agent")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform policy RAG analysis.

        Args:
            input_data: Contains transaction and context

        Returns:
            PolicyAssessment with compliance scores and violations
        """
        transaction = input_data.get('enriched_transaction', {})
        features = input_data.get('extracted_features', {})

        self.logger.debug("Performing policy RAG analysis")

        # Check if we have any policies indexed
        policy_count = vector_store.get_policy_count()

        if policy_count == 0:
            # No policies - assume compliant
            return {
                'success': True,
                'policy_score': 0.0,
                'confidence': 0.3,
                'explanation': "No policy documents indexed. Unable to validate compliance.",
                'organizational_score': 0.0,
                'regulatory_score': 0.0,
                'violations': [],
                'retrieved_policies': []}

        # Create policy query from transaction context
        policy_query = self._create_policy_query(transaction, features)

        # Step 1: Query organizational policies
        org_results = vector_store.search_policies(
            query_text=policy_query,
            policy_type="organizational",
            n_results=settings.policy_k_results
        )

        org_chunks = self._extract_policy_chunks(org_results)

        # Step 2: Query regulatory policies
        reg_results = vector_store.search_policies(
            query_text=policy_query,
            policy_type="regulatory",
            n_results=settings.policy_k_results
        )

        reg_chunks = self._extract_policy_chunks(reg_results)

        # Prepare transaction for LLM
        transaction_for_llm = {
            'amount': transaction.get('amt'),
            'merchant': transaction.get('merchant'),
            'category': transaction.get('category'),
            'city': transaction.get('city'),
            'state': transaction.get('state'),
            'country': transaction.get('country'),
            'timestamp': str(
                transaction.get('trans_date_trans_time')),
            'is_international': features.get(
                'location',
                {}).get('is_international'),
            'amount_ratio_to_max': features.get(
                'amount',
                {}).get('ratio_to_max')}

        # Step 3: LLM analysis for organizational policies
        org_assessment = {
            'compliance_score': 0.0,
            'violations': [],
            'explanation': 'No organizational policies found'}
        if org_chunks:
            org_assessment = await llm_client.analyze_policy_compliance_async(
                transaction=transaction_for_llm,
                policy_chunks=org_chunks,
                policy_type="organizational"
            )

        # Step 4: LLM analysis for regulatory policies
        reg_assessment = {
            'compliance_score': 0.0,
            'violations': [],
            'explanation': 'No regulatory policies found'}
        if reg_chunks:
            reg_assessment = await llm_client.analyze_policy_compliance_async(
                transaction=transaction_for_llm,
                policy_chunks=reg_chunks,
                policy_type="regulatory"
            )

        # Step 5: Fuse scores with regulatory precedence
        organizational_score = org_assessment.get('compliance_score', 0.0)
        regulatory_score = reg_assessment.get('compliance_score', 0.0)

        # Regulatory precedence rule: if regulatory_score >= 0.8, it overrides
        if regulatory_score >= 0.8:
            final_policy_score = regulatory_score
        else:
            # Otherwise, take weighted max
            final_policy_score = max(
                organizational_score,
                regulatory_score * 1.2  # Regulatory violations weighted higher
            )

        final_policy_score = min(1.0, final_policy_score)

        # Combine violations
        all_violations = []
        for v in org_assessment.get('violations', []):
            all_violations.append(f"[ORG] {v}")
        for v in reg_assessment.get('violations', []):
            all_violations.append(f"[REG] {v}")

        # Combine retrieved policies for citation
        retrieved_policies = []
        for chunk in org_chunks:
            retrieved_policies.append({
                'source': chunk.get('source', 'unknown'),
                'type': 'organizational',
                'excerpt': chunk.get('text', '')[:200]
            })
        for chunk in reg_chunks:
            retrieved_policies.append({
                'source': chunk.get('source', 'unknown'),
                'type': 'regulatory',
                'excerpt': chunk.get('text', '')[:200]
            })

        # Generate combined explanation
        explanation = f"Organizational: {
            org_assessment.get(
                'explanation', 'N/A')}. "
        explanation += f"Regulatory: {
            reg_assessment.get(
                'explanation', 'N/A')}"

        # Calculate confidence
        confidence = 0.5
        if org_chunks or reg_chunks:
            confidence = 0.8
        if regulatory_score >= 0.8:
            confidence = 0.95  # High confidence on regulatory violations

        return {
            'success': True,
            'policy_score': round(final_policy_score, 2),
            'confidence': round(confidence, 2),
            'explanation': explanation,
            'organizational_score': round(organizational_score, 2),
            'regulatory_score': round(regulatory_score, 2),
            'violations': all_violations,
            'retrieved_policies': retrieved_policies
        }

    def _create_policy_query(
            self, transaction: Dict[str, Any], features: Dict[str, Any]) -> str:
        """Create a query string for policy retrieval."""
        parts = []

        # Amount-related
        amt = transaction.get('amt', 0)
        if amt > 5000:
            parts.append(f"large transaction ${amt} amount limit")
        if amt > 10000:
            parts.append("high value transaction reporting threshold")

        # Location-related
        country = transaction.get('country', 'US')
        if country != 'US':
            parts.append(f"international transaction {country} cross-border")

        # Check for potentially sanctioned countries
        sanctioned_keywords = [
            'RU',
            'Russia',
            'IR',
            'Iran',
            'KP',
            'North Korea',
            'SY',
            'Syria']
        if any(kw.lower() in str(transaction).lower()
               for kw in sanctioned_keywords):
            parts.append("sanctions restricted country OFAC prohibited")

        # Category-related
        category = transaction.get('category', '')
        if category:
            parts.append(f"{category} merchant category restriction")

        # Time-related
        if features.get('temporal', {}).get('is_night'):
            parts.append("late night transaction unusual hours")

        # Velocity-related
        if features.get('velocity', {}).get('velocity_score', 0) > 0.5:
            parts.append("high velocity multiple transactions limit")

        # Default query if nothing specific
        if not parts:
            parts.append("transaction approval policy limits restrictions")

        return " ".join(parts)

    def _extract_policy_chunks(
            self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract policy chunks from search results."""
        chunks = []

        if not results.get('documents') or not results['documents'][0]:
            return chunks

        documents = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0]

        for doc, meta in zip(documents, metadatas):
            chunks.append({
                'text': doc,
                'source': meta.get('source', 'unknown'),
                'page': meta.get('page', 1),
                'policy_type': meta.get('policy_type', 'unknown')
            })

        return chunks


class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent: Behavioral analysis and policy compliance checking using RAG.

    Operates autonomously and can be invoked on-demand.
    Contains two sub-agents that can run in parallel:
    - Behavioral Sub-agent: RAG-based anomaly detection
    - Policy Sub-agent: RAG-based policy compliance validation
    """

    def __init__(self):
        super().__init__("evaluation_agent", "Evaluation Agent")
        self.behavioral_subagent = BehavioralSubAgent()
        self.policy_subagent = PolicySubAgent()

    def initialize(self) -> bool:
        """Initialize the Evaluation Agent."""
        self.logger.info("Initializing Evaluation Agent")
        self.is_ready = True
        return True

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process transaction through behavioral and policy sub-agents.
        Runs both analyses in parallel for efficiency.

        Args:
            input_data: MonitorAgentOutput data

        Returns:
            EvaluationAgentOutput with both assessments
        """
        # Run both sub-agents in parallel
        behavioral_task = self.behavioral_subagent.execute(input_data)
        policy_task = self.policy_subagent.execute(input_data)

        behavioral_result, policy_result = await asyncio.gather(
            behavioral_task,
            policy_task
        )

        # Check for errors
        if not behavioral_result.get('success'):
            self.logger.warning("Behavioral analysis failed, using default")
            behavioral_result = {
                'anomaly_score': 0.5,
                'confidence': 0.3,
                'explanation': 'Behavioral analysis failed',
                'similar_transactions': [],
                'deviation_factors': ['analysis_error']
            }

        if not policy_result.get('success'):
            self.logger.warning("Policy analysis failed, using default")
            policy_result = {
                'policy_score': 0.0,
                'confidence': 0.3,
                'explanation': 'Policy analysis failed',
                'organizational_score': 0.0,
                'regulatory_score': 0.0,
                'violations': [],
                'retrieved_policies': []
            }

        return {
            'success': True,
            'behavioral_assessment': {
                'anomaly_score': behavioral_result.get('anomaly_score', 0.5),
                'confidence': behavioral_result.get('confidence', 0.5),
                'explanation': behavioral_result.get('explanation', ''),
                'similar_transactions': behavioral_result.get('similar_transactions', []),
                'deviation_factors': behavioral_result.get('deviation_factors', []),
                'statistical_analysis': behavioral_result.get('statistical_analysis')
            },
            'policy_assessment': {
                'policy_score': policy_result.get('policy_score', 0.0),
                'confidence': policy_result.get('confidence', 0.5),
                'explanation': policy_result.get('explanation', ''),
                'organizational_score': policy_result.get('organizational_score', 0.0),
                'regulatory_score': policy_result.get('regulatory_score', 0.0),
                'violations': policy_result.get('violations', []),
                'retrieved_policies': policy_result.get('retrieved_policies', [])
            }
        }

    async def behavioral_analysis(
            self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run only behavioral analysis (for standalone invocation).

        Args:
            input_data: Transaction data with context

        Returns:
            Behavioral assessment
        """
        return await self.behavioral_subagent.execute(input_data)

    async def policy_analysis(
            self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run only policy analysis (for standalone invocation).

        Args:
            input_data: Transaction data

        Returns:
            Policy assessment
        """
        return await self.policy_subagent.execute(input_data)


# Singleton instance for on-demand invocation
evaluation_agent = EvaluationAgent()
