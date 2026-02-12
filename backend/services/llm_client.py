"""
LLM client for handling API calls to OpenAI.
Provides async and sync interfaces for RAG-based analysis.
"""

import json
import asyncio
from typing import Optional, Dict, Any, Union
from openai import OpenAI, AsyncOpenAI

from config.settings import settings


class LLMClient:
    """Handles LLM API calls for RAG analysis."""
    
    def __init__(self):
        """Initialize LLM client with OpenAI API."""
        if not settings.openai_api_key:
            print("⚠ Warning: OPENAI_API_KEY not set. LLM features will be limited.")
            self.client = None
            self.async_client = None
        else:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout
            )
            self.async_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout
            )
        
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: str = None
    ) -> str:
        """
        Generate text completion from prompt (synchronous).
        
        Args:
            prompt: User prompt
            max_tokens: Max tokens for response
            temperature: Temperature for sampling
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if self.client is None:
            return self._fallback_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._fallback_response(prompt)
    
    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        system_prompt: str = None
    ) -> str:
        """
        Generate text completion from prompt (asynchronous).
        
        Args:
            prompt: User prompt
            max_tokens: Max tokens for response
            temperature: Temperature for sampling
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        if self.async_client is None:
            return self._fallback_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM async generation error: {e}")
            return self._fallback_response(prompt)
    
    def analyze_behavioral_anomaly(
        self,
        current_transaction: Dict[str, Any],
        similar_transactions: list,
        similarity_scores: list
    ) -> Dict[str, Any]:
        """
        Analyze transaction for behavioral anomalies using RAG.
        
        Args:
            current_transaction: Current transaction to evaluate
            similar_transactions: Retrieved similar historical transactions
            similarity_scores: Similarity scores for retrieved transactions
            
        Returns:
            Dictionary with anomaly_score, confidence, explanation, deviations
        """
        prompt = f"""You are a fraud detection expert analyzing a transaction for anomalies.

CURRENT TRANSACTION:
{json.dumps(current_transaction, indent=2, default=str)}

SIMILAR HISTORICAL TRANSACTIONS (from user's history):
{json.dumps(similar_transactions, indent=2, default=str)}

SIMILARITY SCORES: {similarity_scores}

Analyze this transaction compared to the user's historical patterns. Consider:
1. Amount deviation from typical spending
2. Merchant/category novelty
3. Location changes (new cities, states, countries)
4. Time patterns (unusual hours, days)
5. Transaction velocity

Return a JSON object with:
- anomaly_score: 0.0 (completely normal) to 1.0 (highly anomalous)
- confidence: 0.0 to 1.0 (how confident you are)
- explanation: detailed reasoning (2-3 sentences)
- deviations: list of specific anomalous factors (e.g., ["amount", "location", "time"])

Return ONLY valid JSON, no other text."""

        system_prompt = "You are a fraud detection AI. Always respond with valid JSON only."
        
        response = self.generate(prompt, system_prompt=system_prompt)
        return self._parse_json_response(response, {
            "anomaly_score": 0.5,
            "confidence": 0.5,
            "explanation": "Unable to analyze - using default assessment",
            "deviations": []
        })
    
    async def analyze_behavioral_anomaly_async(
        self,
        current_transaction: Dict[str, Any],
        similar_transactions: list,
        similarity_scores: list,
        user_baseline: Dict[str, Any] = None,
        calculated_anomaly_score: float = None,
        anomaly_factors: list = None
    ) -> Dict[str, Any]:
        """
        Async version of behavioral anomaly analysis with RAG.
        
        Args:
            current_transaction: Current transaction to analyze
            similar_transactions: Retrieved similar historical transactions
            similarity_scores: Similarity scores for each retrieved transaction
            user_baseline: User's behavioral baseline (avg amounts, typical hours, distances)
            calculated_anomaly_score: Pre-calculated anomaly score from rule-based analysis
            anomaly_factors: List of identified anomaly factors
            
        Returns:
            Dictionary with anomaly_score, confidence, explanation, deviations
        """
        # Build user baseline context
        baseline_context = ""
        if user_baseline:
            typical_cities = ', '.join(user_baseline.get('typical_cities', [])[:3]) or 'N/A'
            typical_states = ', '.join(user_baseline.get('typical_states', [])) or 'N/A'
            avg_amt = user_baseline.get('avg_amount', 0)
            max_amt = user_baseline.get('max_amount', 0)
            curr_amt = current_transaction.get('amount', 0)
            
            # Calculate explicit comparison
            if avg_amt > 0:
                pct_of_avg = (curr_amt / avg_amt) * 100
                
                # Determine if amount is risky
                if curr_amt < avg_amt * 0.5:
                    amount_risk = "MUCH LOWER than normal - NOT suspicious"
                    amount_flag = "✅ NORMAL (below average is fine)"
                elif curr_amt > avg_amt * 2:
                    amount_risk = "MUCH HIGHER than normal - SUSPICIOUS"
                    amount_flag = "⚠️ RISKY (significantly above average)"
                elif curr_amt > max_amt:
                    amount_risk = "HIGHER than maximum ever seen - VERY SUSPICIOUS"
                    amount_flag = "🚨 VERY RISKY (above historical max)"
                else:
                    amount_risk = "Within normal range"
                    amount_flag = "✅ NORMAL"
                    
                amt_comparison = f"${curr_amt:.2f} is {pct_of_avg:.0f}% of user's average ${avg_amt:.2f} → {amount_risk}"
            else:
                amt_comparison = f"${curr_amt:.2f} (no baseline available)"
                amount_flag = "⚠️ UNKNOWN (no history)"
            
            baseline_context = f"""
USER'S BEHAVIORAL BASELINE (Established from Historical Data):
- Average transaction amount: ${avg_amt:.2f}
- Maximum transaction amount: ${max_amt:.2f}
- CURRENT transaction: {amt_comparison}
- AMOUNT ASSESSMENT: {amount_flag}
- Typical shopping hours: {user_baseline.get('typical_hours', [])}
- Common merchants: {', '.join(user_baseline.get('common_merchants', [])[:5])}
- Common categories: {', '.join(user_baseline.get('common_categories', [])[:3])}
- Typical cities: {typical_cities}
- Typical states: {typical_states}
"""
        
        # IMPROVEMENT 2: Move calculated info to the END as reference, not at the start
        # Let LLM analyze first, then validate with calculated score
        
        prompt = f"""You are a fraud detection expert analyzing a transaction using behavioral pattern analysis.

{baseline_context}
CURRENT TRANSACTION:
{json.dumps(current_transaction, indent=2, default=str)}

SIMILAR PAST TRANSACTIONS (Retrieved via RAG - K Most Similar):
{json.dumps(similar_transactions, indent=2, default=str)}
SIMILARITY SCORES: {similarity_scores}
(Note: Only transactions with similarity > 0.5 are included)

ANALYSIS TASK:
Compare the CURRENT transaction to the user's BEHAVIORAL BASELINE and SIMILAR PAST TRANSACTIONS.

⚠️ CRITICAL: Read the "AMOUNT ASSESSMENT" field above - it tells you if the amount is risky or not!

Consider these behavioral deviations:
1. **Amount Pattern**: 
   - If "AMOUNT ASSESSMENT" says "✅ NORMAL" → DO NOT flag amount as risky
   - If amount is BELOW average → it's LESS risky, not more risky
   - ONLY flag if amount is 2x average or above historical maximum
   
2. **Merchant Pattern**: Is this merchant in the user's "Common merchants" list above?

3. **Time Pattern**: Is the transaction hour in the user's "Typical shopping hours" list above?

4. **Location Pattern**: Is the city in "Typical cities" and state in "Typical states" above?

CRITICAL RULES:
- DO NOT invent universal rules (like "night = risky" or "amount seems high")
- ONLY compare against THIS SPECIFIC USER's baseline shown above
- If amount is BELOW user's average → it's safer, not riskier
- If hour IS IN typical hours list → it's normal for this user
- If city/state IS IN typical lists → it's normal location for this user
- If similar transactions list is EMPTY or has LOW similarity scores → lower your confidence
- Focus on CONCRETE DEVIATIONS from this user's patterns, not general fraud indicators

IMPROVEMENT 4: Your task is to:
1. Identify which specific aspects deviate from this user's normal behavior
2. Assess HOW SIGNIFICANT each deviation is based on the baseline data
3. Determine an anomaly score (0.0-1.0) based on deviation severity and frequency
4. Explain your reasoning in 2-3 sentences citing specific data points

For reference, here are statistical indicators detected:
- Statistical factors: {anomaly_factors if anomaly_factors else ['No significant statistical deviations detected']}
- Baseline anomaly estimate: {calculated_anomaly_score if calculated_anomaly_score is not None else 'Not calculated'}
(Use these as GUIDANCE only - your analysis should be based on the full context above)

Return a JSON object with:
- anomaly_score: 0.0 to 1.0 (YOUR assessment based on pattern analysis)
- confidence: 0.0 to 1.0 (based on data quality, number of similar transactions, and pattern clarity)
- explanation: 2-3 sentences explaining WHICH behaviors deviated and WHY that matters for THIS user
- deviations: list of specific deviations you identified (e.g., ["Amount 3x higher than normal", "New city"])

Return ONLY valid JSON, no other text."""

        system_prompt = "You are a fraud detection AI. Always respond with valid JSON only."
        
        # Debug: Log the prompt to see what's being sent
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"=== BEHAVIORAL ANALYSIS PROMPT ===")
        logger.info(f"User baseline avg: {user_baseline.get('avg_amount') if user_baseline else 'None'}")
        logger.info(f"Current amount: {current_transaction.get('amount')}")
        logger.info(f"Prompt (first 500 chars): {prompt[:500]}")
        
        response = await self.generate_async(prompt, system_prompt=system_prompt)
        return self._parse_json_response(response, {
            "anomaly_score": 0.5,
            "confidence": 0.5,
            "explanation": "Unable to analyze - using default assessment",
            "deviations": []
        })
    
    def analyze_policy_compliance(
        self,
        transaction: Dict[str, Any],
        policy_chunks: list,
        policy_type: str
    ) -> Dict[str, Any]:
        """
        Analyze transaction for policy compliance using RAG.
        
        Args:
            transaction: Transaction to evaluate
            policy_chunks: Retrieved relevant policy chunks
            policy_type: Type of policy (organizational/regulatory)
            
        Returns:
            Dictionary with compliance_score, violations, explanation
        """
        prompt = f"""You are a compliance officer evaluating a transaction against {policy_type} policies.

TRANSACTION TO EVALUATE:
{json.dumps(transaction, indent=2, default=str)}

RELEVANT POLICY EXCERPTS:
{json.dumps(policy_chunks, indent=2)}

Evaluate this transaction against the retrieved policies. Check for:
1. Amount limits (single transaction, daily, monthly)
2. Geographic restrictions (countries, regions)
3. Merchant category restrictions
4. Time-based rules
5. Regulatory requirements (sanctions, GDPR, PCI-DSS)

Return a JSON object with:
- compliance_score: 0.0 (fully compliant) to 1.0 (severe violation)
- confidence: 0.0 to 1.0
- violations: list of specific policy violations with citations
- explanation: detailed reasoning (2-3 sentences)

Return ONLY valid JSON, no other text."""

        system_prompt = "You are a compliance AI. Always respond with valid JSON only."
        
        response = self.generate(prompt, system_prompt=system_prompt)
        return self._parse_json_response(response, {
            "compliance_score": 0.0,
            "confidence": 0.5,
            "violations": [],
            "explanation": "Unable to analyze - using default assessment"
        })
    
    async def analyze_policy_compliance_async(
        self,
        transaction: Dict[str, Any],
        policy_chunks: list,
        policy_type: str
    ) -> Dict[str, Any]:
        """Async version of policy compliance analysis."""
        prompt = f"""You are a compliance officer evaluating a transaction against {policy_type} policies.

TRANSACTION TO EVALUATE:
{json.dumps(transaction, indent=2, default=str)}

RELEVANT POLICY EXCERPTS:
{json.dumps(policy_chunks, indent=2)}

Evaluate this transaction against the retrieved policies. Check for:
1. Amount limits (single transaction, daily, monthly)
2. Geographic restrictions (countries, regions)
3. Merchant category restrictions
4. Time-based rules
5. Regulatory requirements (sanctions, GDPR, PCI-DSS)

Return a JSON object with:
- compliance_score: 0.0 (fully compliant) to 1.0 (severe violation)
- confidence: 0.0 to 1.0
- violations: list of specific policy violations with citations
- explanation: detailed reasoning (2-3 sentences)

Return ONLY valid JSON, no other text."""

        system_prompt = "You are a compliance AI. Always respond with valid JSON only."
        
        response = await self.generate_async(prompt, system_prompt=system_prompt)
        return self._parse_json_response(response, {
            "compliance_score": 0.0,
            "confidence": 0.5,
            "violations": [],
            "explanation": "Unable to analyze - using default assessment"
        })
    
    def generate_decision_explanation(
        self,
        transaction: Dict[str, Any],
        behavioral_result: Dict[str, Any],
        policy_result: Dict[str, Any],
        decision: str,
        fused_score: float
    ) -> str:
        """
        Generate a human-readable explanation of the decision.
        
        Args:
            transaction: Transaction details
            behavioral_result: Behavioral analysis result
            policy_result: Policy analysis result
            decision: Final decision (ALLOW/CHALLENGE/DENY)
            fused_score: Combined risk score
            
        Returns:
            Human-readable explanation
        """
        prompt = f"""Generate a concise, professional explanation for a transaction security decision.

TRANSACTION:
- Amount: ${transaction.get('amt', 0):.2f}
- Merchant: {transaction.get('merchant', 'Unknown')}
- Location: {transaction.get('city', '')}, {transaction.get('state', '')}, {transaction.get('country', 'US')}
- Category: {transaction.get('category', 'Unknown')}

BEHAVIORAL ANALYSIS:
- Anomaly Score: {behavioral_result.get('anomaly_score', 0):.2f}
- Explanation: {behavioral_result.get('explanation', 'N/A')}
- Deviations: {behavioral_result.get('deviations', [])}

POLICY ANALYSIS:
- Compliance Score: {policy_result.get('compliance_score', 0):.2f}
- Violations: {policy_result.get('violations', [])}
- Explanation: {policy_result.get('explanation', 'N/A')}

DECISION: {decision}
RISK SCORE: {fused_score:.2f}

Write a 2-3 sentence professional explanation suitable for security audit logs. 
Be specific about the risk factors that led to this decision.
Do not use technical jargon - write for a human reviewer."""

        return self.generate(prompt, max_tokens=300)
    
    def _parse_json_response(
        self,
        response: str,
        default: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with fallback.
        
        Args:
            response: Raw LLM response
            default: Default values if parsing fails
            
        Returns:
            Parsed dictionary
        """
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                response = "\n".join(json_lines)
            
            return json.loads(response)
            
        except json.JSONDecodeError:
            # Try to find JSON object in response
            try:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
            
            print(f"Warning: Could not parse LLM response as JSON")
            return default
    
    def _fallback_response(self, prompt: str) -> str:
        """
        Provide a fallback response when LLM is unavailable.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Fallback response
        """
        if "anomaly" in prompt.lower():
            return json.dumps({
                "anomaly_score": 0.5,
                "confidence": 0.3,
                "explanation": "LLM unavailable - using default moderate risk assessment",
                "deviations": ["analysis_unavailable"]
            })
        elif "compliance" in prompt.lower() or "policy" in prompt.lower():
            return json.dumps({
                "compliance_score": 0.0,
                "confidence": 0.3,
                "violations": [],
                "explanation": "LLM unavailable - unable to verify policy compliance"
            })
        else:
            return "LLM service unavailable. Please configure OPENAI_API_KEY."


# Global LLM client instance
llm_client = LLMClient()
