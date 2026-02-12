"""
Monitor Agent - Perception and feature extraction from incoming transactions.

The Monitor Agent runs continuously and handles:
1. Auto-loading existing CSV files on startup
2. Watching the uploads folder for new/changed files
3. Processing incoming transactions for evaluation

Sub-agents:
- Capture Sub-agent: Normalizes and enriches transactions
- Context Sub-agent: Retrieves user's recent transaction history
- Feature Sub-agent: Extracts statistical features and generates embeddings
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import os
import hashlib
import threading
import time as time_module

from agents.base_agent import BaseAgent, BaseSubAgent
from services.embedding import embedding_service
from services.vector_store import vector_store
from models.db_manager import db_manager
from models.database import Transaction, UserProfile
from utils.helpers import (
    normalize_transaction,
    extract_temporal_features,
    extract_amount_features,
    extract_location_features,
    extract_merchant_features,
    calculate_velocity_features,
    calculate_combined_risk_features,
    generate_transaction_id,
    parse_csv_transactions,
    calculate_user_profile_stats
)
from config.settings import settings


class CaptureSubAgent(BaseSubAgent):
    """
    Capture Sub-agent: Normalizes and enriches transactions with auxiliary signals.
    """

    def __init__(self):
        super().__init__("capture", "monitor_agent")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and enrich the transaction.

        Args:
            input_data: Raw transaction data

        Returns:
            Enriched transaction with auxiliary signals
        """
        self.logger.debug("Capturing and normalizing transaction")

        # Normalize the transaction
        normalized = normalize_transaction(input_data)

        # Generate transaction ID if not present
        if 'transaction_id' not in normalized or not normalized['transaction_id']:
            timestamp = normalized.get(
                'trans_date_trans_time', datetime.utcnow())
            user_id = normalized.get('user_id', 'unknown')
            normalized['transaction_id'] = generate_transaction_id(
                user_id, timestamp)

        # Extract temporal features
        timestamp = normalized.get('trans_date_trans_time', datetime.utcnow())
        temporal_features = extract_temporal_features(timestamp)

        # Enrich with auxiliary signals
        enriched = {
            **normalized,
            'temporal_features': temporal_features,
            'capture_timestamp': datetime.utcnow().isoformat()
        }

        return {
            'success': True,
            'enriched_transaction': enriched
        }


class ContextSubAgent(BaseSubAgent):
    """
    Context Sub-agent: Retrieves user's recent transaction history from the database.
    """

    def __init__(self):
        super().__init__("context", "monitor_agent")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve user context including recent transactions.

        Args:
            input_data: Contains user_id and optional time window

        Returns:
            User context with historical data
        """
        user_id = input_data.get('user_id')
        if not user_id:
            return {'success': False, 'error': 'user_id required'}

        self.logger.debug(f"Retrieving context for user: {user_id}")

        # Get user profile from database
        user_profile = None
        recent_transactions = []

        try:
            with db_manager.session_scope() as session:
                # Get user profile
                profile = session.query(UserProfile).filter(
                    UserProfile.user_id == user_id
                ).first()

                if profile:
                    user_profile = {
                        'user_id': profile.user_id,
                        'total_transactions': profile.total_transactions,
                        'avg_amount': profile.avg_amount,
                        'std_amount': profile.std_amount,
                        'max_amount': profile.max_amount,
                        'min_amount': profile.min_amount,
                        'common_merchants': profile.common_merchants or [],
                        'common_categories': profile.common_categories or [],
                        'common_locations': profile.common_locations or [],
                        'typical_hours': profile.typical_hours or [],
                        'risk_level': profile.risk_level
                    }

                # Get recent transactions (last N days)
                cutoff_date = datetime.utcnow() - timedelta(days=settings.transaction_history_days)
                transactions = session.query(Transaction).filter(
                    Transaction.user_id == user_id,
                    Transaction.trans_date_trans_time >= cutoff_date).order_by(
                    Transaction.trans_date_trans_time.desc()).limit(100).all()

                for txn in transactions:
                    recent_transactions.append({
                        'transaction_id': txn.transaction_id,
                        'amt': txn.amt,
                        'merchant': txn.merchant,
                        'category': txn.category,
                        'city': txn.city,
                        'state': txn.state,
                        'country': txn.country,
                        'trans_date_trans_time': txn.trans_date_trans_time
                    })

        except Exception as e:
            self.logger.warning(f"Database error: {e}")

        # Get transaction count from vector store
        vector_count = vector_store.get_user_transaction_count(user_id)

        return {
            'success': True,
            'user_id': user_id,
            'user_profile': user_profile,
            'recent_transactions': recent_transactions,
            'historical_transaction_count': vector_count,
            'has_history': vector_count > 0 or len(recent_transactions) > 0
        }


class FeatureSubAgent(BaseSubAgent):
    """
    Feature Sub-agent: Extracts statistical features and generates embeddings.
    """

    def __init__(self):
        super().__init__("feature", "monitor_agent")

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features and generate embedding for the transaction.

        Args:
            input_data: Contains enriched_transaction and user_context

        Returns:
            Features and embedding
        """
        enriched_txn = input_data.get('enriched_transaction', {})
        user_context = input_data.get('user_context', {})

        self.logger.debug("Extracting features and generating embedding")

        # Extract amount features
        historical_amounts = []
        if user_context.get('user_profile'):
            profile = user_context['user_profile']
            # Reconstruct approximate historical amounts
            avg = profile.get('avg_amount', 0)
            std = profile.get('std_amount', 0)
            if avg > 0:
                historical_amounts = [avg - std, avg, avg + std]

        for txn in user_context.get('recent_transactions', []):
            if txn.get('amt'):
                historical_amounts.append(txn['amt'])

        amount_features = extract_amount_features(
            enriched_txn.get('amt', 0),
            historical_amounts
        )

        # Extract location features
        historical_locations = []
        if user_context.get('user_profile'):
            historical_locations = user_context['user_profile'].get(
                'common_locations', [])
        for txn in user_context.get('recent_transactions', []):
            historical_locations.append({
                'city': txn.get('city', ''),
                'state': txn.get('state', ''),
                'country': txn.get('country', 'US')
            })

        location_features = extract_location_features(
            enriched_txn.get('city', ''),
            enriched_txn.get('state', ''),
            enriched_txn.get('country', 'US'),
            historical_locations
        )

        # Extract merchant features
        historical_merchants = []
        historical_categories = []
        if user_context.get('user_profile'):
            historical_merchants = user_context['user_profile'].get(
                'common_merchants', [])
            historical_categories = user_context['user_profile'].get(
                'common_categories', [])
        for txn in user_context.get('recent_transactions', []):
            if txn.get('merchant'):
                historical_merchants.append(txn['merchant'])
            if txn.get('category'):
                historical_categories.append(txn['category'])

        merchant_features = extract_merchant_features(
            enriched_txn.get('merchant', ''),
            enriched_txn.get('category', ''),
            historical_merchants,
            historical_categories
        )

        # Calculate velocity features
        current_timestamp = enriched_txn.get(
            'trans_date_trans_time', datetime.utcnow())
        velocity_features = calculate_velocity_features(
            current_timestamp,
            user_context.get('recent_transactions', [])
        )

        # Calculate combined risk features
        temporal_features = enriched_txn.get('temporal_features', {})
        combined_features = calculate_combined_risk_features(
            temporal_features,
            amount_features,
            location_features,
            merchant_features,
            velocity_features
        )

        # Generate embedding
        embedding = embedding_service.embed_transaction(enriched_txn)

        # Compile all features
        all_features = {
            'temporal': temporal_features,
            'amount': amount_features,
            'location': location_features,
            'merchant': merchant_features,
            'velocity': velocity_features,
            'combined': combined_features
        }

        return {
            'success': True,
            'features': all_features,
            'embedding': embedding,
            'preliminary_risk_score': combined_features.get(
                'preliminary_risk_score',
                0)}


class MonitorAgent(BaseAgent):
    """
    Monitor Agent: Perception and feature extraction from incoming transactions.

    Operates continuously in the background:
    - Auto-loads existing CSV files on startup
    - Watches uploads folder for new/changed files
    - Processes transactions on-demand for evaluation

    Contains sub-agents:
    - Capture Sub-agent: Normalizes and enriches transactions
    - Context Sub-agent: Retrieves user's recent transaction history
    - Feature Sub-agent: Extracts statistical features and generates embeddings
    """

    def __init__(self):
        super().__init__("monitor_agent", "Monitor Agent")
        self.capture_subagent = CaptureSubAgent()
        self.context_subagent = ContextSubAgent()
        self.feature_subagent = FeatureSubAgent()

        # File tracking for auto-loading
        self._indexed_files: Dict[str, str] = {}  # filename -> file_hash
        self._uploads_dir = Path(settings.uploads_directory) / "transactions"
        self._watch_interval = 30  # seconds between folder scans
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watcher = threading.Event()

    def initialize(self) -> bool:
        """Initialize the Monitor Agent with continuous file monitoring."""
        self.logger.info("Initializing Monitor Agent")

        # Ensure uploads directory exists
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

        # Load existing CSV files on startup
        self._load_all_csv_files()
        self.logger.info(
            f"Ready for continuous monitoring of {
                self._uploads_dir}")

        # Enable background file watcher for continuous monitoring
        self._start_file_watcher()

        self.is_ready = True
        return True

    def load_user_if_needed(self, user_id: str) -> Dict[str, Any]:
        """
        Load user data on-demand if not already loaded.
        This is called when a user is queried for evaluation.

        Args:
            user_id: The user ID to load data for

        Returns:
            Result with success status and transaction count
        """
        # Check if user data already exists in vector store
        existing_count = vector_store.get_user_transaction_count(user_id)
        if existing_count > 0:
            self.logger.debug(
                f"User '{user_id}' already has {existing_count} transactions loaded")
            return {
                'success': True,
                'already_loaded': True,
                'count': existing_count}

        # Look for CSV file matching the user
        csv_filename = f"{user_id}_transactions.csv"
        csv_path = self._uploads_dir / csv_filename

        if not csv_path.exists():
            self.logger.warning(
                f"No CSV file found for user '{user_id}' at {csv_path}")
            return {
                'success': False,
                'error': f'No data file found for user {user_id}',
                'count': 0}

        # Load the file
        self.logger.info(f"Loading data for user '{user_id}' on-demand...")
        result = self._load_csv_file(csv_path)

        if result.get('success'):
            self._indexed_files[csv_filename] = self._get_file_hash(csv_path)

        return result

    def _get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of a file to detect changes."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _extract_user_id_from_filename(self, filename: str) -> str:
        """Extract user ID from CSV filename (e.g., 'Aaron_transactions.csv' -> 'Aaron')."""
        # Remove extension and common suffixes
        name = filename.replace('.csv', '').replace('_transactions', '')
        return name

    def _load_csv_file(self, filepath: Path) -> Dict[str, Any]:
        """Load a single CSV file and index its transactions."""
        filename = filepath.name
        user_id = self._extract_user_id_from_filename(filename)

        self.logger.info(
            f"Loading transactions for user '{user_id}' from {filename}")

        try:
            # Read file content
            with open(filepath, 'rb') as f:
                content = f.read()

            # Parse transactions
            transactions = parse_csv_transactions(content, user_id)

            if not transactions:
                self.logger.warning(
                    f"No valid transactions found in {filename}")
                return {
                    'success': False,
                    'error': 'No valid transactions',
                    'count': 0}

            # Store transactions in database
            transactions_stored = 0
            with db_manager.session_scope() as session:
                for txn in transactions:
                    try:
                        timestamp = txn.get(
                            'trans_date_trans_time', datetime.utcnow())
                        if not isinstance(timestamp, datetime):
                            timestamp = datetime.utcnow()

                        txn_id = generate_transaction_id(user_id, timestamp)

                        # Check if transaction already exists
                        existing = session.query(Transaction).filter(
                            Transaction.transaction_id == txn_id
                        ).first()

                        if not existing:
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
                            session.add(transaction_record)
                            transactions_stored += 1
                    except Exception as e:
                        continue

                session.commit()

            # Index in vector store
            transaction_ids = []
            for i, txn in enumerate(transactions):
                txn_id = f"{user_id}_hist_{i}_{int(time_module.time())}"
                transaction_ids.append(txn_id)

            embeddings_created = vector_store.add_transactions_batch(
                user_id=user_id,
                transactions=transactions,
                transaction_ids=transaction_ids
            )

            # Update user profile
            profile_stats = calculate_user_profile_stats(transactions)

            with db_manager.session_scope() as session:
                existing_profile = session.query(UserProfile).filter(
                    UserProfile.user_id == user_id
                ).first()

                if existing_profile:
                    existing_profile.total_transactions = profile_stats['total_transactions']
                    existing_profile.avg_amount = profile_stats['avg_amount']
                    existing_profile.std_amount = profile_stats['std_amount']
                    existing_profile.max_amount = profile_stats['max_amount']
                    existing_profile.min_amount = profile_stats['min_amount']
                    existing_profile.common_merchants = profile_stats['common_merchants']
                    existing_profile.common_categories = profile_stats['common_categories']
                    existing_profile.common_locations = profile_stats['common_locations']
                    existing_profile.typical_hours = profile_stats['typical_hours']
                    existing_profile.profile_updated_at = datetime.utcnow()
                else:
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
                        risk_level='low'
                    )
                    session.add(new_profile)

                session.commit()

            self.logger.info(
                f"✓ Loaded {
                    len(transactions)} transactions for user '{user_id}' ({embeddings_created} embeddings)")

            return {
                'success': True,
                'user_id': user_id,
                'transactions_count': len(transactions),
                'embeddings_created': embeddings_created
            }

        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return {'success': False, 'error': str(e), 'count': 0}

    def _load_all_csv_files(self) -> None:
        """Load all existing CSV files from uploads directory on startup."""
        self.logger.info(
            f"Scanning for existing CSV files in {
                self._uploads_dir}")

        if not self._uploads_dir.exists():
            self.logger.info("Uploads directory does not exist, creating...")
            self._uploads_dir.mkdir(parents=True, exist_ok=True)
            return

        csv_files = list(self._uploads_dir.glob("*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files to load")

        total_users = 0
        total_transactions = 0

        for csv_file in csv_files:
            if csv_file.name.startswith('README'):
                continue

            file_hash = self._get_file_hash(csv_file)

            # Check if already indexed with same hash
            if csv_file.name in self._indexed_files:
                if self._indexed_files[csv_file.name] == file_hash:
                    self.logger.debug(
                        f"Skipping {
                            csv_file.name} - already indexed")
                    continue

            # Load the file
            result = self._load_csv_file(csv_file)

            if result.get('success'):
                self._indexed_files[csv_file.name] = file_hash
                total_users += 1
                total_transactions += result.get('transactions_count', 0)

        self.logger.info(
            f"✓ Auto-loaded {total_users} users with {total_transactions} total transactions")

    def _watch_folder(self) -> None:
        """Background thread to watch for new/changed CSV files."""
        self.logger.info("Starting folder watcher thread")

        while not self._stop_watcher.is_set():
            try:
                self._scan_for_changes()
            except Exception as e:
                self.logger.error(f"Error in folder watcher: {e}")

            # Wait for interval or until stopped
            self._stop_watcher.wait(self._watch_interval)

        self.logger.info("Folder watcher stopped")

    def _scan_for_changes(self) -> None:
        """Scan uploads folder for new or changed files."""
        if not self._uploads_dir.exists():
            return

        csv_files = list(self._uploads_dir.glob("*.csv"))

        for csv_file in csv_files:
            if csv_file.name.startswith('README'):
                continue

            try:
                file_hash = self._get_file_hash(csv_file)

                # Check if new file or changed file
                if csv_file.name not in self._indexed_files:
                    self.logger.info(f"Detected new file: {csv_file.name}")
                    result = self._load_csv_file(csv_file)
                    if result.get('success'):
                        self._indexed_files[csv_file.name] = file_hash

                elif self._indexed_files[csv_file.name] != file_hash:
                    self.logger.info(f"Detected changed file: {csv_file.name}")
                    result = self._load_csv_file(csv_file)
                    if result.get('success'):
                        self._indexed_files[csv_file.name] = file_hash

            except Exception as e:
                self.logger.warning(
                    f"Error checking file {
                        csv_file.name}: {e}")

    def _start_file_watcher(self) -> None:
        """Start the background file watcher thread."""
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return

        self._stop_watcher.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_folder,
            daemon=True,
            name="MonitorAgent-FileWatcher"
        )
        self._watcher_thread.start()
        self.logger.info("✓ File watcher started (checking every 30 seconds)")

    def stop_watcher(self) -> None:
        """Stop the background file watcher."""
        self._stop_watcher.set()
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=5)

    def get_indexed_files(self) -> List[str]:
        """Get list of currently indexed files."""
        return list(self._indexed_files.keys())

    def reload_file(self, filename: str) -> Dict[str, Any]:
        """Manually trigger reload of a specific file."""
        filepath = self._uploads_dir / filename
        if not filepath.exists():
            return {'success': False, 'error': f'File not found: {filename}'}

        result = self._load_csv_file(filepath)
        if result.get('success'):
            self._indexed_files[filename] = self._get_file_hash(filepath)
        return result

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming transaction through all sub-agents.

        Args:
            input_data: Raw transaction data

        Returns:
            MonitorAgentOutput with enriched transaction, features, and embedding
        """
        # Step 1: Capture and normalize transaction
        capture_result = await self.capture_subagent.execute(input_data)
        if not capture_result.get('success'):
            return {'success': False, 'error': 'Capture failed'}

        enriched_transaction = capture_result['enriched_transaction']
        user_id = enriched_transaction.get('user_id')

        # Step 2: Get user context (can run in parallel with step 3 for
        # optimization)
        context_result = await self.context_subagent.execute({
            'user_id': user_id,
            'current_transaction': enriched_transaction
        })

        # Step 3: Extract features and generate embedding
        feature_result = await self.feature_subagent.execute({
            'enriched_transaction': enriched_transaction,
            'user_context': context_result
        })

        if not feature_result.get('success'):
            return {'success': False, 'error': 'Feature extraction failed'}

        # Compile output
        return {
            'success': True,
            'transaction_id': enriched_transaction.get('transaction_id'),
            'user_id': user_id,
            'enriched_transaction': enriched_transaction,
            'user_context': context_result,
            'extracted_features': feature_result['features'],
            'embedding': feature_result['embedding'],
            'preliminary_risk_score': feature_result['preliminary_risk_score']
        }


# Singleton instance for on-demand invocation
monitor_agent = MonitorAgent()
