"""
Utility functions for GUARDIAN system.
Feature extraction, data normalization, and helper functions.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import statistics
import pandas as pd
import math


def calculate_haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    Uses the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point (user)
        lat2, lon2: Latitude and longitude of second point (merchant)
        
    Returns:
        Distance in miles
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in miles
    r = 3959
    
    return round(c * r, 2)


def generate_transaction_id(user_id: str, timestamp: datetime) -> str:
    """
    Generate a unique transaction ID.
    
    Args:
        user_id: User identifier
        timestamp: Transaction timestamp
        
    Returns:
        Unique transaction ID
    """
    # Create a hash-based unique ID
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    hash_input = f"{user_id}_{time_str}_{datetime.utcnow().microsecond}"
    hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"txn_{time_str}_{user_id}_{hash_suffix}"


def mask_credit_card(cc_num: str) -> str:
    """
    Mask credit card number for security.
    
    Args:
        cc_num: Credit card number
        
    Returns:
        Masked credit card number
    """
    if not cc_num or len(cc_num) < 4:
        return cc_num
    return '*' * (len(cc_num) - 4) + cc_num[-4:]


def extract_temporal_features(timestamp: datetime) -> Dict[str, Any]:
    """
    Extract temporal features from a timestamp.
    
    Args:
        timestamp: Transaction timestamp
        
    Returns:
        Dictionary of temporal features
    """
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),  # 0=Monday, 6=Sunday
        'day_of_month': timestamp.day,
        'month': timestamp.month,
        'is_weekend': timestamp.weekday() >= 5,
        'is_night': timestamp.hour < 6 or timestamp.hour >= 22,
        'is_business_hours': 9 <= timestamp.hour <= 17 and timestamp.weekday() < 5,
        'quarter': (timestamp.month - 1) // 3 + 1
    }


def extract_amount_features(
    amount: float,
    historical_amounts: List[float]
) -> Dict[str, Any]:
    """
    Extract amount-related features compared to history.
    
    Args:
        amount: Current transaction amount
        historical_amounts: List of historical transaction amounts
        
    Returns:
        Dictionary of amount features
    """
    if not historical_amounts:
        return {
            'amount_zscore': 0,
            'amount_percentile': 50,
            'is_above_average': False,
            'ratio_to_max': 1.0,
            'ratio_to_average': 1.0
        }
    
    avg = statistics.mean(historical_amounts)
    std = statistics.stdev(historical_amounts) if len(historical_amounts) > 1 else 1.0
    max_amt = max(historical_amounts)
    
    # Z-score (how many standard deviations from mean)
    zscore = (amount - avg) / std if std > 0 else 0
    
    # Percentile
    below_count = sum(1 for a in historical_amounts if a <= amount)
    percentile = (below_count / len(historical_amounts)) * 100
    
    return {
        'amount_zscore': round(zscore, 2),
        'amount_percentile': round(percentile, 1),
        'is_above_average': amount > avg,
        'ratio_to_max': round(amount / max_amt, 2) if max_amt > 0 else 1.0,
        'ratio_to_average': round(amount / avg, 2) if avg > 0 else 1.0
    }


def extract_location_features(
    city: str,
    state: str,
    country: str,
    historical_locations: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Extract location-related features.
    
    DEPRECATED: Use extract_geographic_features for RAG-based approach.
    
    Args:
        city: Current transaction city
        state: Current transaction state
        country: Current transaction country
        historical_locations: List of historical locations
        
    Returns:
        Dictionary of location features
    """
    if not historical_locations:
        return {
            'is_new_city': True,
            'is_new_state': True,
            'is_new_country': True,
            'is_international': country != 'US',
            'location_novelty_score': 1.0
        }
    
    # Check if location is new
    historical_cities = {loc.get('city', '').lower() for loc in historical_locations}
    historical_states = {loc.get('state', '').lower() for loc in historical_locations}
    historical_countries = {loc.get('country', 'US').upper() for loc in historical_locations}
    
    is_new_city = city.lower() not in historical_cities
    is_new_state = state.lower() not in historical_states
    is_new_country = country.upper() not in historical_countries
    
    # Calculate novelty score (0 = familiar, 1 = completely new)
    novelty_score = 0.0
    if is_new_city:
        novelty_score += 0.3
    if is_new_state:
        novelty_score += 0.3
    if is_new_country:
        novelty_score += 0.4
    
    return {
        'is_new_city': is_new_city,
        'is_new_state': is_new_state,
        'is_new_country': is_new_country,
        'is_international': country.upper() != 'US',
        'location_novelty_score': novelty_score
    }


def extract_geographic_features(
    merch_lat: float,
    merch_long: float,
    current_city: str,
    current_state: str,
    historical_transactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract geographic features for RAG-based behavioral analysis.
    Since user home coordinates aren't available, we use:
    - City/state patterns to determine typical locations
    - Merchant coordinates to calculate distances between transactions
    
    Args:
        merch_lat: Current merchant latitude
        merch_long: Current merchant longitude
        current_city: Current transaction city
        current_state: Current transaction state
        historical_transactions: List of user's historical transactions
        
    Returns:
        Dictionary of geographic features
    """
    if not historical_transactions:
        return {
            'is_new_city': True,
            'is_new_state': True,
            'avg_distance_between_merchants': 0.0,
            'max_distance_between_merchants': 0.0,
            'typical_cities': [],
            'typical_states': []
        }
    
    # Extract historical cities and states
    historical_cities = [t.get('city', '').lower() for t in historical_transactions if t.get('city')]
    historical_states = [t.get('state', '').lower() for t in historical_transactions if t.get('state')]
    
    # Check if location is new
    is_new_city = current_city.lower() not in set(historical_cities)
    is_new_state = current_state.lower() not in set(historical_states)
    
    # Find most common locations
    from collections import Counter
    typical_cities = [city for city, _ in Counter(historical_cities).most_common(5)]
    typical_states = [state for state, _ in Counter(historical_states).most_common(3)]
    
    # Calculate distances between merchant locations
    distances = []
    historical_coords = [
        (t.get('merch_lat'), t.get('merch_long')) 
        for t in historical_transactions 
        if t.get('merch_lat') and t.get('merch_long')
    ]
    
    if historical_coords and merch_lat and merch_long:
        # Calculate distance from current merchant to all historical merchants
        for hist_lat, hist_long in historical_coords:
            distance = calculate_haversine_distance(
                merch_lat, merch_long, hist_lat, hist_long
            )
            distances.append(distance)
    
    avg_distance = statistics.mean(distances) if distances else 0.0
    max_distance = max(distances) if distances else 0.0
    min_distance = min(distances) if distances else 0.0
    
    return {
        'is_new_city': is_new_city,
        'is_new_state': is_new_state,
        'avg_distance_between_merchants': round(avg_distance, 2),
        'max_distance_between_merchants': round(max_distance, 2),
        'min_distance_between_merchants': round(min_distance, 2),
        'typical_cities': typical_cities,
        'typical_states': typical_states
    }


def extract_merchant_features(
    merchant: str,
    category: str,
    historical_merchants: List[str],
    historical_categories: List[str]
) -> Dict[str, Any]:
    """
    Extract merchant-related features.
    
    Args:
        merchant: Current merchant name
        category: Current transaction category
        historical_merchants: List of historical merchants
        historical_categories: List of historical categories
        
    Returns:
        Dictionary of merchant features
    """
    if not historical_merchants:
        return {
            'is_new_merchant': True,
            'is_new_category': True,
            'merchant_novelty_score': 1.0
        }
    
    # Normalize names for comparison
    historical_merchants_lower = {m.lower() for m in historical_merchants}
    historical_categories_lower = {c.lower() for c in historical_categories}
    
    is_new_merchant = merchant.lower() not in historical_merchants_lower
    is_new_category = category.lower() not in historical_categories_lower
    
    # Merchant frequency (how common is this merchant)
    merchant_count = sum(1 for m in historical_merchants if m.lower() == merchant.lower())
    merchant_frequency = merchant_count / len(historical_merchants) if historical_merchants else 0
    
    # Calculate novelty score
    novelty_score = 0.0
    if is_new_merchant:
        novelty_score += 0.6
    if is_new_category:
        novelty_score += 0.4
    
    return {
        'is_new_merchant': is_new_merchant,
        'is_new_category': is_new_category,
        'merchant_frequency': round(merchant_frequency, 3),
        'merchant_novelty_score': novelty_score
    }


def calculate_velocity_features(
    current_timestamp: datetime,
    recent_transactions: List[Dict[str, Any]],
    window_hours: int = 24
) -> Dict[str, Any]:
    """
    Calculate transaction velocity features.
    
    Args:
        current_timestamp: Current transaction timestamp
        recent_transactions: Recent transactions within window
        window_hours: Time window in hours
        
    Returns:
        Dictionary of velocity features
    """
    if not recent_transactions:
        return {
            'transactions_last_hour': 0,
            'transactions_last_day': 0,
            'total_amount_last_hour': 0.0,
            'total_amount_last_day': 0.0,
            'velocity_score': 0.0
        }
    
    hour_ago = current_timestamp - timedelta(hours=1)
    day_ago = current_timestamp - timedelta(hours=24)
    
    txns_last_hour = []
    txns_last_day = []
    
    for txn in recent_transactions:
        txn_time = txn.get('trans_date_trans_time')
        if isinstance(txn_time, str):
            try:
                txn_time = datetime.fromisoformat(txn_time.replace('Z', '+00:00'))
            except:
                continue
        
        if txn_time and txn_time >= day_ago:
            txns_last_day.append(txn)
            if txn_time >= hour_ago:
                txns_last_hour.append(txn)
    
    amount_last_hour = sum(t.get('amt', 0) for t in txns_last_hour)
    amount_last_day = sum(t.get('amt', 0) for t in txns_last_day)
    
    # Velocity score (normalized, higher = more activity)
    # Normal: ~3 transactions/day, <$500/day
    txn_velocity = len(txns_last_day) / 10.0  # Normalize to 0-1 (10 txns = 1.0)
    amount_velocity = amount_last_day / 5000.0  # Normalize to 0-1 ($5000 = 1.0)
    velocity_score = min(1.0, (txn_velocity + amount_velocity) / 2)
    
    return {
        'transactions_last_hour': len(txns_last_hour),
        'transactions_last_day': len(txns_last_day),
        'total_amount_last_hour': round(amount_last_hour, 2),
        'total_amount_last_day': round(amount_last_day, 2),
        'velocity_score': round(velocity_score, 2)
    }


def normalize_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and clean transaction data.
    
    Args:
        transaction: Raw transaction data
        
    Returns:
        Normalized transaction
    """
    normalized = transaction.copy()
    
    # Ensure amount is float
    if 'amt' in normalized:
        normalized['amt'] = float(normalized['amt'])
    
    # Normalize strings
    for field in ['merchant', 'category', 'city', 'state', 'country']:
        if field in normalized and normalized[field]:
            normalized[field] = str(normalized[field]).strip()
    
    # Default country
    if not normalized.get('country'):
        normalized['country'] = 'US'
    
    # Mask credit card
    if 'cc_num' in normalized and normalized['cc_num']:
        normalized['cc_num'] = mask_credit_card(str(normalized['cc_num']))
    
    # Parse timestamp if string
    if 'trans_date_trans_time' in normalized:
        ts = normalized['trans_date_trans_time']
        if isinstance(ts, str):
            try:
                normalized['trans_date_trans_time'] = datetime.fromisoformat(
                    ts.replace('Z', '+00:00')
                )
            except:
                pass
    
    return normalized


def calculate_combined_risk_features(
    temporal: Dict[str, Any],
    amount: Dict[str, Any],
    location: Dict[str, Any],
    merchant: Dict[str, Any],
    velocity: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate combined risk features from all feature categories.
    
    Args:
        temporal: Temporal features
        amount: Amount features
        location: Location features
        merchant: Merchant features
        velocity: Velocity features
        
    Returns:
        Dictionary with combined risk indicators
    """
    # Calculate weighted risk score
    risk_factors = []
    
    # Temporal risk (late night = higher risk)
    if temporal.get('is_night'):
        risk_factors.append(0.3)
    
    # Amount risk (high z-score = higher risk)
    zscore = abs(amount.get('amount_zscore', 0))
    if zscore > 3:
        risk_factors.append(0.4)
    elif zscore > 2:
        risk_factors.append(0.2)
    
    # Location risk
    location_novelty = location.get('location_novelty_score', 0)
    risk_factors.append(location_novelty * 0.4)
    
    # International transaction
    if location.get('is_international'):
        risk_factors.append(0.2)
    
    # New country
    if location.get('is_new_country'):
        risk_factors.append(0.3)
    
    # Merchant risk
    merchant_novelty = merchant.get('merchant_novelty_score', 0)
    risk_factors.append(merchant_novelty * 0.2)
    
    # Velocity risk
    velocity_score = velocity.get('velocity_score', 0)
    if velocity_score > 0.7:
        risk_factors.append(0.3)
    
    # Calculate overall preliminary risk
    preliminary_risk = min(1.0, sum(risk_factors) / 2.0)
    
    return {
        'preliminary_risk_score': round(preliminary_risk, 2),
        'risk_factors_count': len([r for r in risk_factors if r > 0.1]),
        'high_risk_flags': [
            'late_night' if temporal.get('is_night') else None,
            'high_amount' if zscore > 2 else None,
            'new_location' if location_novelty > 0.5 else None,
            'international' if location.get('is_international') else None,
            'high_velocity' if velocity_score > 0.7 else None
        ]
    }


def parse_csv_transactions(file_content: bytes, user_id: str) -> List[Dict[str, Any]]:
    """
    Parse CSV file content into list of transactions.
    
    Args:
        file_content: Raw CSV file bytes
        user_id: User ID for the transactions
        
    Returns:
        List of transaction dictionaries
    """
    import io
    
    # Read CSV
    df = pd.read_csv(io.BytesIO(file_content))
    
    # Standardize column names
    column_mapping = {
        'user_id': 'user_id',
        'cc_num': 'cc_num',
        'trans_date_trans_time': 'trans_date_trans_time',
        'merchant': 'merchant',
        'category': 'category',
        'amt': 'amt',
        'gender': 'gender',
        'street': 'street',
        'city': 'city',
        'state': 'state',
        'zip': 'zip',
        'trans_num': 'trans_num',
        'is_fraud': 'is_fraud'
    }
    
    # Rename columns that exist
    existing_cols = {c: column_mapping[c] for c in column_mapping if c in df.columns}
    df = df.rename(columns=existing_cols)
    
    # Parse datetime
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    
    # Add user_id if not present
    if 'user_id' not in df.columns:
        df['user_id'] = user_id
    
    # Convert to list of dicts
    transactions = df.to_dict('records')
    
    # Clean None/NaN values
    for txn in transactions:
        for key, value in txn.items():
            if pd.isna(value):
                txn[key] = None
            elif key == 'trans_date_trans_time' and hasattr(value, 'to_pydatetime'):
                txn[key] = value.to_pydatetime()
    
    return transactions


def calculate_user_profile_stats(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate user profile statistics from transaction history.
    
    Args:
        transactions: List of user transactions
        
    Returns:
        Dictionary with profile statistics
    """
    if not transactions:
        return {
            'total_transactions': 0,
            'avg_amount': 0.0,
            'std_amount': 0.0,
            'max_amount': 0.0,
            'min_amount': 0.0,
            'common_merchants': [],
            'common_categories': [],
            'common_locations': [],
            'typical_hours': []
        }
    
    amounts = [t.get('amt', 0) for t in transactions if t.get('amt')]
    merchants = [t.get('merchant', '') for t in transactions if t.get('merchant')]
    categories = [t.get('category', '') for t in transactions if t.get('category')]
    
    # Calculate statistics
    avg_amount = statistics.mean(amounts) if amounts else 0.0
    std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0.0
    
    # Find common values
    from collections import Counter
    common_merchants = [m for m, _ in Counter(merchants).most_common(10)]
    common_categories = [c for c, _ in Counter(categories).most_common(5)]
    
    # Extract locations
    locations = []
    for t in transactions:
        if t.get('city') or t.get('state'):
            locations.append({
                'city': t.get('city', ''),
                'state': t.get('state', ''),
                'country': t.get('country', 'US')
            })
    
    # Unique locations (top 10)
    unique_locations = []
    seen = set()
    for loc in locations:
        key = f"{loc['city']}_{loc['state']}_{loc['country']}"
        if key not in seen:
            seen.add(key)
            unique_locations.append(loc)
            if len(unique_locations) >= 10:
                break
    
    # Typical hours
    hours = []
    for t in transactions:
        ts = t.get('trans_date_trans_time')
        if hasattr(ts, 'hour'):
            hours.append(ts.hour)
    typical_hours = [h for h, _ in Counter(hours).most_common(5)]
    
    # Geographic baseline using merchant locations (no user home coords available)
    # Calculate average distances between merchant locations
    merchant_coords = [
        (t.get('merch_lat'), t.get('merch_long')) 
        for t in transactions 
        if t.get('merch_lat') and t.get('merch_long')
    ]
    
    distances_between_merchants = []
    if len(merchant_coords) > 1:
        # Calculate distances between consecutive transactions
        for i in range(len(merchant_coords) - 1):
            lat1, long1 = merchant_coords[i]
            lat2, long2 = merchant_coords[i + 1]
            if lat1 and long1 and lat2 and long2:
                distance = calculate_haversine_distance(lat1, long1, lat2, long2)
                distances_between_merchants.append(distance)
    
    avg_shopping_distance = statistics.mean(distances_between_merchants) if distances_between_merchants else 0.0
    max_shopping_distance = max(distances_between_merchants) if distances_between_merchants else 0.0
    
    # Get date range
    dates = [t.get('trans_date_trans_time') for t in transactions if t.get('trans_date_trans_time')]
    first_date = min(dates) if dates else None
    last_date = max(dates) if dates else None
    
    return {
        'total_transactions': len(transactions),
        'avg_amount': round(avg_amount, 2),
        'std_amount': round(std_amount, 2),
        'max_amount': max(amounts) if amounts else 0.0,
        'min_amount': min(amounts) if amounts else 0.0,
        'common_merchants': common_merchants,
        'common_categories': common_categories,
        'common_locations': unique_locations,
        'typical_hours': typical_hours,
        'home_lat': None,  # Not available in dataset
        'home_long': None,  # Not available in dataset
        'avg_shopping_distance': round(avg_shopping_distance, 2),
        'max_shopping_distance': round(max_shopping_distance, 2),
        'first_transaction_date': first_date,
        'last_transaction_date': last_date
    }
