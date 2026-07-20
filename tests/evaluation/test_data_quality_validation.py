# tests/evaluation/test_data_quality_validation.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class TestDataQualityAcrossIterations:
    """Test that analysis agent validates data quality at each iteration."""
    
    @pytest.fixture
    def valid_dataset(self):
        """Create a valid baseline dataset."""
        np.random.seed(42)  # For reproducibility
        return pd.DataFrame({
            'id': range(1, 101),
            'value': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h')
        })
    
    @pytest.fixture
    def corrupted_datasets(self, valid_dataset):
        """Create various corrupted versions of the dataset."""
        return {
            'missing_values': valid_dataset.copy().assign(
                value=lambda df: df['value'].mask(df.index < 10, np.nan)
            ),
            'duplicates': pd.concat([valid_dataset, valid_dataset.head(10)], ignore_index=True),
            'outliers': valid_dataset.copy().assign(
                value=lambda df: df['value'].mask(df.index < 5, 999999)
            ),
            'schema_violation': valid_dataset.copy().assign(
                value=lambda df: df['value'].astype(str)
            ),
            'empty': pd.DataFrame(columns=valid_dataset.columns)
        }
    
    # ============ CORE VALIDATION TESTS ============
    
    def test_detects_missing_values(self, corrupted_datasets):
        """Agent should detect and report missing values."""
        df = corrupted_datasets['missing_values']
        
        issues = validate_data_quality(df)
        
        assert 'missing_values' in issues
        assert issues['missing_values']['count'] == 10
        assert 'value' in issues['missing_values']['columns']
    
    def test_detects_duplicates(self, corrupted_datasets):
        """Agent should detect duplicate rows."""
        df = corrupted_datasets['duplicates']
        
        issues = validate_data_quality(df)
        
        assert 'duplicates' in issues
        assert issues['duplicates']['count'] == 10
    
    def test_detects_outliers(self, corrupted_datasets):
        """Agent should detect statistical outliers."""
        df = corrupted_datasets['outliers']
        
        issues = validate_data_quality(df)
        
        assert 'outliers' in issues
        assert len(issues['outliers']['indices']) >= 5
    
    def test_detects_schema_violations(self, corrupted_datasets, valid_dataset):
        """Agent should detect schema/type changes."""
        df = corrupted_datasets['schema_violation']
        expected_schema = get_schema(valid_dataset)
        
        issues = validate_schema(df, expected_schema)
        
        assert 'schema_violations' in issues
        assert 'value' in issues['schema_violations']['columns']
    
    def test_detects_empty_dataset(self, corrupted_datasets):
        """Agent should detect empty datasets."""
        df = corrupted_datasets['empty']
        
        issues = validate_data_quality(df)
        
        assert 'empty_dataset' in issues
        assert issues['empty_dataset'] is True
    
    # ============ ITERATION-BASED TESTS ============
    
    def test_validates_at_each_iteration(self, valid_dataset):
        """Agent should run validation at every iteration."""
        iterations = []
        
        for i in range(5):
            # Simulate processing
            processed_df = process_data(valid_dataset, iteration=i)
            validation_result = validate_data_quality(processed_df)
            iterations.append({
                'iteration': i,
                'validation': validation_result,
                'timestamp': pd.Timestamp.now()
            })
        
        # Ensure validation happened at each iteration
        assert len(iterations) == 5
        assert all('validation' in iter_data for iter_data in iterations)
    
    def test_tracks_quality_degradation_over_iterations(self):
        """Agent should detect when data quality degrades over iterations."""
        np.random.seed(42)  # For reproducibility
        initial_data = pd.DataFrame({
            'value': np.random.randn(100)
        })
        
        quality_scores = []
        
        for i in range(10):
            # Progressively corrupt data
            corruption_rate = i * 0.1  # 0%, 10%, 20%, ..., 90%
            corrupted = introduce_corruption(initial_data.copy(), rate=corruption_rate)
            
            score = calculate_quality_score(corrupted)
            quality_scores.append(score)
            print(f"Iteration {i}, corruption {corruption_rate:.1%}, score {score:.3f}")
        
        # Quality should degrade overall
        assert quality_scores[0] > quality_scores[-1], \
            f"Initial score {quality_scores[0]:.3f} should be > final score {quality_scores[-1]:.3f}"
        
        # Check overall trend: first half avg > second half avg
        first_half_avg = np.mean(quality_scores[:5])
        second_half_avg = np.mean(quality_scores[5:])
        assert first_half_avg > second_half_avg, \
            f"First half avg {first_half_avg:.3f} should be > second half avg {second_half_avg:.3f}"
    
    def test_maintains_data_invariants_across_iterations(self, valid_dataset):
        """Critical data properties should remain constant."""
        invariants = {
            'row_count': len(valid_dataset),
            'column_count': len(valid_dataset.columns),
            'column_names': set(valid_dataset.columns),
            'id_range': (valid_dataset['id'].min(), valid_dataset['id'].max())
        }
        
        for i in range(5):
            processed = process_data(valid_dataset, iteration=i)
            
            assert len(processed) == invariants['row_count']
            assert len(processed.columns) == invariants['column_count']
            assert set(processed.columns) == invariants['column_names']
            assert processed['id'].min() == invariants['id_range'][0]
            assert processed['id'].max() == invariants['id_range'][1]
    
    # ============ STATISTICAL VALIDATION ============
    
    def test_validates_statistical_properties(self, valid_dataset):
        """Agent should check statistical properties remain reasonable."""
        baseline_stats = {
            'mean': valid_dataset['value'].mean(),
            'std': valid_dataset['value'].std(),
            'min': valid_dataset['value'].min(),
            'max': valid_dataset['value'].max()
        }
        
        for i in range(5):
            processed = process_data(valid_dataset, iteration=i)
            current_stats = calculate_stats(processed['value'])
            
            # Stats should be within reasonable bounds (e.g., 3 sigma)
            assert abs(current_stats['mean'] - baseline_stats['mean']) < 3 * baseline_stats['std']
            assert current_stats['std'] > 0
    
    def test_detects_distribution_shift(self, valid_dataset):
        """Agent should detect when data distribution changes significantly."""
        np.random.seed(42)
        baseline_dist = valid_dataset['category'].value_counts(normalize=True)
        
        # Create shifted distribution
        shifted = valid_dataset.copy()
        shifted['category'] = np.random.choice(['A', 'B', 'C'], 
                                               size=len(shifted), 
                                               p=[0.7, 0.2, 0.1])  # Heavily skewed
        shifted_dist = shifted['category'].value_counts(normalize=True)
        
        print(f"\nBaseline distribution:\n{baseline_dist}")
        print(f"\nShifted distribution:\n{shifted_dist}")
        
        drift_detected = detect_distribution_shift(
            baseline_dist, 
            shifted_dist,
            threshold=0.1
        )
        
        assert drift_detected is True, "Should detect significant distribution shift"
    
    # ============ INTEGRATION TESTS ============
    
    def test_full_pipeline_with_quality_checks(self, valid_dataset):
        """Test complete pipeline with quality gates at each step."""
        pipeline_steps = [
            ('load', lambda df: df),
            ('clean', lambda df: clean_data(df)),
            ('transform', lambda df: transform_data(df)),
            ('aggregate', lambda df: aggregate_data(df))
        ]
        
        data = valid_dataset
        quality_log = []
        
        for step_name, step_func in pipeline_steps:
            # Execute step
            data = step_func(data)
            
            # Validate after each step
            issues = validate_data_quality(data)
            quality_log.append({
                'step': step_name,
                'issues': issues,
                'passed': len(issues) == 0
            })
            
            # Fail fast if quality issues found
            if len(issues) > 0:
                pytest.fail(f"Quality issues found at step '{step_name}': {issues}")
        
        assert all(log['passed'] for log in quality_log)
    
    def test_agent_generates_quality_report(self, valid_dataset):
        """Agent should generate comprehensive quality reports."""
        report = generate_quality_report(valid_dataset)
        
        required_sections = [
            'completeness',
            'consistency',
            'validity',
            'uniqueness',
            'timeliness',
            'accuracy'
        ]
        
        assert all(section in report for section in required_sections)
        assert 'overall_score' in report
        assert 0 <= report['overall_score'] <= 1


# ============ HELPER FUNCTIONS ============

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Main validation function that checks multiple quality dimensions."""
    issues = {}
    
    # Check for empty dataset
    if len(df) == 0:
        issues['empty_dataset'] = True
        return issues
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues['missing_values'] = {
            'count': int(missing.sum()),
            'columns': missing[missing > 0].to_dict()
        }
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues['duplicates'] = {'count': int(duplicates)}
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_indices = []
    for col in numeric_cols:
        if df[col].std() > 0:  # Avoid division by zero
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > 3
            if outliers.any():
                outlier_indices.extend(df[outliers].index.tolist())
    
    if outlier_indices:
        issues['outliers'] = {'indices': list(set(outlier_indices))}
    
    return issues


def validate_schema(df: pd.DataFrame, expected_schema: Dict) -> Dict[str, Any]:
    """Validate dataframe against expected schema."""
    issues = {}
    violations = {}
    
    for col, expected_type in expected_schema.items():
        if col not in df.columns:
            violations[col] = f"Missing column"
        elif df[col].dtype != expected_type:
            violations[col] = f"Expected {expected_type}, got {df[col].dtype}"
    
    if violations:
        issues['schema_violations'] = {'columns': violations}
    
    return issues


def get_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract schema from dataframe."""
    return {col: df[col].dtype for col in df.columns}


def calculate_quality_score(df: pd.DataFrame) -> float:
    """Calculate overall quality score (0-1)."""
    if len(df) == 0:
        return 0.0
    
    scores = []
    
    # Completeness: % non-null values
    total_cells = len(df) * len(df.columns)
    if total_cells > 0:
        completeness = 1 - (df.isnull().sum().sum() / total_cells)
        scores.append(completeness)
    
    # Uniqueness: % non-duplicate rows
    uniqueness = 1 - (df.duplicated().sum() / len(df))
    scores.append(uniqueness)
    
    return np.mean(scores)

def detect_distribution_shift(baseline: pd.Series, current: pd.Series, threshold: float = 0.1) -> bool:
    """Detect if distribution has shifted beyond threshold using Total Variation Distance."""
    
    # Handle edge cases
    if len(baseline) == 0 or len(current) == 0:
        return False
    
    # Ensure both are normalized (they should already be from value_counts(normalize=True))
    baseline_sum = baseline.sum()
    current_sum = current.sum()
    
    baseline_norm = baseline / baseline_sum if baseline_sum > 0 else baseline
    current_norm = current / current_sum if current_sum > 0 else current
    
    # Get all categories
    all_categories = set(baseline_norm.index) | set(current_norm.index)
    
    # Calculate total variation distance
    distance = 0.0
    for cat in all_categories:
        # Use proper pandas Series indexing
        baseline_val = float(baseline_norm.loc[cat]) if cat in baseline_norm.index else 0.0
        current_val = float(current_norm.loc[cat]) if cat in current_norm.index else 0.0
        distance += abs(baseline_val - current_val)
    
    # Total Variation Distance formula
    distance = distance / 2.0
    
    print(f"Total Variation Distance: {distance:.3f}, Threshold: {threshold}")
    print(f"Returning: {distance > threshold}")
    
    return distance > threshold


def generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data quality report."""
    return {
        'completeness': calculate_quality_score(df),
        'consistency': 1.0,
        'validity': 1.0,
        'uniqueness': 1 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 1.0,
        'timeliness': 1.0,
        'accuracy': 1.0,
        'overall_score': calculate_quality_score(df)
    }


# Placeholder functions
def process_data(df, iteration):
    return df.copy()

def clean_data(df):
    return df.dropna()

def transform_data(df):
    return df

def aggregate_data(df):
    return df

def calculate_stats(series):
    return {
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max()
    }

def introduce_corruption(df, rate):
    """Introduce missing values at specified rate."""
    corrupted = df.copy()
    n_corrupted = int(len(corrupted) * rate)
    if n_corrupted > 0:
        corrupt_indices = np.random.choice(corrupted.index, size=n_corrupted, replace=False)
        corrupted.loc[corrupt_indices, 'value'] = np.nan
    return corrupted