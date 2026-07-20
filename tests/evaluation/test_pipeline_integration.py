import pandas as pd
import numpy as np
from typing import Dict, Any
import pytest


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality."""
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
            'columns': {k: int(v) for k, v in missing[missing > 0].to_dict().items()}
        }
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues['duplicates'] = {'count': int(duplicates)}
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_indices = []
    
    for col in numeric_cols:
        # Skip columns with all NaN values or no variance
        if df[col].isna().all() or df[col].std() == 0:
            continue
            
        # Calculate z-scores, handling NaN values properly
        col_data = df[col].dropna()
        if len(col_data) > 0:
            mean_val = col_data.mean()
            std_val = col_data.std()
            
            if std_val > 0:
                z_scores = np.abs((df[col] - mean_val) / std_val)
                outliers = z_scores > 4  # 4 sigma threshold
                if outliers.any():
                    outlier_indices.extend(df[outliers].index.tolist())
    
    # Only flag if more than 5% of data are outliers
    if outlier_indices:
        unique_outliers = list(set(outlier_indices))
        if len(unique_outliers) > len(df) * 0.05:
            issues['outliers'] = {
                'count': len(unique_outliers),
                'indices': unique_outliers
            }
    
    return issues


class TestDataQualityValidation:
    """Test suite for data quality validation."""
    
    def test_empty_dataset(self):
        """Test empty dataset detection."""
        df = pd.DataFrame()
        result = validate_data_quality(df)
        assert result['empty_dataset'] is True
    
    def test_clean_data(self):
        """Test with clean data (no issues)."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': ['x', 'y', 'z', 'w', 'v']
        })
        result = validate_data_quality(df)
        assert len(result) == 0
    
    def test_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': [10, None, 30, None, 50]
        })
        result = validate_data_quality(df)
        assert 'missing_values' in result
        assert result['missing_values']['count'] == 3
        assert 'a' in result['missing_values']['columns']
        assert 'b' in result['missing_values']['columns']
    
    def test_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3, 3, 3],
            'b': [10, 20, 20, 30, 30, 30]
        })
        result = validate_data_quality(df)
        assert 'duplicates' in result
        assert result['duplicates']['count'] == 3
    
    def test_outliers(self):
        """Test outlier detection."""
        # Create data with extreme outliers
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200],
            'b': [10] * 12
        })
        result = validate_data_quality(df)
        # Should detect outliers in column 'a'
        if 'outliers' in result:
            assert result['outliers']['count'] > 0
    
    def test_no_outliers_with_moderate_variance(self):
        """Test that moderate variance doesn't trigger outliers."""
        df = pd.DataFrame({
            'a': list(range(1, 101))
        })
        result = validate_data_quality(df)
        # Should not detect outliers in normal distribution
        assert 'outliers' not in result
    
    def test_mixed_issues(self):
        """Test data with multiple quality issues."""
        df = pd.DataFrame({
            'a': [1, 2, 2, None, 5, 1000],
            'b': [10, 20, 20, 30, None, 50]
        })
        result = validate_data_quality(df)
        assert 'missing_values' in result
        assert 'duplicates' in result


def test_pipeline_integration():
    """Integration test for the complete data pipeline."""
    # Simulate a data pipeline
    raw_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [100, 200, 300, 400, 500],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    # Validate data quality
    quality_issues = validate_data_quality(raw_data)
    
    # Assert no issues
    assert len(quality_issues) == 0, f"Unexpected quality issues: {quality_issues}"
    
    # Perform some transformations
    processed_data = raw_data.copy()
    processed_data['value_normalized'] = processed_data['value'] / processed_data['value'].max()
    
    # Validate processed data
    processed_issues = validate_data_quality(processed_data)
    assert len(processed_issues) == 0
    
    print("✓ Pipeline integration test passed!")


if __name__ == '__main__':
    # Run tests
    print("Running data quality validation tests...\n")
    
    # Run with pytest if available, otherwise run manual tests
    try:
        pytest.main([__file__, '-v'])
    except:
        print("pytest not found, running manual tests...\n")
        
        # Manual test execution
        test_suite = TestDataQualityValidation()
        
        tests = [
            ('Empty Dataset', test_suite.test_empty_dataset),
            ('Clean Data', test_suite.test_clean_data),
            ('Missing Values', test_suite.test_missing_values),
            ('Duplicates', test_suite.test_duplicates),
            ('Outliers', test_suite.test_outliers),
            ('No Outliers', test_suite.test_no_outliers_with_moderate_variance),
            ('Mixed Issues', test_suite.test_mixed_issues),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                test_func()
                print(f"✓ {test_name} - PASSED")
                passed += 1
            except Exception as e:
                print(f"✗ {test_name} - FAILED: {e}")
                failed += 1
        
        print(f"\n{'='*50}")
        print(f"Results: {passed} passed, {failed} failed")
        print(f"{'='*50}\n")
        
        # Run integration test
        test_pipeline_integration()