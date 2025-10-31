import pytest
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

from project.DataModule import DataModule 

@pytest.fixture(scope="module")
def dummy_data_path(tmp_path_factory):
    """Creates a dummy CSV file for testing and returns its path."""
    data = {
        'Gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'] * 3, 
        'Age': list(range(20, 50)), 
        'Height': [1.70, 1.85, 1.65, 1.75, 1.72, 1.80, 1.68, 1.90, 1.60, 1.78] * 3, 
        'Weight': [70.5, 90.2, 60.0, 80.1, 75.3, 85.0, 68.9, 95.5, 55.1, 82.7] * 3, 
        'Favorite Food': ['Pizza', 'Burger', 'Salad', 'Pasta', 'Sushi', 'Taco', 'Steak', 'Curry', 'Ramen', 'Pho'] * 3, 
        'TargetColumn': ['ClassA', 'ClassB', 'ClassC'] * 10 
    }
    df = pd.DataFrame(data)

    data_dir = tmp_path_factory.mktemp("data")
    fn = data_dir / "dummy_data.csv"

    df.to_csv(fn, index=False)
    return fn

@pytest.fixture(scope="module")
def basic_data_module(dummy_data_path):
    """Provides a basic DataModule instance for testing."""
    return DataModule(data_path=dummy_data_path, target_column='TargetColumn')

@pytest.fixture(scope="module")
def configured_data_module(dummy_data_path):
    """Provides a DataModule instance with specific configurations."""
    preprocessor_settings = {
        'numerical': {
            'Age': {'scaler': 'standard'},    
            'Height': {'scaler': 'standard'}, 
            'Weight': {'scaler': 'standard'}  
        },
        'categorical': {
            'Gender': {'encoder': 'onehot', 'encoder_options': {'sparse_output': False}} 
        }
    }
    return DataModule(
        data_path=dummy_data_path,
        target_column='TargetColumn',
        columns_to_drop=['Favorite Food'],
        preprocessor_settings=preprocessor_settings,
        visualise=False,
        check_imbalance=True,
        import pytest
        import pandas as pd
        from pathlib import Path
        import numpy as np

        from train.src.DataModule import DataModule
import pytest
import pandas as pd
from pathlib import Path
import numpy as np

from train.src.DataModule import DataModule


@pytest.fixture
def small_csv(tmp_path):
    df = pd.DataFrame({
        'Gender': ['F', 'M', 'F', 'M', 'F', 'M'],
        'Age': [23, 35, 45, 22, 34, 28],
        'Weight': [55.0, 80.2, 68.5, 72.1, 60.3, 77.0],
        'Target': [0, 1, 0, 0, 1, 0]
    })
    p = tmp_path / "small.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_datamodule_load_and_prepare_basic(small_csv):
    dm = DataModule(data_path=small_csv, target_column='Target')
    X, y = dm.load_and_prepare()

    assert X.shape[0] == 6
    assert y.shape[0] == 6
    assert 'Target' not in X.columns
    assert 'Age' in dm.numerical_features
    assert 'Gender' in dm.categorical_features


def test_create_and_apply_preprocessor(small_csv):
    dm = DataModule(data_path=small_csv, target_column='Target')
    X, y = dm.load_and_prepare()

    # Preprocessor should be set up (even if passthrough)
    assert dm.preprocessor is not None

    # Fit preprocessor and transform data
    fitted = dm.create_and_fit_preprocessor(X)
    transformed = dm.transform_data(X)

    assert hasattr(fitted, 'transform')
    assert transformed.shape[0] == X.shape[0]


def test_train_test_split_behavior(small_csv):
    dm = DataModule(data_path=small_csv, target_column='Target', test_size=0.4, stratify=False)
    X, y = dm.load_and_prepare()

    X_train, X_test, y_train, y_test = dm.perform_train_test_split(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

