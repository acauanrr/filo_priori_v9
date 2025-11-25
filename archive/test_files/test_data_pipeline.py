"""
Test V8 Data Loading Pipeline
Tests data loading with 500 samples
"""
import sys
sys.path.insert(0, 'src')

import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open('configs/experiment_v8_baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

logger.info("=" * 70)
logger.info("TEST: V8 DATA LOADING PIPELINE")
logger.info("=" * 70)

# Test 1: Data Loader
logger.info("\nTest 1: Loading data...")
from preprocessing.data_loader import DataLoader

data_loader = DataLoader(config)
data_dict = data_loader.prepare_dataset(sample_size=500)

logger.info(f"✓ Train: {len(data_dict['train'])} samples")
logger.info(f"✓ Val: {len(data_dict['val'])} samples")
logger.info(f"✓ Test: {len(data_dict['test'])} samples")
logger.info(f"✓ Num classes: {data_dict['num_classes']}")
logger.info(f"✓ Label mapping: {data_dict['label_mapping']}")

# Test 2: Structural Features
logger.info("\nTest 2: Extracting structural features...")
from preprocessing.structural_feature_extractor import extract_structural_features

df_train = data_dict['train']
df_val = data_dict['val']
df_test = data_dict['test']

train_struct, val_struct, test_struct = extract_structural_features(
    df_train, df_val, df_test,
    recent_window=5,
    cache_path=None
)

logger.info(f"✓ Train features: {train_struct.shape}")
logger.info(f"✓ Val features: {val_struct.shape}")
logger.info(f"✓ Test features: {test_struct.shape}")

# Test 3: Text Processing
logger.info("\nTest 3: Processing text...")
from preprocessing.text_processor import TextProcessor

text_processor = TextProcessor()
train_texts = text_processor.prepare_batch_texts(
    df_train['TE_Summary'].tolist()[:10],
    df_train['TC_Steps'].tolist()[:10],
    df_train['commit'].tolist()[:10]
)

logger.info(f"✓ Processed {len(train_texts)} texts")
logger.info(f"✓ Sample text length: {len(train_texts[0])} chars")

logger.info("\n" + "=" * 70)
logger.info("✅ ALL DATA PIPELINE TESTS PASSED!")
logger.info("=" * 70)
