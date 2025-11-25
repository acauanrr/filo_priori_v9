"""
Quick V8 Training Test with Small Sample
Tests the complete training pipeline with 1000 samples
"""
import sys
import os

# Modify main_v8.py to accept sample_size
if __name__ == '__main__':
    # Set sample size for quick test
    os.environ['SAMPLE_SIZE'] = '1000'

    # Run with modified args
    sys.argv = [
        'main_v8.py',
        '--config', 'configs/experiment_v8_baseline.yaml',
        '--device', 'cpu'
    ]

    # Import and run main
    exec(open('main_v8.py').read())
