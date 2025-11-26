#!/usr/bin/env python3
"""
Pre-Flight Validation Script for Filo-Priori Experiments

This script performs comprehensive validation checks before running experiments
to catch configuration errors, missing dependencies, dataset issues, and
GPU/CUDA problems early.

Usage:
    python preflight_check.py [--config configs/experiment.yaml]

Returns exit code 0 if all checks pass, non-zero otherwise.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


class PreFlightChecker:
    """Performs comprehensive pre-flight checks"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.errors = []
        self.warnings = []
        self.passed_checks = 0
        self.total_checks = 0

    def run_all_checks(self) -> bool:
        """
        Run all validation checks

        Returns:
            True if all checks pass, False otherwise
        """
        print_header("FILO-PRIORI PRE-FLIGHT VALIDATION")

        checks = [
            ("Python Environment", self.check_python_version),
            ("Configuration File", self.check_config_file),
            ("Required Dependencies", self.check_dependencies),
            ("PyTorch & CUDA", self.check_pytorch_cuda),
            ("Dataset Files", self.check_datasets),
            ("Directory Structure", self.check_directories),
            ("GPU Availability", self.check_gpu),
            ("Configuration Schema", self.check_config_schema),
            ("Memory Requirements", self.check_memory),
        ]

        for check_name, check_func in checks:
            print_header(f"CHECK: {check_name}")
            self.total_checks += 1
            try:
                if check_func():
                    self.passed_checks += 1
                    print_success(f"{check_name} passed")
                else:
                    print_error(f"{check_name} failed")
            except Exception as e:
                print_error(f"{check_name} raised exception: {e}")
                self.errors.append(f"{check_name}: {e}")

        # Print summary
        self._print_summary()

        return len(self.errors) == 0

    def check_python_version(self) -> bool:
        """Check Python version"""
        import sys
        version = sys.version_info

        print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")

        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.errors.append(f"Python 3.8+ required, got {version.major}.{version.minor}")
            return False

        if version.minor >= 12:
            self.warnings.append(f"Python 3.12+ may have compatibility issues with some packages")

        return True

    def check_config_file(self) -> bool:
        """Check configuration file exists and is valid YAML"""
        config_path = Path(self.config_path)

        if not config_path.exists():
            self.errors.append(f"Configuration file not found: {self.config_path}")
            return False

        print_info(f"Config file: {self.config_path}")

        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            print_info(f"Config sections: {', '.join(self.config.keys())}")
            return True

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Failed to load config: {e}")
            return False

    def check_dependencies(self) -> bool:
        """Check required Python packages are installed"""
        required_packages = {
            'torch': 'torch',
            'transformers': 'transformers',
            'sentence-transformers': 'sentence_transformers',
            'torch-geometric': 'torch_geometric',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'scipy': 'scipy',
            'yaml': 'yaml',
            'tqdm': 'tqdm',
            'networkx': 'networkx',
        }

        optional_packages = {
            'tensorboard': 'tensorboard',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'imbalanced-learn': 'imblearn',
        }

        all_ok = True

        print_info("Checking required packages...")
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
                print_info(f"  ✓ {package_name}")
            except ImportError:
                print_error(f"  ✗ {package_name} - NOT INSTALLED")
                self.errors.append(f"Missing required package: {package_name}")
                all_ok = False

        print_info("\nChecking optional packages...")
        for package_name, import_name in optional_packages.items():
            try:
                __import__(import_name)
                print_info(f"  ✓ {package_name}")
            except ImportError:
                print_warning(f"  ⚠ {package_name} - not installed (optional)")
                self.warnings.append(f"Optional package not installed: {package_name}")

        return all_ok

    def check_pytorch_cuda(self) -> bool:
        """Check PyTorch and CUDA configuration"""
        try:
            import torch

            print_info(f"PyTorch version: {torch.__version__}")

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            print_info(f"CUDA available: {cuda_available}")

            if cuda_available:
                print_info(f"CUDA version (PyTorch): {torch.version.cuda}")
                print_info(f"Number of GPUs: {torch.cuda.device_count()}")

                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    print_info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")

                # Test CUDA functionality
                try:
                    test_tensor = torch.zeros(1, device='cuda:0')
                    print_success("CUDA test tensor creation successful")
                except Exception as e:
                    self.errors.append(f"CUDA test failed: {e}")
                    return False

            else:
                # Check if CUDA is required in config
                if self.config and self.config.get('hardware', {}).get('device') == 'cuda':
                    self.errors.append(
                        "Config requires CUDA but no GPU detected. "
                        "Set hardware.device to 'cpu' or fix GPU configuration."
                    )
                    return False

            return True

        except ImportError:
            self.errors.append("PyTorch not installed")
            return False
        except Exception as e:
            self.errors.append(f"PyTorch/CUDA check failed: {e}")
            return False

    def check_datasets(self) -> bool:
        """Check dataset files exist and are valid"""
        if not self.config:
            self.warnings.append("Cannot check datasets (config not loaded)")
            return True

        data_config = self.config.get('data', {})
        all_ok = True

        for path_key in ['train_path', 'test_path']:
            path_str = data_config.get(path_key)

            if not path_str:
                self.warnings.append(f"Dataset path '{path_key}' not specified in config")
                continue

            path = Path(path_str)

            if not path.exists():
                self.errors.append(f"Dataset not found: {path_str}")
                all_ok = False
                continue

            # Check file size
            size_bytes = path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_bytes / (1024 * 1024 * 1024)

            if size_gb >= 1:
                print_info(f"{path_key}: {path_str} ({size_gb:.2f} GB)")
            else:
                print_info(f"{path_key}: {path_str} ({size_mb:.2f} MB)")

            if size_mb < 1:
                self.warnings.append(f"Dataset seems small ({size_mb:.2f} MB): {path_str}")

            # Try to read first few lines
            try:
                import pandas as pd
                df_sample = pd.read_csv(path, nrows=5)
                print_info(f"  Columns ({len(df_sample.columns)}): {', '.join(df_sample.columns[:5])}...")
                print_info(f"  Sample rows: {len(df_sample)}")
            except Exception as e:
                self.errors.append(f"Failed to read dataset {path_str}: {e}")
                all_ok = False

        return all_ok

    def check_directories(self) -> bool:
        """Check required directories exist"""
        required_dirs = ['src', 'configs']
        optional_dirs = ['cache', 'results', 'logs', 'models']

        all_ok = True

        print_info("Checking required directories...")
        for dir_name in required_dirs:
            if Path(dir_name).is_dir():
                print_info(f"  ✓ {dir_name}/")
            else:
                print_error(f"  ✗ {dir_name}/ - NOT FOUND")
                self.errors.append(f"Required directory missing: {dir_name}/")
                all_ok = False

        print_info("\nChecking optional directories (will be created if missing)...")
        for dir_name in optional_dirs:
            if Path(dir_name).is_dir():
                print_info(f"  ✓ {dir_name}/")
            else:
                print_warning(f"  ⚠ {dir_name}/ - will be created")
                try:
                    Path(dir_name).mkdir(parents=True, exist_ok=True)
                    print_info(f"    Created {dir_name}/")
                except Exception as e:
                    self.warnings.append(f"Could not create {dir_name}/: {e}")

        return all_ok

    def check_gpu(self) -> bool:
        """Detailed GPU checks"""
        try:
            import torch

            if not torch.cuda.is_available():
                if self.config and self.config.get('hardware', {}).get('device') == 'cpu':
                    print_warning("No GPU available, using CPU (slow for large models)")
                    return True
                else:
                    print_warning("No GPU available")
                    return True

            # Test GPU memory
            device = torch.device('cuda:0')
            print_info(f"Testing GPU: {torch.cuda.get_device_name(0)}")

            # Get memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_memory / (1024**3)
            print_info(f"Total GPU memory: {total_gb:.2f} GB")

            # Estimate required memory (rough)
            model_size_gb = 6  # Qodo model ~1.5B params * 4 bytes ~6GB
            batch_memory_gb = 2  # Batch processing overhead

            required_gb = model_size_gb + batch_memory_gb

            print_info(f"Estimated required memory: ~{required_gb:.1f} GB")

            if total_gb < required_gb:
                self.warnings.append(
                    f"GPU may have insufficient memory ({total_gb:.1f} GB available, "
                    f"~{required_gb:.1f} GB recommended). Consider reducing batch size."
                )

            return True

        except Exception as e:
            self.warnings.append(f"GPU check encountered error: {e}")
            return True

    def check_config_schema(self) -> bool:
        """Validate configuration against schema"""
        if not self.config:
            self.errors.append("Config not loaded, cannot validate schema")
            return False

        try:
            # Import config validator
            sys.path.insert(0, str(Path(__file__).parent / 'src'))
            from utils.config_validator import validate_config

            # Validate (non-strict to collect all errors)
            is_valid = validate_config(self.config, strict=False)

            if not is_valid:
                self.errors.append("Configuration schema validation failed (see details above)")

            return is_valid

        except ImportError as e:
            self.warnings.append(f"Config validator not available: {e}")
            return True  # Don't fail if validator missing
        except Exception as e:
            self.errors.append(f"Config validation error: {e}")
            return False

    def check_memory(self) -> bool:
        """Check system memory requirements"""
        try:
            import psutil

            # Get system memory
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)

            print_info(f"System RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")

            # Rough estimate: need ~16GB for large datasets + model
            required_gb = 16

            if available_gb < required_gb:
                self.warnings.append(
                    f"Low system memory ({available_gb:.1f} GB available, "
                    f"{required_gb} GB recommended). Consider using sampling."
                )

            return True

        except ImportError:
            print_warning("psutil not installed, cannot check system memory")
            return True
        except Exception as e:
            self.warnings.append(f"Memory check error: {e}")
            return True

    def _print_summary(self):
        """Print validation summary"""
        print_header("VALIDATION SUMMARY")

        print_info(f"Total checks: {self.total_checks}")
        print_info(f"Passed: {self.passed_checks}")
        print_info(f"Failed: {self.total_checks - self.passed_checks}")
        print_info(f"Errors: {len(self.errors)}")
        print_info(f"Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n" + Colors.RED + Colors.BOLD + "ERRORS:" + Colors.END)
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print("\n" + Colors.YELLOW + Colors.BOLD + "WARNINGS:" + Colors.END)
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        print()

        if len(self.errors) == 0:
            print_success("ALL CHECKS PASSED! Ready to run experiment.")
            print()
            return True
        else:
            print_error(f"VALIDATION FAILED with {len(self.errors)} error(s).")
            print_error("Please fix the errors above before running the experiment.")
            print()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Pre-flight validation for Filo-Priori experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment.yaml',
        help='Path to configuration file (default: configs/experiment.yaml)'
    )

    args = parser.parse_args()

    # Run checks
    checker = PreFlightChecker(args.config)
    success = checker.run_all_checks()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
