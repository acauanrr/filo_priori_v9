"""
Code Tokenization Utilities for Filo-Priori V10.

This module provides specialized tokenizers for code, including:
1. CamelCase splitting for Java/JavaScript identifiers
2. Snake_case splitting for Python identifiers
3. Combined preprocessing for CodeBERT input

The goal is to maximize the semantic signal from code identifiers,
which are often highly informative for test-code relationships.

Example:
    >>> splitter = CamelCaseSplitter()
    >>> splitter.split("AbstractFactoryTest")
    ['abstract', 'factory', 'test']

    >>> splitter.split("PaymentProcessingService")
    ['payment', 'processing', 'service']
"""

import re
from typing import List, Optional
from functools import lru_cache


class CamelCaseSplitter:
    """
    Splits CamelCase and PascalCase identifiers into tokens.

    This is crucial for code understanding as identifiers like
    "UserAuthenticationService" contain semantic meaning that
    is lost if treated as a single token.

    Handles:
    - PascalCase: MyClassName -> ['my', 'class', 'name']
    - camelCase: myMethodName -> ['my', 'method', 'name']
    - SCREAMING_CASE: MY_CONSTANT -> ['my', 'constant']
    - Mixed: XMLParser -> ['xml', 'parser']
    - Numbers: Test123 -> ['test', '123']
    """

    # Regex patterns for splitting
    CAMEL_PATTERN = re.compile(r'''
        # Match sequences of:
        [A-Z]?[a-z]+     |  # lowercase word (optionally starting with uppercase)
        [A-Z]+(?=[A-Z]|$) |  # uppercase sequence at end or before another uppercase
        [A-Z]+(?=[a-z])   |  # uppercase sequence before lowercase
        \d+                   # numbers
    ''', re.VERBOSE)

    SNAKE_PATTERN = re.compile(r'[_\-\s]+')

    def __init__(
        self,
        lowercase: bool = True,
        min_token_length: int = 1,
        filter_numbers: bool = False
    ):
        """
        Args:
            lowercase: Convert all tokens to lowercase.
            min_token_length: Minimum length for a token to be included.
            filter_numbers: Whether to exclude numeric tokens.
        """
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.filter_numbers = filter_numbers

    @lru_cache(maxsize=10000)
    def split(self, identifier: str) -> List[str]:
        """
        Split an identifier into semantic tokens.

        Args:
            identifier: The code identifier to split.

        Returns:
            List of tokens.

        Examples:
            >>> s = CamelCaseSplitter()
            >>> s.split("getUserById")
            ['get', 'user', 'by', 'id']
            >>> s.split("XMLHttpRequest")
            ['xml', 'http', 'request']
            >>> s.split("test_user_creation")
            ['test', 'user', 'creation']
        """
        if not identifier:
            return []

        # First split by underscores/dashes/spaces
        parts = self.SNAKE_PATTERN.split(identifier)

        tokens = []
        for part in parts:
            if not part:
                continue

            # Then split CamelCase
            matches = self.CAMEL_PATTERN.findall(part)
            tokens.extend(matches)

        # Post-process
        result = []
        for token in tokens:
            if len(token) < self.min_token_length:
                continue

            if self.filter_numbers and token.isdigit():
                continue

            if self.lowercase:
                token = token.lower()

            result.append(token)

        return result

    def split_qualified_name(self, qualified_name: str) -> List[str]:
        """
        Split a fully qualified name (e.g., package.class.method).

        Args:
            qualified_name: Dot-separated qualified name.

        Returns:
            All tokens from all parts.

        Example:
            >>> s.split_qualified_name("com.example.UserService.createUser")
            ['com', 'example', 'user', 'service', 'create', 'user']
        """
        parts = qualified_name.split('.')
        tokens = []
        for part in parts:
            tokens.extend(self.split(part))
        return tokens


class CodeTokenizer:
    """
    Full tokenizer for code text, combining identifier splitting
    with natural language processing.

    This tokenizer is designed to preprocess code for CodeBERT input,
    maximizing semantic signal while maintaining compatibility with
    the model's tokenizer.
    """

    def __init__(
        self,
        splitter: Optional[CamelCaseSplitter] = None,
        max_tokens: int = 512,
        add_special_tokens: bool = True
    ):
        self.splitter = splitter or CamelCaseSplitter()
        self.max_tokens = max_tokens
        self.add_special_tokens = add_special_tokens

        # Common code keywords to preserve
        self.keywords = {
            'java': {
                'public', 'private', 'protected', 'class', 'interface',
                'void', 'static', 'final', 'abstract', 'extends', 'implements',
                'return', 'if', 'else', 'for', 'while', 'try', 'catch', 'throw',
                'new', 'this', 'super', 'null', 'true', 'false'
            },
            'python': {
                'def', 'class', 'self', 'return', 'if', 'else', 'elif',
                'for', 'while', 'try', 'except', 'finally', 'with', 'as',
                'import', 'from', 'None', 'True', 'False', 'and', 'or', 'not'
            }
        }

    def tokenize_test_name(self, test_name: str) -> str:
        """
        Tokenize a test name for CodeBERT input.

        Converts: "TestUserAuthentication::testLoginSuccess"
        To: "test user authentication test login success"

        This format maximizes the semantic understanding by CodeBERT.
        """
        # Split by common separators
        parts = re.split(r'[.:@#$]+', test_name)

        all_tokens = []
        for part in parts:
            tokens = self.splitter.split(part)
            all_tokens.extend(tokens)

        return ' '.join(all_tokens)

    def tokenize_code_change(
        self,
        file_path: str,
        changed_lines: Optional[List[str]] = None
    ) -> str:
        """
        Tokenize a code change (file + optional diff lines).

        Args:
            file_path: Path to the changed file.
            changed_lines: Optional list of changed code lines.

        Returns:
            Tokenized string for CodeBERT input.
        """
        tokens = []

        # Extract filename and split
        filename = file_path.split('/')[-1]
        if '.' in filename:
            filename = filename.rsplit('.', 1)[0]
        tokens.extend(self.splitter.split(filename))

        # Process changed lines if provided
        if changed_lines:
            for line in changed_lines[:10]:  # Limit to first 10 lines
                # Remove comments and strings (simplified)
                line = re.sub(r'//.*$', '', line)
                line = re.sub(r'/\*.*?\*/', '', line)
                line = re.sub(r'"[^"]*"', '', line)
                line = re.sub(r"'[^']*'", '', line)

                # Extract identifiers
                identifiers = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', line)
                for ident in identifiers:
                    tokens.extend(self.splitter.split(ident))

        # Deduplicate while preserving order
        seen = set()
        unique_tokens = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        # Limit length
        result = ' '.join(unique_tokens[:self.max_tokens])

        return result

    def prepare_pair_input(
        self,
        test_name: str,
        code_change: str
    ) -> str:
        """
        Prepare input for CodeBERT with test and code as a pair.

        Format: [CLS] test tokens [SEP] code tokens [SEP]
        """
        test_tokens = self.tokenize_test_name(test_name)
        code_tokens = code_change

        if self.add_special_tokens:
            return f"[CLS] {test_tokens} [SEP] {code_tokens} [SEP]"
        else:
            return f"{test_tokens} {code_tokens}"


class JavaTestTokenizer(CodeTokenizer):
    """
    Specialized tokenizer for Java test files.

    Handles Java-specific patterns like:
    - JUnit annotations (@Test, @Before)
    - Package names (com.example.MyTest)
    - Method signatures
    """

    # Common test annotations to highlight
    TEST_ANNOTATIONS = {
        '@Test', '@Before', '@After', '@BeforeEach', '@AfterEach',
        '@BeforeAll', '@AfterAll', '@Ignore', '@Disabled'
    }

    def tokenize_java_test(self, test_class: str, method: Optional[str] = None) -> str:
        """
        Tokenize a Java test identifier.

        Args:
            test_class: Fully qualified class name.
            method: Optional method name.

        Returns:
            Tokenized string.
        """
        # Extract class name from qualified name
        if '.' in test_class:
            package_parts = test_class.split('.')
            class_name = package_parts[-1]
            package_tokens = []
            for p in package_parts[:-1]:
                package_tokens.extend(self.splitter.split(p))
        else:
            class_name = test_class
            package_tokens = []

        class_tokens = self.splitter.split(class_name)

        if method:
            method_tokens = self.splitter.split(method)
        else:
            method_tokens = []

        # Combine with semantic markers
        all_tokens = []

        if package_tokens:
            all_tokens.extend(package_tokens)

        all_tokens.extend(class_tokens)

        if method_tokens:
            all_tokens.append('method')  # Semantic marker
            all_tokens.extend(method_tokens)

        return ' '.join(all_tokens)
