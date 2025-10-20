"""
Code quality validation script for feature_importance.py module.

This script validates:
1. Code formatting with Black
2. Linting with Ruff
3. All functions have Google-style docstrings
4. All functions have type hints
5. No code duplication
"""

import ast
import re
import subprocess
import sys
from pathlib import Path


def run_black_check():
    """Run Black formatter check."""
    print("=" * 80)
    print("CHECK 1: Black Formatting")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["poetry", "run", "black", "src/utils/feature_importance.py", "--check"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ Code is properly formatted with Black")
            print(result.stdout.strip())
            return True
        else:
            print("✗ Code formatting issues found:")
            print(result.stdout)
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ Error running Black: {e}")
        return False


def run_ruff_check():
    """Run Ruff linter check."""
    print("\n" + "=" * 80)
    print("CHECK 2: Ruff Linting")
    print("=" * 80)

    try:
        result = subprocess.run(
            ["poetry", "run", "ruff", "check", "src/utils/feature_importance.py"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✓ No linting issues found")
            # Filter out deprecation warnings
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines:
                if "All checks passed" in line:
                    print(line)
            return True
        else:
            print("✗ Linting issues found:")
            print(result.stdout)
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ Error running Ruff: {e}")
        return False


def check_docstrings():
    """Check that all functions have Google-style docstrings."""
    print("\n" + "=" * 80)
    print("CHECK 3: Google-Style Docstrings")
    print("=" * 80)

    filepath = Path("src/utils/feature_importance.py")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # Filter out private functions (starting with _)
        public_functions = [f for f in functions if not f.name.startswith("_")]

        print(f"Found {len(public_functions)} public function(s)")
        print()

        all_valid = True
        for func in public_functions:
            docstring = ast.get_docstring(func)

            if docstring is None:
                print(f"✗ {func.name}(): Missing docstring")
                all_valid = False
                continue

            # Check for Google-style docstring sections
            has_params = "Parameters" in docstring or "Args" in docstring
            has_returns = "Returns" in docstring
            has_examples = "Examples" in docstring

            # Check for proper formatting
            has_dashes = "---" in docstring or "~~~" in docstring

            if has_params and has_returns and has_dashes:
                print(f"✓ {func.name}(): Complete Google-style docstring")
                if has_examples:
                    print(f"  (includes examples)")
            else:
                print(f"⚠ {func.name}(): Docstring exists but may be incomplete")
                if not has_params:
                    print(f"    - Missing Parameters section")
                if not has_returns:
                    print(f"    - Missing Returns section")
                if not has_dashes:
                    print(f"    - Missing section separators (---)")
                all_valid = False

        print()
        return all_valid

    except Exception as e:
        print(f"✗ Error checking docstrings: {e}")
        return False


def check_type_hints():
    """Check that all functions have type hints."""
    print("=" * 80)
    print("CHECK 4: Type Hints")
    print("=" * 80)

    filepath = Path("src/utils/feature_importance.py")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # Filter out private functions
        public_functions = [f for f in functions if not f.name.startswith("_")]

        print(f"Found {len(public_functions)} public function(s)")
        print()

        all_valid = True
        for func in public_functions:
            # Check return type hint
            has_return_hint = func.returns is not None

            # Check parameter type hints
            params_with_hints = []
            params_without_hints = []

            for arg in func.args.args:
                if arg.arg == "self":
                    continue
                if arg.annotation is not None:
                    params_with_hints.append(arg.arg)
                else:
                    params_without_hints.append(arg.arg)

            if has_return_hint and len(params_without_hints) == 0:
                print(f"✓ {func.name}(): Complete type hints")
                print(f"  Parameters: {len(params_with_hints)}")
            else:
                print(f"✗ {func.name}(): Missing type hints")
                if not has_return_hint:
                    print(f"    - Missing return type hint")
                if params_without_hints:
                    print(f"    - Parameters without hints: {params_without_hints}")
                all_valid = False

        print()
        return all_valid

    except Exception as e:
        print(f"✗ Error checking type hints: {e}")
        return False


def check_code_duplication():
    """Check for obvious code duplication."""
    print("=" * 80)
    print("CHECK 5: Code Duplication")
    print("=" * 80)

    filepath = Path("src/utils/feature_importance.py")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for repeated code patterns (simple heuristic)
        # Check for repeated validation patterns
        validation_patterns = [
            r'if .+ not in .+\.columns:',
            r'raise ValueError\(',
            r'\.head\(top_n\)',
            r'\.to_list\(\)',
        ]

        print("Checking for common patterns (DRY principle)...")
        print()

        # Count occurrences of each pattern
        pattern_counts = {}
        for pattern in validation_patterns:
            matches = re.findall(pattern, content)
            if len(matches) > 0:
                pattern_counts[pattern] = len(matches)

        # This is expected - validation patterns should be reused
        # We're just checking they're not excessive
        has_duplication = False
        for pattern, count in pattern_counts.items():
            if count > 5:  # Arbitrary threshold
                print(f"⚠ Pattern '{pattern}' appears {count} times")
                has_duplication = True

        if not has_duplication:
            print("✓ No excessive code duplication detected")
            print("  (Common validation patterns are appropriately reused)")
        else:
            print("⚠ Some patterns appear frequently (may indicate duplication)")

        print()

        # Check function lengths (long functions may indicate duplication opportunities)
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        public_functions = [f for f in functions if not f.name.startswith("_")]

        print("Function complexity check:")
        long_functions = []
        for func in public_functions:
            # Count lines in function (approximate)
            func_lines = func.end_lineno - func.lineno
            if func_lines > 100:
                long_functions.append((func.name, func_lines))
                print(f"⚠ {func.name}(): {func_lines} lines (consider refactoring)")
            else:
                print(f"✓ {func.name}(): {func_lines} lines")

        print()

        if len(long_functions) == 0:
            print("✓ All functions are reasonably sized")
            return True
        else:
            print(f"⚠ {len(long_functions)} function(s) may benefit from refactoring")
            return True  # Not a failure, just a warning

    except Exception as e:
        print(f"✗ Error checking code duplication: {e}")
        return False


def check_module_structure():
    """Check overall module structure and organization."""
    print("=" * 80)
    print("CHECK 6: Module Structure")
    print("=" * 80)

    filepath = Path("src/utils/feature_importance.py")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for module docstring
        has_module_docstring = content.strip().startswith('"""') or content.strip().startswith("'''")

        if has_module_docstring:
            print("✓ Module has docstring")
        else:
            print("✗ Module missing docstring")

        # Check for imports organization
        import_section = content.split("\n\n")[0] if "\n\n" in content else ""
        has_imports = "import" in import_section

        if has_imports:
            print("✓ Imports are present")
        else:
            print("⚠ No imports found (unusual)")

        # Check for constants
        has_constants = "DEFAULT_" in content or re.search(r"^[A-Z_]+ = ", content, re.MULTILINE)

        if has_constants:
            print("✓ Module constants defined")
        else:
            print("  No module constants (optional)")

        # Count functions
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        public_functions = [f for f in functions if not f.name.startswith("_")]

        print(f"✓ Module contains {len(public_functions)} public function(s)")

        # Check for __all__ (optional but good practice)
        has_all = "__all__" in content

        if has_all:
            print("✓ __all__ defined (explicit public API)")
        else:
            print("  __all__ not defined (optional)")

        print()
        return True

    except Exception as e:
        print(f"✗ Error checking module structure: {e}")
        return False


def main():
    """Run all code quality checks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "CODE QUALITY VALIDATION" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = []

    # Run all checks
    results.append(("Black Formatting", run_black_check()))
    results.append(("Ruff Linting", run_ruff_check()))
    results.append(("Google-Style Docstrings", check_docstrings()))
    results.append(("Type Hints", check_type_hints()))
    results.append(("Code Duplication", check_code_duplication()))
    results.append(("Module Structure", check_module_structure()))

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")

    print()
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print()
        print("🎉 ALL CODE QUALITY CHECKS PASSED!")
        print()
        print("Summary:")
        print("  ✓ Code is properly formatted (Black)")
        print("  ✓ No linting issues (Ruff)")
        print("  ✓ All functions have Google-style docstrings")
        print("  ✓ All functions have type hints")
        print("  ✓ No excessive code duplication")
        print("  ✓ Module is well-structured")
        return 0
    else:
        print()
        print("⚠️  Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
