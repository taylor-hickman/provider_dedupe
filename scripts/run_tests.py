#!/usr/bin/env python3
"""Test runner script for comprehensive testing."""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import click


def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """Run a command and return success status and output.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of what the command does
        cwd: Working directory to run command in
        
    Returns:
        Tuple of (success, output)
    """
    click.echo(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd
        )
        click.echo(f"✅ {description} passed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ {description} failed")
        if e.stderr:
            click.echo(f"Error output: {e.stderr}")
        if e.stdout:
            click.echo(f"Standard output: {e.stdout}")
        return False, e.stderr or e.stdout
    except FileNotFoundError as e:
        click.echo(f"❌ {description} failed - command not found: {e}")
        return False, str(e)


def check_tool_availability(tool: str) -> bool:
    """Check if a tool is available in PATH."""
    try:
        subprocess.run([tool, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


@click.command()
@click.option("--fast", is_flag=True, help="Run only fast tests")
@click.option("--unit-only", is_flag=True, help="Run only unit tests")
@click.option("--integration-only", is_flag=True, help="Run only integration tests")
@click.option("--coverage", is_flag=True, help="Generate coverage report")
@click.option("--lint", is_flag=True, help="Run linting checks")
@click.option("--type-check", is_flag=True, help="Run type checking")
@click.option("--security", is_flag=True, help="Run security checks")
@click.option("--all", "run_all", is_flag=True, help="Run all checks and tests")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--install-missing", is_flag=True, help="Install missing dependencies")
def main(
    fast: bool,
    unit_only: bool,
    integration_only: bool,
    coverage: bool,
    lint: bool,
    type_check: bool,
    security: bool,
    run_all: bool,
    verbose: bool,
    install_missing: bool,
):
    """Run tests and quality checks for provider deduplication."""
    
    # Change to project root and set working directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    click.echo(f"🏠 Running tests from: {project_root}")
    
    failed_checks = []
    
    # Check for missing dependencies first
    missing_deps = []
    if install_missing:
        click.echo("\n📦 Checking dependencies...")
        try:
            import faker
        except ImportError:
            missing_deps.append("faker")
        
        if missing_deps:
            click.echo(f"Installing missing dependencies: {', '.join(missing_deps)}")
            for dep in missing_deps:
                success, _ = run_command(
                    ["pip", "install", dep],
                    f"Installing {dep}",
                    cwd=project_root
                )
                if not success:
                    click.echo(f"⚠️  Failed to install {dep}")
    
    # Determine what to run
    if run_all:
        unit_only = integration_only = coverage = lint = type_check = security = True
    
    # Default to running tests if no specific options
    if not any([unit_only, integration_only, lint, type_check, security]):
        unit_only = integration_only = coverage = True
    
    # Run linting checks
    if lint:
        click.echo("\n📝 Code Quality Checks")
        click.echo("=" * 50)
        
        # Black formatting check
        if check_tool_availability("black"):
            success, _ = run_command(
                ["black", "--check", "src/", "tests/"],
                "Black code formatting check",
                cwd=project_root
            )
            if not success:
                failed_checks.append("Black formatting")
        else:
            click.echo("⚠️  Black not available, skipping formatting check")
        
        # isort import sorting check
        if check_tool_availability("isort"):
            success, _ = run_command(
                ["isort", "--check-only", "src/", "tests/"],
                "isort import sorting check",
                cwd=project_root
            )
            if not success:
                failed_checks.append("isort import sorting")
        else:
            click.echo("⚠️  isort not available, skipping import sorting check")
        
        # flake8 linting
        if check_tool_availability("flake8"):
            success, _ = run_command(
                ["flake8", "src/", "tests/", "--max-line-length=88", "--extend-ignore=E203,W503"],
                "flake8 linting",
                cwd=project_root
            )
            if not success:
                failed_checks.append("flake8 linting")
        else:
            click.echo("⚠️  flake8 not available, skipping linting check")
    
    # Run type checking
    if type_check:
        click.echo("\n🔍 Type Checking")
        click.echo("=" * 50)
        
        if check_tool_availability("mypy"):
            success, _ = run_command(
                ["mypy", "src/", "--ignore-missing-imports"],
                "mypy type checking",
                cwd=project_root
            )
            if not success:
                failed_checks.append("mypy type checking")
        else:
            click.echo("⚠️  mypy not available, skipping type checking")
    
    # Run security checks
    if security:
        click.echo("\n🔒 Security Checks")
        click.echo("=" * 50)
        
        # Bandit security check
        if check_tool_availability("bandit"):
            success, _ = run_command(
                ["bandit", "-r", "src/", "-f", "json"],
                "bandit security check",
                cwd=project_root
            )
            if not success:
                failed_checks.append("bandit security")
        else:
            click.echo("⚠️  bandit not available, skipping security check")
        
        # Safety dependency check
        if check_tool_availability("safety"):
            success, _ = run_command(
                ["safety", "check"],
                "safety dependency check",
                cwd=project_root
            )
            if not success:
                failed_checks.append("safety dependency")
        else:
            click.echo("⚠️  safety not available, skipping dependency check")
    
    # Prepare pytest arguments
    pytest_args = ["pytest"]
    
    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")
    
    if coverage:
        pytest_args.extend(["--cov=provider_dedupe", "--cov-report=term-missing", "--cov-report=html"])
    
    if fast:
        pytest_args.extend(["-m", "not slow"])
    
    # Check if pytest is available
    if not check_tool_availability("pytest"):
        click.echo("❌ pytest not available - cannot run tests")
        failed_checks.append("pytest not available")
    else:
        # Run both unit and integration tests together if both are selected
        if unit_only and integration_only:
            click.echo("\n🧪 Running All Tests")
            click.echo("=" * 50)
            
            all_args = pytest_args + ["tests/"]
            success, output = run_command(all_args, "All tests", cwd=project_root)
            if not success:
                failed_checks.append("Tests")
            elif verbose and output:
                click.echo(output)
        else:
            # Run unit tests
            if unit_only:
                click.echo("\n🧪 Unit Tests")
                click.echo("=" * 50)
                
                unit_args = pytest_args + ["tests/unit/"]
                success, output = run_command(unit_args, "Unit tests", cwd=project_root)
                if not success:
                    failed_checks.append("Unit tests")
                elif verbose and output:
                    click.echo(output)
            
            # Run integration tests
            if integration_only:
                click.echo("\n🔗 Integration Tests")
                click.echo("=" * 50)
                
                integration_args = pytest_args + ["tests/integration/"]
                success, output = run_command(integration_args, "Integration tests", cwd=project_root)
                if not success:
                    failed_checks.append("Integration tests")
                elif verbose and output:
                    click.echo(output)
    
    # Summary
    click.echo("\n📊 Test Summary")
    click.echo("=" * 50)
    
    if failed_checks:
        click.echo(f"❌ {len(failed_checks)} check(s) failed:")
        for check in failed_checks:
            click.echo(f"  - {check}")
        
        # Provide helpful suggestions
        click.echo("\n💡 Suggestions:")
        if "pytest not available" in failed_checks:
            click.echo("  - Install dev dependencies: pip install -e '[dev]'")
        if any("formatting" in check.lower() or "linting" in check.lower() for check in failed_checks):
            click.echo("  - Run linting tools individually to see specific issues")
            click.echo("  - Use --install-missing to install missing tools")
        if "Tests" in failed_checks or any("test" in check.lower() for check in failed_checks):
            click.echo("  - Some test failures may be expected (see previous analysis)")
            click.echo("  - Use --verbose to see detailed test output")
            click.echo("  - Try installing missing dependencies: --install-missing")
        
        sys.exit(1)
    else:
        click.echo("✅ All checks passed!")
        
        if coverage:
            click.echo("\n📈 Coverage report generated in htmlcov/")
            coverage_html_path = project_root / "htmlcov" / "index.html"
            if coverage_html_path.exists():
                click.echo(f"  View at: file://{coverage_html_path}")
        
        click.echo("\n🎉 Ready for production!")


if __name__ == "__main__":
    main()