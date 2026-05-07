"""Runners that orchestrate model × intervention × benchmark evaluation."""

from .eval_runner import run_evaluation, print_comparison_table

__all__ = ["run_evaluation", "print_comparison_table"]
