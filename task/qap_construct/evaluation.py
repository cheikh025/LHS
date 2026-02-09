# Module Name: QAPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the Quadratic Assignment Problem (QAP).
#       The QAP involves assigning a set of facilities to a set of locations in such a way that the total cost of interactions between facilities is minimized.
#       This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#   - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 20).
#   - n_facilities: Number of facilities to assign: int (default: 50).
#   - n_instance: Number of problem instances to generate: int (default: 10).
# 
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Callable, Any, List, Tuple

from base import Evaluation
from task.qap_construct.get_instance import GetData
from task.qap_construct.template import template_program, task_description

__all__ = ['QAPEvaluation']


class QAPEvaluation(Evaluation):
    """Evaluator for the Quadratic Assignment Problem."""

    def __init__(self,
                 timeout_seconds=20,
                 n_facilities=50,
                 n_instance=16,
                 **kwargs):
        """
        Initializes the QAP evaluator.
        """
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = n_instance
        self.n_facilities = n_facilities
        self.data_generator = GetData(self.n_instance, self.n_facilities)
        self._datasets = self.data_generator.generate_instances()

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        """
        Evaluates the program (constructive heuristic) for the QAP.

        Args:
            program_str: Program string (not used here, but kept for compatibility).
            callable_func: The constructive heuristic function to evaluate.

        Returns:
            The average total cost across all instances.
        """
        return self.evaluate_qap(callable_func)

    def qap_evaluate(self, current_assignment: List[int], flow_matrix: np.ndarray, distance_matrix: np.ndarray, eva: Callable) -> List[int]:
        """
        Evaluate the next assignment for the Quadratic Assignment Problem using a constructive heuristic.

        Args:
            current_assignment: Current assignment of facilities to locations.
            flow_matrix: Flow matrix between facilities.
            distance_matrix: Distance matrix between locations.
            eva: The constructive heuristic function to select the next assignment.

        Returns:
            Updated assignment of facilities to locations.
        """
        # Use the heuristic to select the next assignment

        n_facilities = flow_matrix.shape[0]
        # We assume the heuristic fills one slot at a time, or modifies the state.
        # We call it n_facilities times to ensure it has a chance to complete the assignment
        # if it's doing one-at-a-time. If it does all at once, this loop is redundant but harmless
        # provided the heuristic checks for assigned slots.
        for _ in range(n_facilities):
            # Check if full (optimization)
            if -1 not in current_assignment:
                break
            next_assignment = eva(current_assignment, flow_matrix, distance_matrix)
            current_assignment = next_assignment

        return current_assignment

    def evaluate_qap(self, eva: Callable) -> float:
        """
        Evaluate the constructive heuristic for the Quadratic Assignment Problem.

        Args:
            eva: The constructive heuristic function to evaluate.

        Returns:
            The average total cost across all instances.
        """
        total_cost = 0

        for instance in self._datasets[:self.n_instance]:
            flow_matrix, distance_matrix = instance
            n_facilities = flow_matrix.shape[0]
            current_assignment = [-1] * n_facilities  # Initialize with no assignments
            current_assignment = self.qap_evaluate(current_assignment, flow_matrix, distance_matrix, eva)

            # Check if current_assignment is a feasible solution
            if -1 in current_assignment:
                # Infeasible (incomplete)
                return None 
            if any(not (0 <= x < n_facilities) for x in current_assignment):
                # Infeasible (out of bounds)
                return None
            if len(set(current_assignment)) != n_facilities:
                # Infeasible (duplicates)
                return None

            # Calculate the total cost of the assignment
            cost = 0
            for i in range(n_facilities):
                for j in range(n_facilities):
                    cost += flow_matrix[i, j] * distance_matrix[current_assignment[i], current_assignment[j]]
            total_cost += cost

        average_cost = total_cost / self.n_instance
        return -average_cost  # We want to minimize the total cost


if __name__ == '__main__':

    def select_next_assignment(current_assignment: List[int], flow_matrix: np.ndarray, distance_matrix: np.ndarray) -> List[int]:
        """
        A greedy heuristic for the Quadratic Assignment Problem.

        Args:
            current_assignment: Current assignment of facilities to locations (-1 means unassigned).
            flow_matrix: Flow matrix between facilities.
            distance_matrix: Distance matrix between locations.

        Returns:
            Updated assignment of facilities to locations.
        """
        n_facilities = len(current_assignment)

        # Find the first unassigned facility and the first available location
        for facility in range(n_facilities):
            if current_assignment[facility] == -1:
                # Find the first available location
                for location in range(n_facilities):
                    if location not in current_assignment:
                        current_assignment[facility] = location
                        break
                break

        return current_assignment


    bp1d = QAPEvaluation()
    ave_bins = bp1d.evaluate_program('_', select_next_assignment)
    print(ave_bins)
