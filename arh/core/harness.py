"""
ARH Reliability Harness

Main orchestrator for running reliability tests on agents.
Provides unified interface for running individual tests or full suite.
"""

from typing import List, Optional, Dict
from .agent_wrapper import AgentWrapper
from .models import TestResult, TrustReport, TestStatus
from ..tests.robustness import RobustnessTest
from ..tests.consistency import ConsistencyTest
from ..tests.groundedness import GroundednessTest
from ..tests.predictability import PredictabilityTest
import uuid
from datetime import datetime


class ReliabilityHarness:
    """
    Main orchestrator for running reliability tests on agents.
    
    Provides methods for running individual tests, full test suites,
    and generating comprehensive reports.
    """
    
    def __init__(self, agent: AgentWrapper):
        """
        Initialize the reliability harness.
        
        Args:
            agent: The agent wrapper to test
        """
        self.agent = agent
        self.results: Dict[str, TestResult] = {}
    
    def run_all(self, prompts: List[str]) -> Dict[str, TestResult]:
        """
        Run all reliability tests.
        
        Args:
            prompts: List of prompts to test with
            
        Returns:
            Dictionary mapping test names to results
        """
        tests = [
            RobustnessTest(),
            ConsistencyTest(),
            GroundednessTest(),
            PredictabilityTest()
        ]
        
        for test in tests:
            result = test.run(self.agent, prompts)
            self.results[result.name] = result
        
        return self.results
    
    def run_test(self, test_name: str, prompts: List[str], **kwargs) -> TestResult:
        """
        Run a specific test.
        
        Args:
            test_name: Name of test to run (robustness, consistency, groundedness, predictability)
            prompts: List of prompts to test with
            **kwargs: Additional arguments to pass to the test constructor
            
        Returns:
            TestResult for the specified test
        """
        test_map = {
            "robustness": RobustnessTest,
            "consistency": ConsistencyTest,
            "groundedness": GroundednessTest,
            "predictability": PredictabilityTest
        }
        
        if test_name not in test_map:
            raise ValueError(f"Unknown test: {test_name}. Available: {list(test_map.keys())}")
        
        test = test_map[test_name](**kwargs)
        result = test.run(self.agent, prompts)
        self.results[result.name] = result
        
        return result
    
    def get_overall_score(self) -> float:
        """
        Calculate weighted overall score.
        
        Weights:
        - Groundedness: 30% (most important for trust)
        - Robustness: 25%
        - Consistency: 25%
        - Predictability: 20%
        
        Returns:
            Weighted overall score (0-1)
        """
        if not self.results:
            return 0.0
        
        weights = {
            "robustness": 0.25,
            "consistency": 0.25,
            "groundedness": 0.30,
            "predictability": 0.20
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for name, result in self.results.items():
            if result.status == TestStatus.SKIPPED:
                continue  # e.g. groundedness with no source context — undefined, not zero
            weight = weights.get(name, 0.25)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_verdict(self) -> str:
        """
        Get deployment verdict based on results.
        
        Returns:
            - "PASS": Score >= 0.85 with no critical failures
            - "CONDITIONAL_PASS": Score >= 0.70
            - "BLOCK": Score < 0.70 or critical failures
        """
        score = self.get_overall_score()
        
        # Check for any critical failures
        has_critical = any(
            r.status == TestStatus.FAIL and r.score < 0.5
            for r in self.results.values()
        )
        
        if has_critical:
            return "BLOCK"
        elif score >= 0.85:
            return "PASS"
        elif score >= 0.70:
            return "CONDITIONAL_PASS"
        else:
            return "BLOCK"
    
    def generate_report(self) -> Dict:
        """
        Generate full reliability report.
        
        Returns:
            Dictionary containing complete assessment results
        """
        return {
            "assessment_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent.model,
            "overall_score": round(self.get_overall_score(), 3),
            "verdict": self.get_verdict(),
            "dimensions": {
                name: {
                    "score": round(result.score, 3),
                    "status": result.status.value,
                    "details": result.details,
                    "failures": result.failures,
                    "recommendations": result.recommendations
                }
                for name, result in self.results.items()
            }
        }
    
    def clear_results(self):
        """Clear all stored test results."""
        self.results = {}
    
    def print_summary(self):
        """Print a human-readable summary of results."""
        if not self.results:
            print("No test results available. Run tests first.")
            return
        
        print("\n" + "=" * 60)
        print("RELIABILITY ASSESSMENT SUMMARY")
        print("=" * 60)
        print(f"Agent: {self.agent.model}")
        print(f"Overall Score: {self.get_overall_score():.1%}")
        print(f"Verdict: {self.get_verdict()}")
        print("-" * 60)
        
        for name, result in self.results.items():
            status_icon = "✅" if result.status == TestStatus.PASS else "❌"
            print(f"{status_icon} {name.capitalize()}: {result.score:.1%}")
            
            if result.failures:
                for failure in result.failures[:3]:
                    print(f"   └─ {failure}")
        
        print("=" * 60)
