"""
ARH Core Data Models

Defines the fundamental data structures used throughout the Agent Reliability Harness.
Includes enums for test status, flaw types, severity levels, and dataclasses for
responses, test results, findings, and reports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class TestStatus(Enum):
    """Possible outcomes for a reliability test."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    ERROR = "error"
    SKIPPED = "skipped"


class FlawType(Enum):
    """Categories of flaws that can be detected in documents or agent behavior."""
    AMBIGUOUS = "ambiguous"
    MISSING_PREREQ = "missing_prerequisite"
    IMPLICIT_ASSUMPTION = "implicit_assumption"
    CONTRADICTION = "contradiction"
    TEMPORAL_GAP = "temporal_gap"
    SAFETY_GAP = "safety_gap"


class Severity(Enum):
    """Severity levels for findings and issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentResponse:
    """
    Normalized response from any LLM agent endpoint.
    
    Captures the response content along with performance metrics
    and metadata for analysis.
    """
    content: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


@dataclass
class TestResult:
    """
    Result of a reliability test run.
    
    Contains the score, status, detailed breakdown of findings,
    and recommendations for improvement.
    """
    name: str
    score: float
    status: TestStatus
    details: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Finding:
    """
    A specific issue found during document auditing.
    
    Includes location, classification, and remediation guidance.
    """
    line: int
    text: str
    flaw_type: FlawType
    severity: Severity
    question: str
    solver_response: str
    recommendation: str


@dataclass
class AuditReport:
    """
    Complete audit report for a document or section.
    
    Aggregates all findings with an overall score.
    """
    document: str
    section: str
    overall_score: float
    findings: List[Finding]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrustReport:
    """
    Trust assessment report combining agent and knowledge scores.
    
    Provides a verdict on whether the agent can be trusted
    along with any blockers and recommendations.
    """
    assessment_id: str
    agent_score: float
    knowledge_score: float
    trust_score: float
    verdict: str
    blockers: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
