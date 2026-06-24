"""
ARH Adversarial Auditor Module

Main orchestrator for adversarial documentation auditing.
Implements the Dr. Zero proposer-solver loop for finding doc flaws.
"""

from typing import List, Dict, Optional
from ..core.agent_wrapper import AgentWrapper
from ..core.models import AuditReport, Finding, FlawType, Severity
from .proposer import Proposer, HopComplexity
from .solver import Solver
from .evaluator import Evaluator
from datetime import datetime
import json


class AdversarialAuditor:
    """
    Main orchestrator for adversarial documentation auditing.
    Implements the Dr. Zero proposer-solver loop for finding doc flaws.
    """
    
    def __init__(
        self,
        proposer_model: AgentWrapper,
        solver_model: AgentWrapper = None,
        hop_complexity: List[HopComplexity] = None,
        flaw_types: List[FlawType] = None
    ):
        """
        Initialize the adversarial auditor.
        
        Args:
            proposer_model: LLM wrapper for generating questions
            solver_model: LLM wrapper for answering (defaults to proposer_model)
            hop_complexity: Complexity levels to test
            flaw_types: Filter for specific flaw types (None = all)
        """
        self.proposer = Proposer(proposer_model)
        self.solver = Solver(solver_model or proposer_model)
        self.evaluator = Evaluator()
        self.hop_complexity = hop_complexity or [
            HopComplexity.ONE, 
            HopComplexity.TWO
        ]
        self.flaw_types = flaw_types  # Filter for specific flaws
        self.proposer_model = proposer_model
    
    def audit(
        self,
        document: str,
        sections: List[Dict[str, str]] = None,
        document_name: str = "document"
    ) -> AuditReport:
        """
        Audit a document for flaws.
        
        Args:
            document: Full document text
            sections: Optional list of {"name": str, "content": str}
                     If not provided, treats entire doc as one section
            document_name: Name of the document for reporting
            
        Returns:
            AuditReport with findings and score
        """
        if sections is None:
            sections = [{"name": "full_document", "content": document}]
        
        all_findings: List[Finding] = []
        
        for section in sections:
            section_findings = self._audit_section(
                document=document,
                section_name=section["name"],
                section_content=section["content"]
            )
            all_findings.extend(section_findings)
        
        # Filter by flaw type if specified
        if self.flaw_types:
            all_findings = [
                f for f in all_findings 
                if f.flaw_type in self.flaw_types
            ]
        
        # Calculate overall score
        score = self._calculate_score(all_findings, document)
        
        return AuditReport(
            document=document_name,
            section="all",
            overall_score=score,
            findings=all_findings,
            timestamp=datetime.now()
        )
    
    def _audit_section(
        self,
        document: str,
        section_name: str,
        section_content: str
    ) -> List[Finding]:
        """Audit a single section of the document."""
        findings = []
        
        # Generate adversarial questions
        questions = self.proposer.generate_questions(
            document=document,
            section=section_content,
            hop_complexity=self.hop_complexity,
            questions_per_hop=3
        )
        
        # Test each question with solver
        for question in questions:
            solver_response = self.solver.answer(
                question=question.get("question", ""),
                document=document
            )
            
            # Evaluate for flaw
            finding = self.evaluator.evaluate(
                question=question,
                solver_response=solver_response,
                section_text=section_content
            )
            
            if finding:
                findings.append(finding)
        
        return findings
    
    def _calculate_score(self, findings: List[Finding], document: str) -> float:
        """Calculate document reliability score based on findings."""
        if not findings:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            Severity.CRITICAL: 0.25,
            Severity.HIGH: 0.15,
            Severity.MEDIUM: 0.08,
            Severity.LOW: 0.03
        }
        
        total_penalty = sum(
            severity_weights.get(f.severity, 0.05)
            for f in findings
        )
        
        # Cap penalty at 0.8 (minimum score of 0.2)
        score = max(0.2, 1.0 - total_penalty)
        
        return round(score, 3)
    
    def audit_file(self, filepath: str) -> AuditReport:
        """
        Convenience method to audit a file directly.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            AuditReport for the file
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Use filename as document name
        import os
        doc_name = os.path.basename(filepath)
        
        return self.audit(content, document_name=doc_name)
    
    def audit_simple(self, document: str, document_name: str = "document") -> AuditReport:
        """
        DEMO-ONLY — NOT A RELIABILITY SIGNAL. Keyword-matching audit with no
        LLM; flaw findings are illustrative only. Use audit() with an LLM
        proposer for real results.

        Args:
            document: Document text to audit
            document_name: Name for reporting
            
        Returns:
            AuditReport based on simple keyword analysis
        """
        findings = []
        
        # Generate simple questions
        questions = self.proposer.generate_questions_simple(document[:500], num_questions=5)
        
        # Answer using simple matching
        for question in questions:
            solver_response = self.solver.answer_simple(
                question=question.get("question", ""),
                document=document
            )
            
            # Evaluate for flaw
            finding = self.evaluator.evaluate(
                question=question,
                solver_response=solver_response,
                section_text=document[:500]
            )
            
            if finding:
                findings.append(finding)
        
        score = self._calculate_score(findings, document)
        
        return AuditReport(
            document=document_name,
            section="all",
            overall_score=score,
            findings=findings,
            timestamp=datetime.now()
        )
    
    def generate_report_dict(self, report: AuditReport) -> Dict:
        """
        Convert AuditReport to a dictionary for JSON serialization.
        
        Args:
            report: AuditReport to convert
            
        Returns:
            Dictionary representation
        """
        return {
            "document": report.document,
            "section": report.section,
            "overall_score": report.overall_score,
            "timestamp": report.timestamp.isoformat(),
            "findings_count": len(report.findings),
            "findings": [
                {
                    "line": f.line,
                    "text": f.text,
                    "flaw_type": f.flaw_type.value,
                    "severity": f.severity.value,
                    "question": f.question,
                    "recommendation": f.recommendation
                }
                for f in report.findings
            ],
            "summary": self.evaluator.summarize_findings(report.findings)
        }
    
    def print_report(self, report: AuditReport):
        """Print a human-readable report to console."""
        print("\n" + "=" * 60)
        print("DOCUMENT AUDIT REPORT")
        print("=" * 60)
        print(f"Document: {report.document}")
        print(f"Score: {report.overall_score:.1%}")
        print(f"Findings: {len(report.findings)}")
        print("-" * 60)
        
        if not report.findings:
            print("✅ No documentation flaws detected!")
        else:
            for i, finding in enumerate(report.findings, 1):
                severity_icons = {
                    Severity.CRITICAL: "🚨",
                    Severity.HIGH: "⚠️",
                    Severity.MEDIUM: "📝",
                    Severity.LOW: "💡"
                }
                icon = severity_icons.get(finding.severity, "•")
                print(f"\n{icon} Finding {i}: {finding.flaw_type.value}")
                print(f"   Line {finding.line}: {finding.text[:60]}...")
                print(f"   Question: {finding.question[:60]}...")
                print(f"   Recommendation: {finding.recommendation}")
        
        print("\n" + "=" * 60)
