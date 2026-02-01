# Agent Reliability Harness (ARH)

> SRE Principles for AI Agents - Because "It Usually Works" Isn't Good Enough

ARH is a production-oriented trust and evaluation layer for LLM agents. Instead of asking "Is this model smart?", ARH asks **"Is this agent safe to deploy?"**

## Quick Start

```bash
# Clone and install
git clone https://github.com/akhilkinnera01/agent-reliability-harness.git
cd agent-reliability-harness
pip install -r requirements.txt

# Test an agent
python -m arh.cli.main test --agent https://api.openai.com/v1/chat/completions \
       --api-key $OPENAI_API_KEY

# Audit documentation
python -m arh.cli.main audit ./docs/safety_manual.md --simple

# With LLM-powered analysis
python -m arh.cli.main audit ./docs/safety_manual.md --api-key $OPENAI_API_KEY
```

## The Four Dimensions of Reliability

| Dimension | Question | Method |
|-----------|----------|--------|
| **Robustness** | Does it break with typos? | Prompt perturbation |
| **Consistency** | Same question → same answer? | Multi-sample variance |
| **Groundedness** | Is it making things up? | Hallucination detection |
| **Predictability** | Stable under load? | Latency distribution |

## Adversarial Auditor

Inspired by Meta's Dr. Zero paper, ARH includes an **Adversarial Auditor** that finds documentation flaws by generating questions that SHOULD be answerable but ARE NOT.

```bash
python -m arh.cli.main audit ./lab_manual.md --hops 1,2,3
```

### Flaw Types Detected

| Flaw | Severity | Description |
|------|----------|-------------|
| `SAFETY_GAP` | Critical | Missing safety information |
| `MISSING_PREREQ` | High | Missing prerequisites |
| `AMBIGUOUS` | Medium | Unclear language |
| `IMPLICIT_ASSUMPTION` | Medium | Unstated assumptions |
| `TEMPORAL_GAP` | Low | Missing sequence info |

## Usage Examples

### Test Agent Reliability

```python
from arh.core.agent_wrapper import OpenAIWrapper
from arh.core.harness import ReliabilityHarness

agent = OpenAIWrapper(api_key="your-key")
harness = ReliabilityHarness(agent)

prompts = ["What is 2 + 2?", "What is the capital of France?"]
harness.run_all(prompts)

print(f"Score: {harness.get_overall_score():.1%}")
print(f"Verdict: {harness.get_verdict()}")
```

### Audit Documentation

```python
from arh.auditor import AdversarialAuditor
from arh.core.agent_wrapper import OpenAIWrapper

agent = OpenAIWrapper(api_key="your-key")
auditor = AdversarialAuditor(proposer_model=agent)

report = auditor.audit_file("./documentation.md")
auditor.print_report(report)
```

### Export Metrics

```python
from arh.metrics import MetricsExporter

exporter = MetricsExporter(system_name="my-agent")
exporter.export_agent_results("gpt-4", report)
print(exporter.get_metrics().decode())
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `arh test --agent URL` | Run reliability tests |
| `arh audit file.md` | Audit documentation |
| `arh trust-eval` | Combined trust evaluation |
| `arh version` | Show version |

## Project Structure

```
agent-reliability-harness/
├── arh/
│   ├── core/           # Agent wrapper, models, harness
│   ├── tests/          # Reliability tests (4 dimensions)
│   ├── auditor/        # Adversarial documentation auditor
│   ├── metrics/        # Prometheus metrics export
│   └── cli/            # Command-line interface
├── examples/           # Demo scripts
└── docs/               # Documentation
```

## Research Lineage

ARH's Adversarial Auditor is inspired by the proposer-solver framework from:

> Yue et al. "Dr. Zero: Self-Evolving Search Agents without Training Data" — arXiv:2601.07055 (January 2026)

We apply the insight that "partial solver failure indicates interesting problems" to documentation quality, rather than model training.

## Requirements

- Python 3.9+
- Dependencies: `httpx`, `typer`, `rich`, `numpy`, `prometheus_client`

## License

MIT License

---

**Built with ❤️ for reliable AI agents**
