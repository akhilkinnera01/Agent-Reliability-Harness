<div align="center">

# 🛡️ Agent Reliability Harness (ARH)

### SRE for AI Agents — Trust, but Verify

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

### 📊 Premium Dashboard Output

<div align="center">
<img src="examples/UI_1.png" alt="ARH Dashboard" width="500"/>
</div>

<div align="center">
<img src="examples/UI_2.png" alt="ARH Dashboard" width="500"/>
</div>
---

**Before you deploy an AI agent to production, ask: How reliable is it really?**

<img src="examples/architecture_1.png" alt="ARH Architecture" width="400"/>

</div>

---

## 🚀 What is ARH?

ARH is an **end-to-end reliability testing framework** for AI agents. It applies Site Reliability Engineering (SRE) principles to answer the question: *"Is this AI agent safe to deploy?"*

<div align="center">
<img src="examples/flow_1.png" alt="ARH Pipeline" width="200"/>
</div>

### The Problem

AI agents are increasingly making real-world decisions, but we lack standardized ways to measure their reliability:

- ❌ Do they hallucinate under pressure?
- ❌ Are their responses consistent?
- ❌ Can they handle adversarial inputs?
- ❌ Is the knowledge base they use complete?

### The Solution

ARH provides a **Trust Report** that combines:

| Component | What It Measures |
|-----------|------------------|
| **Agent Reliability** | How the model behaves (robustness, consistency, groundedness) |
| **Documentation Quality** | How complete the knowledge base is (finds gaps and flaws) |
| **Trust Score** | Combined metric for deployment readiness |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔬 Reliability Testing
- **Robustness** — Prompt perturbation resistance
- **Consistency** — Response variance analysis
- **Groundedness** — Hallucination detection
- **Predictability** — Latency profiling

</td>
<td width="50%">

### 🎯 Adversarial Auditor
- **Proposer** — Generates adversarial questions
- **Solver** — Document-constrained answering
- **Evaluator** — Flaw classification & severity

</td>
</tr>
</table>



## 🏃 Quick Start

### Installation

```bash
git clone https://github.com/akhilkinnera01/agent-reliability-harness.git
cd agent-reliability-harness
pip install -r requirements.txt
```

### Run the Demo

```bash
# Set your API key
export GEMINI_API_KEY="your-key"

# Run premium demo
python3 examples/demo_premium.py
```

### Audit Any Document

```bash
# Supports PDF, DOCX, EPUB, Markdown, and more!
python3 examples/run_on_file.py your_document.pdf
```

---

## 📖 Usage

### Test an AI Agent

```python
from arh.core import UniversalWrapper, ReliabilityHarness

# Create agent wrapper (supports 100+ models via LiteLLM)
agent = UniversalWrapper(model="gemini/gemini-2.5-flash", api_key="...")

# Run reliability tests
harness = ReliabilityHarness(agent)
harness.run_test("robustness", prompts=["What is 2+2?", "Explain quantum computing"])
harness.run_test("consistency", prompts=["What is the capital of France?"])
harness.run_test("groundedness", prompts=["Who invented the telephone?"])

# Get report
report = harness.generate_report()
print(f"Trust Score: {report['overall_score']:.1%}")
print(f"Verdict: {report['verdict']}")
```

### Audit Documentation

```python
from arh.core import UniversalWrapper
from arh.auditor import AdversarialAuditor
from arh.document_loader import load_document

# Load any document format
doc = load_document("safety_manual.pdf")

# Run adversarial audit
agent = UniversalWrapper(model="gemini/gemini-2.5-flash", api_key="...")
auditor = AdversarialAuditor(proposer_model=agent)
report = auditor.audit(doc.content, document_name=doc.filename)

# View findings
for finding in report.findings:
    print(f"[{finding.severity.value}] {finding.flaw_type.value}")
    print(f"  Question: {finding.question}")
    print(f"  Recommendation: {finding.recommendation}")
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Reliability Harness                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐   │
│  │   AI Agent   │───▶│ Reliability Tests │───▶│ Trust Report │   │
│  │  (Any LLM)   │    │  • Robustness     │    │  • Score     │   │
│  └──────────────┘    │  • Consistency    │    │  • Verdict   │   │
│                      │  • Groundedness   │    │  • Findings  │   │
│  ┌──────────────┐    │  • Predictability │    └──────────────┘   │
│  │  Documents   │───▶├──────────────────┤                        │
│  │ (PDF/DOCX/)  │    │ Adversarial      │                        │
│  │  EPUB/MD)    │    │ Auditor          │                        │
│  └──────────────┘    │  • Proposer      │                        │
│                      │  • Solver        │                        │
│                      │  • Evaluator     │                        │
│                      └──────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔌 Supported Models

ARH uses **LiteLLM** to support 100+ AI models:

| Provider | Models |
|----------|--------|
| **Google** | Gemini 2.5 Flash, Gemini 2.0, Gemini 1.5 Pro |
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5 Turbo |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus |
| **Groq** | Llama 3.1, Mixtral |
| **Ollama** | Any local model |
| **+90 more** | AWS Bedrock, Azure, Cohere, etc. |

---

## 📂 Project Structure

```
agent-reliability-harness/
├── arh/
│   ├── core/           # Agent wrappers, models, harness
│   ├── tests/          # Reliability tests
│   ├── auditor/        # Adversarial auditor components
│   ├── cli/            # CLI commands
│   ├── metrics/        # Prometheus exporter
│   ├── dashboard.py    # Premium visual output
│   └── document_loader.py  # Multi-format loader
├── examples/           # Demo scripts
├── docs/               # Documentation
└── assets/             # Architecture diagrams
```

---

## 📚 Research Lineage

ARH's Adversarial Auditor is inspired by the **Dr. Zero** research paper:

> *Dr. Zero: Self-Evolving Search Agents without Training Data* — arXiv:2601.07055 (January 2026)

We apply its insight that "partial solver failure indicates interesting
problems" to documentation quality rather than model training.

---

## 🔬 Validation

A score you cannot validate is just an opinion. ARH ships a meta-evaluation
harness (`benchmarks/`) that runs the metrics against **labeled data** and
reports how well each score tracks ground truth — the number ARH itself should
be judged on.

The robustness/consistency backbone is scored against a labeled similarity set;
groundedness is scored against a sample of **[HaluEval](https://github.com/RUCAIBox/HaluEval)**
(Li et al., 2023) — the standard hallucination-detection benchmark — so the
numbers are comparable to published work rather than self-defined.

| Metric | Backend | Dataset | AUC | Precision / Recall |
|--------|---------|---------|-----|--------------------|
| Similarity (robustness/consistency) | `all-MiniLM-L6-v2` | 12 paraphrase / off-topic / contradiction pairs | **0.72** | 0.75 / 1.00 |
| Groundedness | `gpt-4o-mini` | HaluEval QA, 500 cases | **0.87** | 0.79 / 0.99 |

### Model comparison — the harness must *discriminate*

A reliability layer is only worth anything if it ranks a weaker agent below a
stronger one on the same data. **Six cheap models from three providers**, scored
on the **same 500 HaluEval QA cases**:

| Provider | Judge model | AUC | Precision | Recall | F1 |
|----------|-------------|-----|-----------|--------|-----|
| OpenAI | `gpt-3.5-turbo` | 0.757 | 0.727 | 0.840 | 0.779 |
| OpenAI | **`gpt-4o-mini`** | **0.869** | 0.794 | 0.988 | **0.881** |
| OpenAI | `gpt-4.1-mini` | 0.805 | 0.766 | 0.892 | 0.824 |
| Google | `gemini-2.5-flash-lite` | 0.787 | 0.770 | 0.872 | 0.818 |
| Google | `gemini-2.5-flash` | 0.814 | 0.772 | 0.868 | 0.817 |
| Anthropic | `claude-haiku-4-5` | 0.703 | 0.612 | 0.928 | 0.738 |

<p align="center">
  <img src="benchmarks/plots/comparison.png" width="640"/><br/>
  <img src="benchmarks/plots/roc.png" width="430"/>
  <img src="benchmarks/plots/distributions.png" width="430"/>
</p>

What the data actually says — tested for significance with a 2000-sample
bootstrap (95% CIs are the error bars above), nothing tuned to a target:

- **`gpt-4o-mini` leads the field** (AUC 0.87, CI [0.84, 0.90]) with the cleanest
  score separation — hallucinated answers collapse to 0, grounded ones to 1.
- **Newer ≠ better on a narrow task.** `gpt-4o-mini` (2024) beats the *later*
  `gpt-4.1-mini` (2025) — and the gap **is** real (paired bootstrap P=1.00).
- **`claude-haiku-4-5` trails** (AUC 0.70) with high recall / low precision: it
  *over-trusts* answers, calling hallucinations grounded — real (P=0.97 vs
  gpt-3.5-turbo). A behavioral signal, not a bug; exactly what the harness exists
  to surface.
- **What is _not_ significant:** `gemini-2.5-flash` vs `flash-lite` (0.814 vs
  0.787) sits inside the noise (P=0.84) — at n=500 we can't call that a real
  difference, so we don't.

**What these numbers are (and aren't).** They measure how well each model, used
as a *judge*, separates grounded from hallucinated answers — i.e. each model's
skill as a faithfulness **detector**, not how much it itself hallucinates. The
hallucinations in HaluEval are model-generated (synthetic), which the literature
shows is *easier* than organic, real-world hallucinations; expect lower AUC on
production data (e.g. RAGTruth). AUCs land in the **0.70–0.87** band, consistent
with published LLM-detector results on HaluEval — realistic, not saturated.

Reproduce:

```bash
pip install -e .[semantic] matplotlib
python -m benchmarks.build_dataset 250                     # 500 HaluEval cases (verbatim)
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
  python -m benchmarks.run_comparison gpt-4o-mini gemini/gemini-2.5-flash anthropic/claude-haiku-4-5
python -m benchmarks.plot_results                          # regenerate the figures (free)
```

The residual similarity errors are **contradiction pairs** ("turn left" vs
"turn right") — topically near-identical, so cosine rates them similar. That gap
is exactly why groundedness uses **NLI entailment**, not cosine. CI enforces the
similarity baseline and fails on regression (`.github/workflows/ci.yml`);
groundedness/auditor self-skip without an API key.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

```bash
# Run tests
pytest

# Run linting
ruff check arh/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for reliable AI**

[Quick Start](Start.md) • [Examples](examples/)

</div>
