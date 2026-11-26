# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Python CLI (`srs_audit`) generates every artifact. The HTML report is one self-contained file produced by `src/srs_audit/reporting/html_renderer.py`: inline CSS and JS, no CDN, no build step, opens offline from disk.

## Users

- **Hackathon jury (BCC-HUB x MNU FinTech Hackathon).** Opens the report on a laptop or projector during the demo or while scoring. Needs the verdict and the proof that it is trustworthy within seconds, then digs into a few defects to check that the citations are real.
- **Bank analyst / reviewer (BA, SA, QA, PO, InfoSec).** Uses the report as a working tool before development starts: finds defects in the source document, filters by severity and role, copies clarifying questions into Jira, checks API ↔ business mismatches.

Both readers matter equally (confirmed by the user).

## Product Purpose

Static AI audit of a software requirements document (SRS) against ISO/IEC/IEEE 29148, IEEE 830 and Kazakhstan banking practice. Produces a GO / CONDITIONAL GO / NO-GO verdict, a quality index 0–100, defects anchored to exact lines of the source, and a backlog of decision questions per role with EARS rewrites. Success: a reviewer trusts the verdict and can act on it without opening the source document separately.

## Positioning

Every quoted citation in the report is verified by deterministic code against the source text (0 % hallucinated quotes, unconfirmed hypotheses quarantined, not shown as defects). The hybrid of deterministic rules and typed LLM agents is the mechanism; the Citation Audit badge is its proof.

## Operating Context

Report languages: Russian UI and content. Opened from `reports/report.html` in a browser, often projected. Also shipped as Markdown and PDF with identical content.

## Capabilities and Constraints

- Content is generated per run: number of requirements, defects, questions and table rows varies (tens to hundreds); long quotes, code fragments and JSON appear in source text.
- Severities: Blocker, Critical, Major, Minor. Verdicts: GO, CONDITIONAL GO, NO-GO, UNPROCESSABLE.
- Must work with no network access, in light and dark system themes.

## Brand Commitments

None. The user explicitly left branding open; do not use BCC or MNU logos or brand palettes.

## Evidence on Hand

Real run data only (reports/report.json-equivalent model `AuditReport`). No testimonials, benchmarks or customer claims may be invented.

## Product Principles

1. Proof before claims: every number and defect links back to the exact line it came from.
2. Verdict first, depth on demand.
3. Actionable over decorative: everything a reviewer needs to act (fix text, question, Jira copy) is one click away.
4. Honest degradation: when an LLM stage failed, the report says so.
