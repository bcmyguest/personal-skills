# vuln-app (test fixture)

A deliberately-flawed miniature Django-style app used to test the `security-audit`
and `efficiency-audit` skills. It contains a small number of realistic planted defects
mixed with benign code. **Not runnable, not real** — it exists only for auditing.

Ground truth (the plants a good auditor should surface) lives in the eval metadata, not
here, so audit agents pointed at this directory don't get the answers for free.
