# Design notes

## 1. Why classical ML plus rules, with the LLM as an optional, checked helper

Legal review needs answers that are exact, explainable and repeatable. The same document must always
produce the same deposit amount and the same red flags. So:

| Task | Approach | Reason |
|---|---|---|
| Document and clause type | TF-IDF + logistic regression | Fast (about 2 s to train, milliseconds to predict), gives a confidence, easy to retrain, no GPU |
| Field values | Rules scoped to the classified clause | Exact, auditable, and every value has its quote |
| Risk review | YAML playbook | Lawyers can change thresholds without code; every flag shows its evidence |
| Gaps, summaries, Q&A, reply drafting | LLM (optional) | Handles wording the rules do not cover, but every output is checked |

Clause classification is what makes simple rules accurate. "The first amount in the deposit clause" is
reliable; "the first amount in the document" is not. Clause labels are also restricted by document
type, so a loan agreement can never have a "deposit" clause.

## 2. Checking LLM output

- **Gap filling.** The model is asked only for fields the rules missed. Its value is accepted only if its
  quote appears in the document word for word (ignoring case and spacing), **and** the value is then
  re-parsed from that quote. The model's own value is never used directly.
- **Summary, answers and reply drafts.** Every amount, percentage and date in the output must exist in
  the document (or in the user's instructions). Otherwise the output is discarded and the offline version
  is shown, with a warning for reply drafts.
- **Answers** must cite clauses that were actually given to the model, and are checked against those
  clauses.
- **Inputs.** Contact details and ID numbers are masked before sending. Document text is wrapped as data,
  and the prompt tells the model to ignore instructions inside it.

## 3. Indian-specific parsing

- Rupee amounts: "Rs. 5,00,000/-", "Rs 45 lakh", "₹1.5 crore", and Indian digit grouping in the output.
- Percentages keep their period, so "3% per month" becomes 36% p.a. for comparisons.
- Dates in dd/mm/yyyy order ("05.08.2026" is 5 August), and "5th day of March, 2026".
- Durations written in words ("one month's notice"), and full stops after "Rs.", "No." and initials are
  not treated as sentence ends.

## 4. Dates and deadlines

Deadlines are calculated from the document, not guessed:

- last EMI = first EMI + (tenure − 1) months;
- lock-in end = start + lock-in;
- last day to give notice = expiry − notice period;
- reply deadline = notice date + days given.

For notices, the period legally runs from receipt, so the app says so and counts from the notice date
only as an estimate. "Today" can be fixed with `NYAYALENS_TODAY`, so tests and the evaluation are
reproducible.

## 5. Evaluation

The evaluation covers 12 hand-labelled documents (82 clauses, 96 fields, 23 expected flags, 14 dates) and
23 questions, including questions the document cannot answer, which must be declined. CI gates on every
metric.

The classifiers are trained only on template-generated text. The evaluation documents were written
separately, so the classifier scores measure generalisation from templates to more natural documents. The
two remaining clause errors are left visible rather than patched into the training templates.

## 6. What I would change for production

| Area | Change |
|---|---|
| Clause classification | Fine-tune a small transformer (for example, LoRA on a legal BERT or a small Llama) on labelled real clauses, and keep the linear model as a fast fallback |
| Layout | Layout-aware parsing (Textract or Azure Document Intelligence) for tables, stamps and two-column layouts |
| Languages | Hindi and Tamil documents |
| Playbook | Expand it with practising advocates, and version it with an approval workflow |
| Platform | Authentication, per-user storage, encryption at rest and a retention policy for uploaded documents |
