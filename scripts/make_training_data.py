"""Generates the synthetic training data for the two classifiers:

  data/training/clauses.jsonl  - one labelled clause per line (clause-type classifier)
  data/training/docs.jsonl     - one labelled document per line (document-type classifier)

The templates below were written separately from the evaluation documents in data/samples/, with
different wording, so the evaluation measures how well the models generalise beyond the templates.

    python scripts/make_training_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "training"
R = random.Random(42)

CITIES = ["Delhi", "Kolkata", "Jaipur", "Lucknow", "Kochi", "Nagpur", "Indore", "Mysuru", "Vijayawada", "Surat"]
NAMES = ["Ravi Shankar", "Neha Gupta", "Imran Khan", "Asha Menon", "Pooja Reddy", "Vikram Singh", "Joseph Mathew",
         "Sunita Rao", "Deepak Jain", "Lavanya Pillai"]
COS = ["Ganga Credit Ltd", "Sahyadri Finserv Pvt Ltd", "Brahmaputra Leasing Ltd", "Narmada Tech Pvt Ltd",
       "Kalinga Foods LLP", "Indus Logistics Pvt Ltd", "Godavari Infra Ltd", "Malabar Traders"]


def amt():
    return R.choice(["Rs. {:,}/-".format(R.randint(1, 90) * 10000), "Rs {} lakh".format(R.randint(1, 80)),
                     "INR {:,}".format(R.randint(2, 99) * 5000), "₹{:,}".format(R.randint(3, 60) * 10000)])


def pct():
    return f"{R.choice([7, 8.5, 9.25, 10, 11.5, 12, 14, 15, 18, 19.5, 22, 24])}%"


def n(a, b):
    return str(R.randint(a, b))


def date():
    return f"{R.randint(1, 28)} {R.choice(['January', 'March', 'May', 'July', 'September', 'November'])} {R.randint(2023, 2027)}"


def pick(*opts):
    return R.choice(opts)


# label -> (heading choices, body generators)
T = {
    "definitions": (["DEFINITIONS", "INTERPRETATION", "Definitions and Interpretation"], [
        lambda: f"In this agreement, unless the context otherwise requires, \"{pick('Facility', 'Loan', 'Premises', 'Information')}\" shall mean {pick('the amount sanctioned herein', 'the property described in the schedule', 'all data shared under this agreement')}, and words in the singular include the plural.",
        lambda: "Capitalised terms used in this agreement shall have the meanings given to them in this clause. Headings are for convenience only and shall not affect interpretation.",
        lambda: f"\"Business Day\" means a day on which banks are open in {pick(*CITIES)}, and \"Confidential Information\" means any information marked or understood to be confidential."]),
    "principal": (["LOAN AMOUNT", "SANCTION", "FACILITY", "Amount of Facility", "DISBURSEMENT"], [
        lambda: f"The {pick('Bank', 'Company', 'Lender')} hereby sanctions a loan of {amt()} to the Borrower, to be disbursed {pick('in one tranche', 'in two tranches', 'directly to the seller')}.",
        lambda: f"Subject to the terms herein, the Lender shall advance to the Borrower a sum of {amt()} for {pick('purchase of a vehicle', 'business expansion', 'medical expenses', 'higher education')}.",
        lambda: f"The facility amount is {amt()}, which shall be credited to the Borrower's account after execution of this agreement and submission of documents."]),
    "interest": (["INTEREST", "RATE OF INTEREST", "Interest Rate", "PRICING"], [
        lambda: f"The loan shall bear interest at {pct()} per annum on a {pick('monthly reducing', 'daily reducing', 'flat')} balance basis.",
        lambda: f"Interest at the rate of {pct()} p.a. shall be charged, {pick('which is fixed for the tenure', 'linked to the MCLR of the Lender', 'subject to reset every year')}.",
        lambda: f"The applicable rate is {pct()} per annum. {pick('The Lender may change the rate with notice to the Borrower.', 'The rate shall remain unchanged during the tenure.', '')}"]),
    "repayment": (["REPAYMENT", "TENURE", "Repayment Terms", "EMI"], [
        lambda: f"The Borrower shall repay the loan in {n(6, 240)} monthly instalments of {amt()} each, commencing from {date()}.",
        lambda: f"The tenure of the loan is {n(1, 20)} years and repayment shall be by equated monthly instalments through {pick('NACH mandate', 'standing instruction', 'post-dated cheques')}.",
        lambda: f"Instalments of {amt()} shall be paid on the {n(1, 28)}th of each month until the loan is fully repaid."]),
    "prepayment": (["PREPAYMENT", "FORECLOSURE", "Pre-closure", "Part Payment"], [
        lambda: f"The Borrower may foreclose the loan {pick('at any time', 'after 6 instalments', 'after one year')} on payment of a foreclosure charge of {pick('2%', '3%', '5%', 'nil')} on the outstanding principal.",
        lambda: "Part prepayment is permitted, subject to a minimum of two instalments, and the tenure shall be reduced accordingly.",
        lambda: "No charges shall be levied for prepayment of a floating rate loan by an individual borrower."]),
    "penal_charges": (["PENAL INTEREST", "LATE PAYMENT", "Default Interest", "Overdue Charges"], [
        lambda: f"Delayed instalments shall attract penal charges of {pick('1%', '2%', '3%')} per month on the overdue amount.",
        lambda: f"In case of delay in payment, additional interest at {pick('2', '4', '6')}% per annum shall be levied on the amount in arrears till payment.",
        lambda: f"A late payment fee of {amt()} shall be charged for each delayed instalment, in addition to penal interest."]),
    "default": (["EVENTS OF DEFAULT", "DEFAULT", "Consequences of Default"], [
        lambda: f"If the Borrower fails to pay {pick('any instalment', 'two instalments', 'any amount')} on the due date, the Lender may declare the entire outstanding immediately payable.",
        lambda: "Any of the following constitutes default: non-payment, breach of covenant, insolvency, or misrepresentation by the Borrower.",
        lambda: "Upon default, the Lender shall be entitled to recall the loan, enforce the security and report the default to credit information companies."]),
    "security": (["SECURITY", "GUARANTEE", "Collateral", "Hypothecation"], [
        lambda: f"The loan shall be secured by {pick('hypothecation of the vehicle', 'mortgage of the property', 'pledge of fixed deposits', 'a personal guarantee')} in favour of the Lender.",
        lambda: f"{pick(*NAMES)} shall stand as guarantor and shall be jointly and severally liable with the Borrower.",
        lambda: "The original title documents shall be deposited with the Lender and returned after closure of the loan."]),
    "set_off": (["SET-OFF", "LIEN", "Right of Set Off"], [
        lambda: "The Lender may adjust any dues against any money of the Borrower lying with it, without prior notice.",
        lambda: "The Bank shall have a general lien and right of set-off over all accounts and deposits of the Borrower held with it.",
        lambda: "Any credit balance of the Borrower may be appropriated by the Lender towards the outstanding dues."]),
    "property": (["PREMISES", "SCHEDULE PROPERTY", "Description of Property", "THE PROPERTY"], [
        lambda: f"The Owner lets out the {pick('first floor flat', 'independent house', 'office space', 'godown')} situated at {n(1, 99)}, {pick('Station Road', 'Market Street', 'Lake Road')}, {pick(*CITIES)}, together with fittings.",
        lambda: f"The premises measure {n(300, 3000)} square feet and include {pick('one car parking', 'two bedrooms', 'a store room')}.",
        lambda: "The schedule property is more fully described in the schedule annexed hereto."]),
    "term": (["TERM", "PERIOD OF TENANCY", "Duration", "LOCK-IN", "Term and Survival"], [
        lambda: f"This lease shall be valid for {n(11, 60)} months from {date()}, with a lock-in of {n(3, 24)} months.",
        lambda: f"The agreement shall remain in force for {n(1, 5)} years and may be extended by written consent.",
        lambda: f"The confidentiality obligations shall survive for {n(1, 7)} years after termination of this agreement.",
        lambda: f"The licence is granted for a period of {n(11, 36)} months commencing on {date()}."]),
    "rent": (["RENT", "LICENCE FEE", "Monthly Rent", "Payment of Rent"], [
        lambda: f"The Tenant shall pay rent of {amt()} per month in advance by the {n(1, 10)}th of each month.",
        lambda: f"The monthly licence fee is {amt()}, payable by bank transfer, exclusive of {pick('GST', 'maintenance', 'utilities')}.",
        lambda: f"Rent of {amt()} per month shall be paid without any deduction."]),
    "deposit": (["SECURITY DEPOSIT", "DEPOSIT", "Refundable Deposit", "ADVANCE"], [
        lambda: f"The Tenant has paid an interest free refundable deposit of {amt()}, refundable {pick('within 30 days of vacating', 'on handing over possession', 'after adjusting dues')}.",
        lambda: f"A sum of {amt()} has been paid as advance, which the Landlord shall return at the end of the tenancy.",
        lambda: "The deposit shall not carry interest and shall be adjusted against arrears, if any, on termination."]),
    "escalation": (["ESCALATION", "RENT REVISION", "Increase in Rent", "Enhancement"], [
        lambda: f"The rent shall be increased by {pick('5%', '7%', '10%', '12%')} every {pick('year', '11 months', 'two years')}.",
        lambda: f"On renewal, the rent shall be revised upward by {pick('5', '8', '10')} percent.",
        lambda: "Rent shall be revised annually by mutual agreement of the parties."]),
    "maintenance": (["MAINTENANCE", "REPAIRS", "Utilities", "Upkeep"], [
        lambda: "The Tenant shall pay electricity, water and gas charges and keep the premises in good condition.",
        lambda: "Day-to-day repairs shall be done by the Tenant, while structural repairs and painting shall be done by the Owner.",
        lambda: "Society maintenance charges shall be borne by the Licensee during the licence period."]),
    "use": (["USE", "PURPOSE OF LETTING", "Use of Premises", "RESTRICTIONS"], [
        lambda: f"The premises shall be used only for {pick('residence', 'office', 'retail shop', 'storage')} and for no other purpose.",
        lambda: "The Tenant shall not sublet, assign or part with possession of the premises or make structural alterations.",
        lambda: "The Licensee shall not store hazardous goods or cause nuisance to neighbours."]),
    "termination": (["TERMINATION", "Termination and Notice", "VACATING"], [
        lambda: f"Either party may terminate this agreement by giving {pick('one month', 'two months', '30 days', '60 days')} written notice.",
        lambda: f"The Owner may terminate the lease if rent is unpaid for {n(2, 3)} months, after giving {n(15, 30)} days' notice.",
        lambda: "This agreement may be terminated by either party on breach of its terms which is not remedied within 30 days."]),
    "confidentiality": (["CONFIDENTIALITY", "OBLIGATIONS", "Non-Disclosure", "Duty of Confidence"], [
        lambda: "The Recipient shall keep all confidential information secret and shall use it solely for the permitted purpose.",
        lambda: "Neither party shall disclose the other party's confidential information to any third person without prior written consent.",
        lambda: "The receiving party shall restrict access to confidential information to employees who need to know it."]),
    "exclusions": (["EXCLUSIONS", "EXCEPTIONS", "Permitted Disclosure"], [
        lambda: "Information which is in the public domain or was lawfully known to the recipient before disclosure is excluded.",
        lambda: "Disclosure required by any court order or law shall not be a breach, provided prompt notice is given.",
        lambda: "The obligations shall not apply to information independently developed by the recipient."]),
    "non_compete": (["NON-COMPETE", "NON-SOLICITATION", "Restrictive Covenants"], [
        lambda: f"For {n(6, 36)} months after termination, the {pick('Employee', 'Consultant', 'Recipient')} shall not engage in any competing business.",
        lambda: f"During the term and for {n(6, 24)} months thereafter, neither party shall solicit the employees or customers of the other.",
        lambda: "The Consultant shall not join a competitor of the Company during the engagement."]),
    "indemnity": (["INDEMNITY", "INDEMNIFICATION"], [
        lambda: "The Recipient shall indemnify the Discloser against all claims, losses and expenses arising from any breach of this agreement.",
        lambda: "Each party shall indemnify the other against third-party claims caused by its negligence.",
        lambda: "The Tenant shall indemnify the Owner for any damage caused to the premises by the Tenant's acts."]),
    "liability_cap": (["LIMITATION OF LIABILITY", "LIABILITY", "Cap on Liability"], [
        lambda: f"The total liability of either party shall not exceed {amt()}.",
        lambda: "Neither party shall be liable for any indirect, special or consequential damages.",
        lambda: "Liability under this agreement is limited to the fees paid in the twelve months preceding the claim."]),
    "remedies": (["REMEDIES", "INJUNCTION", "Equitable Relief"], [
        lambda: "The parties agree that damages may not be an adequate remedy and the disclosing party may seek specific performance or an injunction.",
        lambda: "In addition to other remedies, the aggrieved party shall be entitled to injunctive relief.",
        lambda: "Remedies under this agreement are cumulative and not exclusive."]),
    "return_of_info": (["RETURN OF INFORMATION", "Return of Materials", "DESTRUCTION"], [
        lambda: f"Within {n(7, 30)} days of a request, the recipient shall return or destroy all confidential material and certify the same.",
        lambda: "On termination, all documents and copies containing confidential information shall be returned.",
        lambda: "The recipient shall delete all electronic copies on completion of the purpose."]),
    "purpose": (["PURPOSE", "BACKGROUND", "Recitals"], [
        lambda: f"The parties wish to discuss a possible {pick('joint venture', 'vendor engagement', 'software development project', 'acquisition')} and may exchange information for that purpose.",
        lambda: "The Discloser intends to share certain information with the Recipient to evaluate a business opportunity.",
        lambda: "The Company proposes to engage the Consultant for a project and will share information for that purpose."]),
    "disputes": (["GOVERNING LAW", "JURISDICTION", "ARBITRATION", "Dispute Resolution"], [
        lambda: f"This agreement shall be governed by Indian law and the courts in {pick(*CITIES)} shall have exclusive jurisdiction.",
        lambda: f"Disputes shall be referred to arbitration under the Arbitration and Conciliation Act, 1996, with the seat at {pick(*CITIES)}.",
        lambda: "The parties shall first try to resolve disputes amicably within 30 days before initiating legal proceedings."]),
    "notices": (["NOTICES", "Communication", "Addresses for Notice"], [
        lambda: "All notices shall be in writing and delivered by hand, courier, registered post or email.",
        lambda: "Notice sent to the address mentioned above shall be deemed served on the third day after dispatch.",
        lambda: "Either party may change its address for notices by informing the other in writing."]),
    "facts": ([], [
        lambda: f"My client {pick('sold goods', 'advanced a hand loan', 'rendered services', 'let out premises')} to you on {date()} and you agreed to pay for the same.",
        lambda: f"A sum of {amt()} is outstanding from you, and you have not paid it despite several reminders.",
        lambda: f"You issued cheque number {n(100000, 999999)} for {amt()}, which was returned unpaid by your bank with the remark \"{pick('Account Closed', 'Exceeds Arrangement', 'Payment Stopped by Drawer')}\".",
        lambda: f"You are in occupation of the premises as a tenant on a monthly rent of {amt()} and have defaulted in payment of rent.",
        lambda: "Under the terms agreed between you and my client, interest is payable on delayed payments."]),
    "demand": ([], [
        lambda: f"I hereby call upon you to pay the sum of {amt()} within {pick('7', '10', '15', '21', '30')} days of receipt of this notice.",
        lambda: f"You are hereby required to pay the dues with interest and vacate the premises within {pick('15', '30', '60')} days from the receipt of this notice.",
        lambda: f"My client demands that you settle the outstanding amount of {amt()} within {pick('15', '30')} days."]),
    "consequences": ([], [
        lambda: "Failing which my client shall initiate appropriate civil and criminal proceedings against you at your risk as to costs.",
        lambda: "If you fail to comply, my client will file a suit for recovery before the competent court, and you will be liable for costs.",
        lambda: "In default, my client will be constrained to take legal action against you without further notice."]),
}

PREAMBLES = {
    "loan_agreement": [
        lambda: f"LOAN AGREEMENT\nThis Loan Agreement is made on {date()} between {pick(*COS)} (the \"Lender\") and {pick(*NAMES)} (the \"Borrower\").",
        lambda: f"FACILITY AGREEMENT\nThis agreement dated {date()} is between {pick(*COS)}, a lender, and {pick(*NAMES)}, the borrower, for a term loan.",
    ],
    "rent_agreement": [
        lambda: f"RENT AGREEMENT\nThis agreement is made on {date()} between {pick(*NAMES)} (the \"Owner\") and {pick(*NAMES)} (the \"Tenant\") for letting out residential premises.",
        lambda: f"LEAVE AND LICENCE AGREEMENT\nThis licence is executed on {date()} between {pick(*NAMES)} (\"Licensor\") and {pick(*COS)} (\"Licensee\").",
    ],
    "nda": [
        lambda: f"NON-DISCLOSURE AGREEMENT\nThis agreement is effective from {date()} between {pick(*COS)} (\"Discloser\") and {pick(*COS)} (\"Recipient\").",
        lambda: f"CONFIDENTIALITY AGREEMENT\nThis agreement dated {date()} is made between {pick(*COS)} and {pick(*NAMES)}, a consultant.",
    ],
    "legal_notice": [
        lambda: f"LEGAL NOTICE\nDate: {date()}\nTo: {pick(*NAMES)}\nUnder instructions from my client {pick(*COS)}, I issue you this notice.",
        lambda: f"Advocate's Notice\nDated {date()}\nOn behalf of my client {pick(*NAMES)}, I hereby give you notice as follows.",
    ],
}
DOC_CLAUSES = {
    "loan_agreement": ["definitions", "principal", "interest", "repayment", "prepayment", "penal_charges", "default",
                       "security", "set_off", "disputes", "notices"],
    "rent_agreement": ["property", "term", "rent", "deposit", "escalation", "maintenance", "use", "termination",
                       "disputes", "notices"],
    "nda": ["purpose", "definitions", "confidentiality", "exclusions", "term", "non_compete", "indemnity",
            "liability_cap", "remedies", "return_of_info", "disputes", "notices"],
    "legal_notice": ["facts", "facts", "demand", "consequences"],
}


def clause(label: str, with_heading: float = 0.6) -> str:
    heads, bodies = T[label]
    body = R.choice(bodies)()
    if heads and R.random() < with_heading:
        h = R.choice(heads)
        return f"{h.upper() if R.random() < 0.6 else h}\n{body}"
    return body


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "clauses.jsonl").open("w") as f:
        for label in T:
            for _ in range(120):
                f.write(json.dumps({"text": clause(label), "label": label}) + "\n")
    with (OUT / "docs.jsonl").open("w") as f:
        for dtype, labels in DOC_CLAUSES.items():
            for _ in range(80):
                chosen = [lb for lb in labels if R.random() < 0.75] or labels[:2]
                body = "\n".join(f"{i}. {clause(lb)}" for i, lb in enumerate(chosen, 1))
                f.write(json.dumps({"text": R.choice(PREAMBLES[dtype])() + "\n" + body, "label": dtype}) + "\n")
    print("wrote", OUT / "clauses.jsonl", OUT / "docs.jsonl")


if __name__ == "__main__":
    main()
