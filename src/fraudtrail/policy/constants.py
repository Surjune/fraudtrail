"""Thresholds from Fraud Policy v1.0 (dataset README).

Each constant cites the section or rule it comes from. Where the policy leaves a
value open, the constant is marked as an interpretation and says why.
"""

from __future__ import annotations

# R1: a case resting on a single signal below this probability must be verified before
# any block.
R1_SINGLE_SIGNAL_BLOCK_MIN_PROBABILITY = 0.70

# R1: "a single signal" means at most this many independent pieces of evidence.
R1_SINGLE_SIGNAL_MAX_EVIDENCE = 1

# 3a: open a case once fraud probability reaches this level.
CASE_OPEN_MIN_PROBABILITY = 0.30

# 6: stop when probability is at or beyond these bounds, supported by at least this many
# independent pieces of evidence.
STOP_FRAUD_MIN_PROBABILITY = 0.85
STOP_LEGITIMATE_MAX_PROBABILITY = 0.15
STOP_MIN_INDEPENDENT_EVIDENCE = 2

# 2: BLOCK_CARD needs L1 approval up to this exposure and L2 above it.
BLOCK_CARD_L1_MAX_EXPOSURE_USD = 2_500.0

# R2 and 3a: exposure above this makes a suspicious activity report mandatory.
REPORT_EXPOSURE_THRESHOLD_USD = 1_000.0

# R4 and R8: exposure above this requires escalation to an analyst.
ESCALATION_EXPOSURE_THRESHOLD_USD = 500.0

# R4: a verification request unanswered after this many hours counts as no reply.
VERIFICATION_REPLY_WINDOW_HOURS = 24

# R5: BLOCK_CARD when a purchase above this amount has already cleared after testing.
CARD_TESTING_CLEARED_PURCHASE_BLOCK_USD = 100.0

# R10: BLOCK_ALL_CARDS needs at least this many of the customer's cards with confirmed fraud
# (or confirmed credential compromise).
BLOCK_ALL_CARDS_MIN_CONFIRMED_CARDS = 2

# Interpretation. 3a files a report when fraud is "confirmed or strongly suspected". We use
# the R1 blocking threshold as "strongly suspected", so a report is never filed on a
# probability too weak to justify blocking the card.
STRONG_SUSPICION_MIN_PROBABILITY = R1_SINGLE_SIGNAL_BLOCK_MIN_PROBABILITY

# Interpretation. The verdict bands reuse policy thresholds: at or above the R1 blocking
# threshold is fraud; below the 3a case-opening threshold is legitimate; in between is
# uncertain, which the README accepts on ambiguous cases when R1 and R8 are followed.
VERDICT_FRAUD_MIN_PROBABILITY = R1_SINGLE_SIGNAL_BLOCK_MIN_PROBABILITY
VERDICT_LEGITIMATE_BELOW_PROBABILITY = CASE_OPEN_MIN_PROBABILITY
