"""Detector thresholds.

Each value cites the policy's pattern description or the measurement in
docs/data_findings.md that set it. Values marked "interpretation" are ours, chosen to be
defensible rather than tuned to an answer key we cannot see.
"""

from __future__ import annotations

# Pattern 1 and R5: "three or more tiny online authorizations, often under $5", within an
# hour, followed by a larger purchase.
CARD_TESTING_SMALL_MAX_USD = 5.0
CARD_TESTING_MIN_SMALL_AUTHS = 3
CARD_TESTING_WINDOW_MINUTES = 60

# Pattern 2: card-not-present fraud arrives "in a burst of two to four within 48 hours".
CNP_BURST_HOURS = 48
CNP_BURST_MIN_TXNS = 2

# Interpretation: an amount this many standard deviations above the card's own mean is
# "inconsistent with the cardholder's amounts". Two standard deviations keeps ordinary
# spending out.
AMOUNT_UNUSUAL_Z = 2.0

# Interpretation: a card with fewer transactions than this has no usable baseline, so
# amount and product novelty carry no weight.
BASELINE_MIN_TXNS = 8

# Pattern 4: "Several days of purchases in one new region is a trip, not a clone."
TRIP_MIN_DAYS = 2
# Interpretation of "while their normal activity continues at home": home-region use
# within this many hours of the out-of-region transaction means the card is in two places.
CLONE_HOME_ACTIVITY_HOURS = 24

# R7 recurring charges. The dataset's amount offsets stay within about $0.10
# (docs/data_findings.md), and "monthly" needs repeats across separate months.
RECURRING_TOLERANCE_USD = 0.15
RECURRING_MIN_OCCURRENCES = 3
RECURRING_MIN_MONTHS = 2

# Undocumented pattern seen in closed cases CC-3748, CC-3841, CC-3907, CC-4086, CC-4124:
# four online purchases within about forty minutes, each just under a $500 limit.
SUBLIMIT_THRESHOLD_USD = 500.0
SUBLIMIT_BAND_USD = 60.0
SUBLIMIT_MIN_TXNS = 3
SUBLIMIT_WINDOW_MINUTES = 60

# R6 calls a shared element an origin when "several cards" show it. Three is the smallest
# number that reads as several.
SHARED_ORIGIN_MIN_CARDS = 3
# Interpretation: device profiles this widely used are ordinary software fingerprints
# (for example "Windows | | chrome 66.0 | " reaches 253 customers), not a ring.
SHARED_ORIGIN_MAX_CARDS = 60
# A ring device is new to every account it touches, which is what separates it from a
# browser fingerprint thousands of ordinary customers share.
SHARED_ORIGIN_MIN_NEW_DEVICE_SHARE = 0.8

# How far back an episode can reach. Closed-case episodes span hours to a few days, with
# card testing the long exception (median 175 hours).
EPISODE_LOOKBACK_DAYS = 7
CARD_TESTING_LOOKBACK_DAYS = 14
# The window used when asking how far a device or a ring reaches.
RING_WINDOW_DAYS = 30

# Account takeover: "mixed-channel activity inconsistent with the cardholder", with match
# flags that fail. M-flag values are T/F (or M0/M1/M2 for M4).
ATO_MIN_CHANNELS = 2
MATCH_FLAG_FALSE = "F"
