# Replay evaluation

120 closed cases replayed from TigerGraph, seed 7. The case under test is hidden from every query, so the agent cannot retrieve its own answer.

| Measure | Result |
| --- | --- |
| Verdict accuracy | 85.0% |
| Confirmed fraud called fraud | 80.0% |
| Cleared alerts cleared | 90.0% |
| Verdict left uncertain | 0.8% |
| Pattern named correctly | 56.2% |
| Suspicious transactions found | 47.4% |
| Suspicious transactions precision | 52.4% |
| Exposure matched | 38.3% |
| Action agreement (F1) | 58.5% |
| Report decision agreement | 87.5% |
| Reports the bank filed, also filed | 50.0% |
| Reports filed the bank did not | 10.5% |

## Pattern confusion

| The bank said | The agent said | Cases |
| --- | --- | --- |
| card_not_present_new_device | card_not_present_new_device | 13 |
| account_takeover | none | 9 |
| card_not_present_fraud | card_not_present_fraud | 9 |
| out_of_region_use | none | 4 |
| account_takeover | account_takeover | 3 |
| out_of_region_use | out_of_region_use | 2 |
| account_takeover | card_not_present_new_device | 2 |
| card_not_present_new_device | card_not_present_fraud | 2 |
| account_takeover | card_not_present_fraud | 1 |
| account_takeover | out_of_region_use | 1 |
| card_not_present_fraud | card_not_present_new_device | 1 |
| card_not_present_new_device | undocumented | 1 |
