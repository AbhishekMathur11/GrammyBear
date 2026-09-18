# Game Judge Evaluation Report

## Overall accuracy
- Correct predictions: 66 / 100
- Accuracy: 66.0%

## Macro F1 score
| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| correct | 0.880 | 0.489 | 0.629 | 45 |
| partially_correct | 0.609 | 0.700 | 0.651 | 20 |
| incorrect | 0.476 | 0.800 | 0.597 | 25 |
| no_response | 1.000 | 1.000 | 1.000 | 10 |

**Macro F1: 0.719**

## Confusion matrix (rows = expected, columns = predicted)
| Expected \ Predicted | correct | partially_correct | incorrect | no_response |
|---|---|---|---|---|
| correct | 22 | 6 | 17 | 0 |
| partially_correct | 1 | 14 | 5 | 0 |
| incorrect | 2 | 3 | 20 | 0 |
| no_response | 0 | 0 | 0 | 10 |

## Accuracy by game
- complete_the_sentence: 44.0%
- story_challenge: 88.0%

## Error analysis
34 of 100 examples were misclassified.
- 17x expected `correct`, predicted `incorrect`
- 6x expected `correct`, predicted `partially_correct`
- 5x expected `partially_correct`, predicted `incorrect`
- 3x expected `incorrect`, predicted `partially_correct`
- 2x expected `incorrect`, predicted `correct`

Sample misclassifications:
- [eval_001] "under the table" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_002] "under" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_003] "in" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_004] "the ball rolled into the box" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_005] "on the wall" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_006] "between the chairs" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_007] "an elephant" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
- [eval_008] "a red umbrella" — expected `correct`, got `incorrect` (Judge call failed: Request timed out.)
