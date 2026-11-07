# MSCS 634 – Lab 2: KNN vs. Radius Neighbors on Wine Data

**Student:** Rajasekhar Reddy Kayyam  
**Course:** MSCS 634 – Advanced Big Data & Data Mining

## Purpose
Compare K-Nearest Neighbors (KNN) and Radius Neighbors classifiers on the scikit-learn Wine Dataset (178 samples, 13 features, three classes). Explore parameter sensitivity and visualize test accuracy.

## Method
- Stratified 80% training / 20% testing split (`random_state=42`; 142 train, 36 test).
- Euclidean distance on **unscaled** features to preserve the assignment's radius values.
- KNN: `k = 1, 5, 11, 15, 21`.
- Radius Neighbors: `radius = 350, 400, 450, 500, 550, 600`.
- Accuracy measured on the same held-out test set.
- Radius Neighbors uses `outlier_label="most_frequent"` for possible zero-neighbor cases; none occurred on this test set.

## Results

### KNN
| k | Accuracy |
|---:|---:|
| 1 | 77.78% |
| 5 | 80.56% |
| 11 | 80.56% |
| 15 | 80.56% |
| 21 | 80.56% |

### Radius Neighbors
| Radius | Accuracy |
|---:|---:|
| 350 | 72.22% |
| 400 | 69.44% |
| 450 | 69.44% |
| 500 | 69.44% |
| 550 | 66.67% |
| 600 | 66.67% |

## Key insights
- KNN performs better than Radius Neighbors at the tested settings; best observed accuracy is 80.56% versus 72.22% (8.33 percentage-point difference).
- KNN accuracy plateaus at k=5 through k=21 for this split.
- Radius Neighbors accuracy generally decreases as the radius grows, possibly because more distant examples from other classes are included.
- KNN guarantees a fixed neighbor count; Radius Neighbors adapts its neighbor count to local density, which is helpful if a meaningful distance threshold exists.

## Challenges and decisions
The Wine features have different numeric scales. The required radii of 350–600 are most interpretable on the original feature values, so scaling was intentionally omitted. This means high-magnitude features can dominate Euclidean distance. A follow-up study should standardize features and tune radii using cross-validation. The `outlier_label` parameter handles potential samples with no neighbors.

## Files
- `MSCS_634_Lab_2.ipynb` – executable notebook with exploration, experiments, charts, and discussion.
- `knn_accuracy.png`, `rnn_accuracy.png` – accuracy plots.
- `requirements.txt` – Python dependencies.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook MSCS_634_Lab_2.ipynb
```
Run all cells from top to bottom. The notebook also saves the two plots as PNG files.

## Submission
Create a public GitHub repository named `MSCS_634_Lab_2`, upload these files, and submit its URL in Blackboard. The repository must be created in your own GitHub account.
