# TranPy

<!-- intro-start -->
Machine learning for power-system transient stability.
<!-- intro-end -->

## Installation

```bash
pip install tranpy-ai
```

Optional extras: `tranpy-ai[explain]`, `tranpy-ai[neural]`, `tranpy-ai[boosting]`, `tranpy-ai[plot]`, or `tranpy-ai[all]`.

Requires Python 3.11 or newer.

## Get started

```python
import tranpy
from tranpy_core import evaluate, make_pipeline, train_test_split

data = tranpy.load_new_england()
split = train_test_split(data.labels, test_size=0.2, seed=42)

model = make_pipeline("random_forest", seed=0)
model.fit(data.features[split.train], data.labels[split.train])

scores = evaluate(model, data.features[split.test], data.labels[split.test])
print(scores.accuracy, scores.roc_auc)
```

## Example: early prediction

How soon after a fault can stability be predicted?

```python
import tranpy
from tranpy.early_prediction import run_early_prediction
from tranpy_core.models import make_pipeline

events = tranpy.open_event_shards("ieee_9bus")

results = run_early_prediction(
    events,
    {"random forest": make_pipeline("random_forest", seed=0)},
    reference="fault_start",
    windows_s=(0.05, 0.1, 0.2),
)
print(results[["window_s", "model", "accuracy_mean"]])
```

## Learn more

- [Examples](examples/)
- [Architecture](docs/architecture.md)
- [Data](docs/data.md)
- [Validation](docs/validation.md)

## License

MIT
