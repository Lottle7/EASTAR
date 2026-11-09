# EASTAR

**EASTAR: Event-Anchored Semantic and Task-Adaptive Region Representation Learning** (ICDE 2027).

## Installation

    python -m pip install -r requirements.txt

## Data

`data_NY/` and `data_Chi/` provide preprocessed mobility, POI, and land-use
inputs, labels for crime, check-in, service call, and population prediction,
and precomputed task prompt embeddings. Channel priors are in
`task_channel_priors/`.

### Meetup event data

Obtain event records separately through the
[official Meetup GraphQL API](https://www.meetup.com/graphql/). Before running
Stage 1, prepare the following arrays in each city's `semantic/` folder. Use
the same region order as the city inputs and the eight event channels defined
in `prompts/channel_schema.json`.

| File suffix | Shape | Contents |
| --- | --- | --- |
| `_k8_semantic_profiles.npy` | `(N, 8, 384)` | Region-by-channel event embeddings |
| `_k8_semantic_distribution.npy` | `(N, 8)` | Region-by-channel event proportions |
| `_k8_semantic_mask.npy` | `(N, 8)` | Availability of each event channel |

Use the filename prefix `Manh` for New York (`N=180`) and `CHI` for Chicago
(`N=77`).

## Quick Start

After preparing the event arrays, run from the repository root. This example
uses New York crime prediction.

    python EASTAR_train.py --city NY
    python EASTAR_route.py --city NY --task crime

Stage 1 learns one set of semantic channels per city without task labels.
Stage 2 adapts those channels to the task using ten-fold cross-validation.
