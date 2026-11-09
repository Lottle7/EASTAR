# MLLM Prompt Embedding Template (Stage 2 text input)

This file is the **input prompt** given to a multimodal LLM to produce the
per-region text embeddings (`prompt_embs_llama_crime.npy`) that Stage 2's
task-adaptive router conditions on.

The prompt is **task-specific**; only the `crime` variant is included in this
release.

---

## Model

| Field | Value |
|---|---|
| Model | `Llama-3.2-11B-Vision-Instruct` (Meta) |
| Loader | `modelscope.MllamaForConditionalGeneration` (or `transformers`) |
| Precision | `fp16` |
| Hidden size | `4096` |

## Per-region multimodal input

Each region is presented to the MLLM as a single chat message containing:

- **1 satellite image** of the region's 100 m tile.
- **N street-view images** sampled within the region (variable per region).
- **Text** = latitude, longitude, human-readable address, and POI category list.

All images are resized to `640 × 640` before being fed to the processor.

## Crime task prompt

The full text prompt concatenated after the chat template is:

```
You are a top expert in urban criminology and urban planning. Your task is to analyze various data of a city area and deeply understand the potential characteristics related to crime risk in that area.
[Input Information]
Given are three types of information:
1. Satellite images: Please focus on the macroscopic layout of the area, such as: land use types (residential areas, commercial areas, industrial areas, open spaces), building density, road network structure (grid-like or disordered), connectivity with surrounding areas (such as central commercial areas, transportation hubs), green coverage rate, etc.
2. Street view images: Please focus on the micro environment of the area, such as: the condition of buildings (such as graffiti on walls, broken windows), the condition of public facilities (such as whether the streetlights are intact, whether there are surveillance cameras), the cleanliness of the streets, accessibility of sidewalks, signs of commercial activities (such as busy shops or closed and bankrupt ones), etc.
3. Text information: Contains the latitude and longitude of the area, the address, and POI category information.
[Your Core Task]
Please conduct a step-by-step reasoning to analyze which features and patterns in the multi-modal information are highly correlated with potential urban crime risk. Combine all the information to conduct a comprehensive 'region diagnosis'.
[Please think and output in the following structure]
1. Physical environment analysis： Based on satellite and street view images， list all the physical features that may indicate an increase or decrease in crime risk (such as： blind spots without natural surveillance， abandoned buildings， poorly maintained public spaces).
2. Social and economic background analysis： Based on text data and the inferred information in the images (such as the level of regional prosperity)， analyze the possible social and economic factors that may affect the crime rate (such as： signs of economic recession， population mobility， community cohesion).
3. Risk comprehensive assessment： Combine all these points to make a preliminary judgment of the overall crime risk level (high， medium， low)， and explain your core reasons. Focus on how the different modalities of information mutually confirm or provide new perspectives.
4. Conclusion and feature extraction： Finally， please summarize a set of key feature labels that best represent the crime prediction risk of the area (such as： #High-density old residences #Insufficient commercial vitality #Insufficient lighting #Low-income population concentration #Near transportation hubs).
Please focus your thinking on the structured analysis above. Your final output should be this detailed analysis report.
```

## Embedding extraction

```python
outputs     = model(input_ids=..., attention_mask=..., images=..., output_hidden_states=True)
last_hidden = outputs.hidden_states[-1]            # [1, seq_len, 4096]
emb         = last_hidden.mean(dim=1).cpu().numpy()  # [1, 4096]  →  float16 on save
```

Per region → one 4096-dim vector. After running on all `N` regions:

```
prompt_embs_llama_crime.npy  shape = (N, 4096), dtype = float16
```

## How it is consumed

`EASTAR_route.py` loads `prompt_embs_llama_<task>.npy` for the requested
`(city, task)` pair and feeds each region's 4096-dim vector through a
bottleneck MLP (`text_bottleneck` in `TaskAdaptiveRouter`) before cross-attending
it against the fused structural embedding. See
`EASTAR_route.py` loads the embeddings for `TaskAdaptiveRouter.forward`.

## Reproducibility notes

- The chat template is `processor.apply_chat_template(..., add_generation_prompt=True)`.
- `output_hidden_states=True` (we do not decode; only the hidden states are used).
- The `crime` prior weights in `task_channel_priors/crime_prior.json` were
  produced independently by a text-only LLM (see `llm_prior_prompt.md`); the
  two pipelines (prior + MLLM embedding) are decoupled.
