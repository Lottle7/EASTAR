# LLM Prior Prompt (Stage 2 task-channel routing)

This file is the **input template** given to a text-only LLM to produce the
task-channel prior used to initialize Stage 2's task-adaptive router
(`EASTAR_route.py`).

The provided prior file in `task_channel_priors/crime_prior.json` was produced
by this template with the parameters listed at the bottom.

---

## System prompt

```
You are an urban AI assistant for task-channel prior generation.
Your job is to assign importance weights to predefined representation channels for ONE task.
Output must be valid JSON only. No markdown, no extra text.
Weights must be non-negative and sum to 1.
```

## User prompt template

```
Task: {task}
Task description: {task_description}

We have C={C} representation channels:
{channel_block}

Please infer task-channel prior weights using urban common-sense reasoning.
Rules:
1) Return exactly {C} weights, in channel_index order.
2) All weights must be >= 0.
3) Sum(weights) must equal 1.
4) Free channels (Free-*) are usually lower but can be non-zero.
5) Give one concise rationale sentence per channel.

Return JSON only with this schema:
{{
  "task": "{task}",
  "weights": [w1, w2, ..., w{C}],
  "rationale": {{
    "1:{ch_1_name}": "...",
    "2:{ch_2_name}": "..."
  }}
}}
```

## `channel_block` (C=11)

This is the static channel catalog used in this release (see
`channel_schema.json` for the machine-readable form).

```
1. Tech: Technology, startup, software, digital product, and professional innovation activities.
2. Social: General socializing, dating, entertainment, game nights, and hobby-centered gatherings with strong social intent.
3. Arts: Arts, music, film, design, photography, writing, crafts, and cultural creation or appreciation.
4. Health: Fitness, sports, outdoor activities, wellbeing, and support-oriented health-related participation.
5. Education: Learning, study, reading, workshops, and intellectually oriented knowledge exchange.
6. Community: Neighborhood, civic, political, family, animal, and community-support activities.
7. Food: Food-, drink-, dining-, tasting-, and culinary-centered meetup activities.
8. Diversity: Identity, ethnicity, faith, spirituality, alternative lifestyle, and belief-centered communities.
9. Free-1: Auxiliary interaction pattern not fully covered by anchored semantic channels.
10. Free-2: Auxiliary interaction pattern not fully covered by anchored semantic channels.
11. Global-Structure: Consensus structural context learned from mobility + POI + land-use global fusion.
```

## `task_descriptions["crime"]` (filled into `{task_description}`)

```
Predict relative crime incidence intensity across urban regions.
Focus on social exposure, nighttime dynamics, disorder risk, and guardianship-related structure.
```

## Generation parameters used in this release

| Parameter | Value |
|---|---|
| Model | `gpt-5.4` (any chat-completion model with JSON mode can be substituted) |
| Temperature | `0.1` |
| Response format | `json_object` |
| K (anchored channels) | `8` |
| M (free channels) | `2` |
| Total channels C | `K + M + 1 = 11` |

## Output schema (what the LLM must return)

```json
{
  "task": "crime",
  "weights": [w_1, w_2, ..., w_11],
  "rationale": {
    "1:Tech": "...",
    "2:Social": "...",
    "3:Arts": "...",
    "4:Health": "...",
    "5:Education": "...",
    "6:Community": "...",
    "7:Food": "...",
    "8:Diversity": "...",
    "9:Free-1": "...",
    "10:Free-2": "...",
    "11:Global-Structure": "..."
  }
}
```

`weights` are then L1-normalized (clipped to `>= 1e-6`) and converted to
log-initialization via `log(w + eps)` inside `EASTAR_route.py`.
