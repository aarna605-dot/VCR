"""
score.py

Candidate scoring functions for the joint Q->A / QA->R system.

VCR is 4-way multiple choice, not free-text generation, so each candidate
answer (and, subsequently, each candidate rationale) is scored via a
constrained Yes/No prompt rather than asking the model to generate text.
The softmax probability assigned to the "Yes" token is the candidate's
score; argmax over the four candidates gives the prediction.

The rationale scorer is conditioned on the model's own *predicted* answer
(not the gold answer) -- this is what makes the pipeline a genuine joint
Q->AR system rather than two independent classifiers.
"""

import torch


def _score_yes_no(model, processor, image, prompt: str) -> float:
    """Run one forward pass and return softmax P(Yes) vs P(No) for the next token."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits[0, -1]
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    yn_logits = torch.tensor([logits[yes_id], logits[no_id]])
    return torch.softmax(yn_logits, dim=0)[0].item()


def score_candidates(model, processor, image, question: str, candidates: list[str]):
    """Score each candidate answer; return (argmax index, all scores)."""
    scores = []
    for cand in candidates:
        prompt = (
            f"Question: {question}\n"
            f"Proposed answer: {cand}\n"
            "Is this proposed answer correct? Answer strictly Yes or No."
        )
        scores.append(_score_yes_no(model, processor, image, prompt))
    return scores.index(max(scores)), scores


def score_rationale(model, processor, image, question: str, predicted_answer: str, rationale_candidates: list[str]):
    """Score each candidate rationale, conditioned on the *predicted* answer; return (argmax index, all scores)."""
    scores = []
    for rat in rationale_candidates:
        prompt = (
            f"Question: {question}\n"
            f"Answer: {predicted_answer}\n"
            f"Proposed rationale: {rat}\n"
            "Does this rationale correctly explain the answer? Answer strictly Yes or No."
        )
        scores.append(_score_yes_no(model, processor, image, prompt))
    return scores.index(max(scores)), scores
