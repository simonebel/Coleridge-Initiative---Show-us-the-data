import re
import math

import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import f1_score
import torch
from torch.nn.utils import clip_grad_norm_


def train(
    model,
    device,
    accumulation_steps,
    train_dataset,
    train_data_loader,
    loss_fn,
    optimizer,
    clip,
    batch_size,
    scheduler=None,
):
    epoch_loss, nb_tr_steps = 0, 0
    model.train()
    tot_iteration = math.ceil(train_dataset.__len__() / batch_size)
    model.zero_grad()
    for step, (input_ids, attention_mask, ans_tok_pos, answerable) in enumerate(
        train_data_loader
    ):
        input_ids, attention_mask, ans_tok_pos, answerable = (
            input_ids.to(device),
            attention_mask.to(device),
            ans_tok_pos.to(device),
            answerable.to(device),
        )

        is_answerable = answerable.bool()
        start_logits, end_logits, answerable_logits = model(input_ids, attention_mask)

        start_loss = loss_fn(start_logits.squeeze(-1), ans_tok_pos[:, 0])
        end_loss = loss_fn(end_logits.squeeze(-1), ans_tok_pos[:, 1])
        answerable_loss = loss_fn(answerable_logits, answerable)

        loss = (start_loss + end_loss + answerable_loss) / 3
        loss = loss / accumulation_steps
        loss.backward()

        if ((step + 1) % accumulation_steps == 0) or ((step + 1) == tot_iteration):
            clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            model.zero_grad()

        epoch_loss += loss.item() * accumulation_steps
        nb_tr_steps += 1

    return epoch_loss / nb_tr_steps


def find_best_indices(beam_size, start_logits, end_logits):

    batch_size, seq_length, _ = start_logits.size()

    span_score = start_logits + end_logits.transpose(1, 2)
    span_score = span_score.triu()
    span_score = span_score.reshape(batch_size, 1, seq_length * seq_length)

    best_prob_span, best_indc = span_score.topk(beam_size, 2, largest=True)

    best_indc = best_indc.view(batch_size, beam_size, 1)
    candidates_spans = torch.cat(
        (best_indc // seq_length, best_indc % seq_length), dim=2
    )

    return candidates_spans


def decode_candidates_answer(beam_size, start_logits, end_logits, input_ids, tokenizer):
    batch_size, seq_length, _ = start_logits.size()
    candidates_spans = find_best_indices(beam_size, start_logits, end_logits)
    pred_answers = []
    for i in range(batch_size):
        batch_candidates = candidates_spans[i]
        pred_batch = []
        for n in range(beam_size):
            pred_span = batch_candidates[n]

            if (pred_span[0].item() == 0) and (pred_span[1].item() == 0):
                pred_batch.append("")
            else:

                candidate_ids = input_ids[i][
                    pred_span[0].item() : pred_span[1].item() + 1
                ]
                pred_ans = tokenizer.decode(candidate_ids)
                pred_batch.append(pred_ans)

        pred_answers.append(pred_batch)

    return pred_answers


def clean_text(txt):
    return re.sub("[^A-Za-z0-9]+", " ", str(txt).lower())


def compute_jaccard(str1, str2):
    str1 = clean_text(str1)
    str2 = clean_text(str2)
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_F_05_Score(TP, FP, FN):
    """
    Return the F(0,5) Score given True Postive, False Positive and False Negative numbers
    """

    TP *= 1 + (0.5 ** 2)
    FN *= (0.5) ** 2

    if TP + FN + FP == 0:
        F_05_Score = 0

    else:
        F_05_Score = TP / (TP + FN + FP)

    return F_05_Score


def F_beta_score(
    beam_size, start_logits, end_logits, input_ids, is_answerable, answer, tokenizer
):

    start_logits = start_logits[is_answerable]
    end_logits = end_logits[is_answerable]
    input_ids = input_ids[is_answerable]
    answer = np.array(answer)[is_answerable.cpu()]

    pred_answers = decode_candidates_answer(
        beam_size, start_logits, end_logits, input_ids, tokenizer
    )

    TP, FP, FN = 0, 0, 0
    for bean_ans, truth_ans in zip(pred_answers, answer):
        for pred_ans in bean_ans:
            if len(pred_ans) >= 1:
                jaccard_score = compute_jaccard(pred_ans, truth_ans)

                if jaccard_score >= 0.5:
                    TP += 1
                else:
                    FP += 1
            else:
                FN += 1

    F_05_Score = calculate_F_05_Score(TP, FP, FN)

    return F_05_Score


def evaluate(model, device, tokenizer, dev_data_loader, loss_fn, beam_size, batch_size):

    epoch_loss, epoch_f_beta, epoch_f1, nb_dev_steps = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for (
            input_ids,
            attention_mask,
            ans_tok_pos,
            answerable,
            answer,
        ) in dev_data_loader:
            input_ids, attention_mask, ans_tok_pos, answerable = (
                input_ids.to(device),
                attention_mask.to(device),
                ans_tok_pos.to(device),
                answerable.to(device),
            )

            is_answerable = answerable.bool()

            start_logits, end_logits, answerable_logits = model(
                input_ids, attention_mask
            )

            start_loss = loss_fn(start_logits.squeeze(-1), ans_tok_pos[:, 0])
            end_loss = loss_fn(end_logits.squeeze(-1), ans_tok_pos[:, 1])
            answerable_loss = loss_fn(answerable_logits, answerable)

            loss = (start_loss + end_loss + answerable_loss) / 3

            F_05_Score = F_beta_score(
                beam_size,
                start_logits,
                end_logits,
                input_ids,
                is_answerable,
                answer,
                tokenizer,
            )
            F1_classifier = f1_score(
                answerable.detach().cpu().numpy(), is_answerable.detach().cpu().numpy()
            )

            epoch_loss += loss.item()
            epoch_f_beta += F_05_Score
            epoch_f1 += F1_classifier
            nb_dev_steps += 1

    return (
        epoch_loss / nb_dev_steps,
        epoch_f_beta / nb_dev_steps,
        epoch_f1 / nb_dev_steps,
    )


def test(tokenizer, beam_size, model, device, test_data_loader, test_dataset):
    pred_by_doc = {k: {"pred": [], "truth": ""} for k in np.unique(test_dataset.doc_id)}
    model.eval()
    with torch.no_grad():
        for (
            input_ids,
            attention_mask,
            ans_tok_pos,
            answerable,
            answer,
            doc_ids,
        ) in test_data_loader:
            input_ids, attention_mask, ans_tok_pos, answerable, doc_ids = (
                input_ids.to(device),
                attention_mask.to(device),
                ans_tok_pos.to(device),
                answerable.to(device),
                np.array(doc_ids),
            )

            start_logits, end_logits, answerable_logits = model.evaluate(
                input_ids, attention_mask
            )

            answerable_pred = answerable_logits.argmax(1)
            answerable_bool = answerable_pred.bool()

            print(answerable_bool)
            for ans, doc_id in zip(answer, doc_ids):
                if pred_by_doc[doc_id]["truth"] == "":
                    pred_by_doc[doc_id]["truth"] = ans

            start_logits = start_logits[answerable_bool]
            end_logits = end_logits[answerable_bool]
            input_ids = input_ids[answerable_bool]
            doc_ids = doc_ids[answerable_bool.cpu().numpy()]

            pred_answers = decode_candidates_answer(
                beam_size, start_logits, end_logits, input_ids, tokenizer
            )
            if len(pred_answers) >= 1:
                for pred, doc_id in zip(pred_answers, doc_ids):
                    pred_by_doc[doc_id]["pred"].extend(pred)

    TP, FP, FN = 0, 0, 0
    for key in pred_by_doc.keys():
        pred_answers = pred_by_doc[key]["pred"]
        truth_ans = pred_by_doc[key]["truth"]

        if len(pred_answers) >= 1:
            for pred in pred_answers:
                jaccard_score = compute_jaccard(pred, truth_ans)
                if jaccard_score >= 0.5:
                    TP += 1
                else:
                    FP += 1
        else:
            FN += 1

    F_05_Score = calculate_F_05_Score(TP, FP, FN)

    return F_05_Score, pred_by_doc


def plot_curves(n_epochs, train_loss_set, dev_loss_set, dev_f_beta_set, dev_f1_set):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[i + 1 for i in range(len(train_loss_set))],
            y=train_loss_set,
            mode="lines+markers",
            name="Train Loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[i + 1 for i in range(len(dev_loss_set))],
            y=dev_loss_set,
            mode="lines+markers",
            name="Validation Loss",
        )
    )
    fig.update_layout(
        xaxis_title="Epochs",
        yaxis_title="Loss Function",
        title=f"Evolution of the Loss Function during {n_epochs} epochs",
        title_x=0.5,
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=[i + 1 for i in range(len(dev_f_beta_set))],
            y=dev_f_beta_set,
            mode="lines+markers",
            name="Developpement F_beta Score",
        )
    )
    fig2.update_layout(
        xaxis_title="Epochs",
        yaxis_title="F_beta Score",
        title=f"Evolution of the F_beta Score during {n_epochs} epochs",
        title_x=0.5,
    )

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=[i + 1 for i in range(len(dev_f1_set))],
            y=dev_f1_set,
            mode="lines+markers",
            name="Developpement F1 Score",
        )
    )
    fig3.update_layout(
        xaxis_title="Epochs",
        yaxis_title="F1 Score",
        title=f"Evolution of the F1 Score (classification objectif) during {n_epochs} epochs",
        title_x=0.5,
    )
    fig.show()
    fig2.show()
    fig3.show()
