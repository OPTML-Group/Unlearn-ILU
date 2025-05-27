import evaluate
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from tqdm import tqdm


def run_generation(batch, model, tokenizer, max_length=512, max_new_tokens=512):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = "### Answer: "
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]

    # now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = "left"
    left_pad_tokenizer.padding_size = "longest"
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to(model.device)
    # now generate
    out = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=left_pad_tokenizer.eos_token_id,
    )
    strs = left_pad_tokenizer.batch_decode(
        out[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )
    return input_strings, strs, ground_truth


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss


def eval_bleu(gen_outputs, ground_truths):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)

    eval_result = {
        "rouge": rouge_res,
        "bleu": bleu_res,
    }
    return eval_result


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    for gen, gt in zip(gen_outputs, ground_truths):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores["rouge1"].recall)
        rougeL_recall.append(rouge_scores["rougeL"].recall)

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def get_all_evals(
    model, tokenizer, eval_dataloader, max_length=512, max_new_tokens=512
):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    for batch in tqdm(eval_dataloader):
        # send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(
                batch,
                model,
                tokenizer=tokenizer,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
            )
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)

        gt_loss = get_batch_loss(outputs.logits, batch["labels"])
        num_token_gt = (batch["labels"] != -100).sum(-1)

        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs["avg_gt_loss"] = (
            eval_logs.get("avg_gt_loss", [])
            + (gt_loss / num_token_gt).cpu().numpy().tolist()
        )
        eval_logs["gt_loss"] = eval_logs.get("gt_loss", []) + gt_loss.tolist()
        eval_logs["num_token_gt"] = (
            eval_logs.get("num_token_gt", []) + num_token_gt.tolist()
        )

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths))

    eval_logs["generated_text"] = list(zip(input_strings, gen_outputs, ground_truths))
    return eval_logs
