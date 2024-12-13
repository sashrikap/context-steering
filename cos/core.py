from typing import List
import tqdm
from functools import partial
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from cos.utils import *


def apply_cos(
    logits: torch.tensor, 
    logits_nc: torch.tensor, 
    temperature: float, 
    lambdas: torch.tensor,
    mask: torch.tensor = None,
    return_probs: bool = False,
    last_token: bool = True,
): 
    """Calculate contexual steering influence for the next token
    inf(c, t) = LLM(l1 + c).logprob(x+t) - LLM(l1).logprob(x+t)

    Shape:
        N: batch size
        L: length of sequence
        V: vocab size

    Input:
        logits: (N, L, V)
        logits_nc: logits with no context (N, L, V)
        lambdas: (N,)
        mask: (N, L) if provided
        last_token (bool): whether to return the last token only
    
    Returns:
        probs: (N, V) if `last_token` is True else (N, L, V). If mask is provided, then only
        the masked token gets assigned probability/log probability. Non masked ones get 
        prob=1/logprob=0.
    """
    assert len(logits.shape) == len(logits_nc.shape) == 3
    assert len(lambdas.shape) == 1
    assert len(lambdas) == logits.shape[0]

    if mask is None:
        # same shape as logits, all 0, only the last one is 1
        mask = torch.zeros(logits.shape[:-1], device=logits.device)
        mask[:, -1] = 1

    # TODO: can put temperature later?
    next_lp = F.log_softmax(logits / temperature, dim=-1) # (N, L, V)
    next_lp_nc = F.log_softmax(logits_nc / temperature, dim=-1) # (N, L, V)
    next_lp = next_lp.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0) # (N, L, V)
    next_lp_nc = next_lp_nc.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)
    
    influence = next_lp - next_lp_nc # (N, L, V)
    cos_lp = next_lp + lambdas[:, None, None] * influence # (N, L, V)
    # normalize probabilities
    cos_lp = F.log_softmax(cos_lp, dim=-1)
    cos_lp = cos_lp.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)
    
    output = torch.exp(cos_lp) if return_probs else cos_lp
    return output[:, -1] if last_token else output


def apply_multi_cos(
    all_logits: List[torch.tensor], 
    logits_nc: torch.tensor, 
    temperature: float, 
    all_lambdas: List[torch.tensor],
    mask: torch.tensor = None,
    return_probs: bool = False,
    last_token: bool = True,
): 
    """Calculate contexual steering influence for the next token
    inf(c, t) = LLM(l1 + c).logprob(x+t) - LLM(l1).logprob(x+t)
    """
    for logits, lambdas in zip(all_logits, all_lambdas):
        assert len(logits.shape) == len(logits_nc.shape) == 3
        assert len(lambdas) == logits.shape[0]
        assert len(lambdas.shape) == 1

    if mask is None:
        # same shape as logits, all 0, only the last one is 1
        mask = torch.zeros(logits.shape[:-1], device=logits.device)
        mask[:, -1] = 1

    # TODO: can put temperature later?
    next_lp_nc = F.log_softmax(logits_nc / temperature, dim=-1) # (N, L, V)
    next_lp_nc = next_lp_nc.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)
    all_next_lp = []
    for logits in all_logits:
        next_lp = F.log_softmax(logits / temperature, dim=-1) # (N, L, V)
        next_lp = next_lp.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0) # (N, L, V)
        all_next_lp.append(next_lp)

    cos_lp = next_lp_nc.clone()
    all_influences = []
    for next_lp, lambdas in zip(all_next_lp, all_lambdas):
        influence = next_lp - next_lp_nc # (N, L, V)
        all_influences.append(influence)
        cos_lp += lambdas[:, None, None] * influence # (N, L, V)
        # normalize probabilities

    cos_lp = cos_lp.masked_fill(torch.logical_not(mask.bool())[:, :, None], 0)
    cos_lp = F.log_softmax(cos_lp, dim=-1)
    output = torch.exp(cos_lp) if return_probs else cos_lp
    return output[:, -1] if last_token else output

        
def get_seqs_logits(input_ids, mask, model, tokenizer, cache=None):
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=mask, 
            max_new_tokens=1, 
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
            past_key_values=cache, 
            use_cache=True
        )
    assert len(output.scores) == 1
    logits = output.scores[0] # (batch_size, vocab_size ish)
    assert logits.size(0) == len(input_ids)
    return logits

def contextual_steering_hf(
    model, 
    tokenizer,
    prompts: List[str], 
    contexts: List[str], 
    lambdas: List[float], 
    put_context_first: bool = True, 
    is_chat: bool = True,
    top_p: float = 0.9,
    temperature: float = 0.6,
    show_progress: bool = False,
    max_gen_len: int = None,
    max_batch_size: int = 8,
    max_seq_len: int = 4096,
    verbose: bool = False,
    skip_nan: bool = False,
    prompts_nc: List[str] = None,
) -> dict:
    """Generate response via Contextual Steering forward model p(response | lambda, prompt, context). Output one generation per prompt per lambda.

    Note:
        - supports top_p sampling

    Shape:
        Np: number of prompts, same as the number of contexts
        Nl: number of lambdas

    Args:
        model: Hugging Face model.
        tokenizer: Hugging Face tokenizer.
        prompts(List[str]): list of prompts (Np,)
        contexts(List[str]): list of contexts (Np,)
        prompts_nc(List[str]): list of prompts without context (Np,). If provided, then assume that the context is already included in the prompt. Contexts will be ignored.
            Needs to follow dialog format!
        lambdas(List[float]): List of lambdas (Nl,)
        put_context_first(bool): put context before the prompt
        is_chat(bool): whether the model is a chat model
        max_gen_len(int): max generation length
        top_p(float): top-p sampling
        temperature(float): temperature for sampling
        show_progress(bool): show progress bar
        max_batch_size(int): max batch size for generation
        max_seq_len(int): max sequence length for generation

    Return: 
        [dict], {
            'generation': list of [{'role': 'assistant', 'content': ' Of course! ...'}]
            'tokens': [...],
            'logprobs': [...]
        }

    """
    max_gen_len = max_gen_len or max_seq_len - 1
    mbsz = max_batch_size

    if prompts_nc is None:
        if is_chat:
            prompts, prompts_nc = get_context_pair_dialogs(prompts, contexts, put_context_first)
        else:
            prompts, prompts_nc = get_context_pair_texts(prompts, contexts, put_context_first)
    else:
        assert len(prompts) == len(prompts_nc)
        if is_chat:
            for p, p_nc in zip(prompts, prompts_nc):
                assert_dialog(p)
                assert_dialog(p_nc)

    repeated_prompts = tile_seqs(prompts, len(lambdas))
    repeated_prompts_nc = tile_seqs(prompts_nc, len(lambdas))
    repeated_contexts = tile_seqs(contexts, len(lambdas))
    repeated_lambdas = repeat_seqs(lambdas, len(prompts))
    tokenize_chat = partial(
        tokenizer.apply_chat_template, 
        tokenize=True, 
        return_tensors='pt', 
        padding=True, 
        return_dict=True
    )
    tokenize_text = partial(tokenizer, return_tensors="pt", padding=True)
    get_tokens = tokenize_chat if is_chat else tokenize_text
    repeated_inputs = get_tokens(repeated_prompts).to(model.device)
    repeated_inputs_nc = get_tokens(repeated_prompts_nc).to(model.device)

    batch_size = max_batch_size
    if show_progress:
        pbar_batch = tqdm.tqdm(total=len(repeated_prompts))

    output_tokens, output_logprobs = [], []
    for i in range(0, len(repeated_prompts), mbsz):
        batch_ids = repeated_inputs.input_ids[i: i + mbsz]
        batch_ids_nc = repeated_inputs_nc.input_ids[i: i + mbsz]
        batch_mask = repeated_inputs.attention_mask[i: i + mbsz]
        batch_mask_nc = repeated_inputs_nc.attention_mask[i: i + mbsz]
        batch_lambdas = torch.tensor(repeated_lambdas[i: i + mbsz], device=model.device)
        batch_size = min(mbsz, len(batch_ids))
        cur_len = batch_ids.shape[1]
        total_len = min(max_seq_len, max_gen_len + cur_len)
        remain_len = total_len - cur_len
        if show_progress:
            pbar_prompt = tqdm.tqdm(total=remain_len, leave=False, desc='Generating')

        eos_reached = torch.zeros((batch_size, 1), device=model.device).bool()
        batch_out_tokens, batch_out_logprobs, batch_out_masks = [], [], []
        cache_kv = DynamicCache()
        cache_kv_nc = DynamicCache()
        pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        neg_inf = torch.tensor(-float('inf'), device=model.device)

        for _ in range(cur_len, total_len):
            cur_logits = get_seqs_logits(
                batch_ids, batch_mask, model, tokenizer, cache_kv
            )
            cur_logits_nc = get_seqs_logits(
                batch_ids_nc, batch_mask_nc, model, tokenizer, cache_kv_nc
            )
            cos_probs = apply_cos(
                logits=cur_logits.unsqueeze(1), 
                logits_nc=cur_logits_nc.unsqueeze(1),
                temperature=temperature, 
                lambdas=batch_lambdas,
                return_probs=True,
            ) # (N, V)
            if torch.any(torch.isnan(cos_probs)):
                print(f"Lambdas {batch_lambdas} NaNs in cos_probs")
                if skip_nan:
                    # fill nan with 0
                    cos_probs = torch.where(torch.isnan(cos_probs), torch.ones_like(cos_probs), cos_probs)
                    cos_probs = torch.where(torch.isinf(cos_probs), torch.ones_like(cos_probs), cos_probs)
                    cos_probs = torch.where(cos_probs < 0, torch.ones_like(cos_probs), cos_probs)
            next_token = sample_top_p(cos_probs, top_p) # (N, 1)
            next_logprobs = torch.log(torch.gather(cos_probs, 1, next_token))
            batch_out_tokens.append(next_token)
            batch_out_logprobs.append(next_logprobs)
            batch_out_masks.append(~eos_reached)
            eos_reached |= (next_token == pad_id) | (next_token == eos_id) # </s>
            if show_progress:
                pbar_prompt.update(1)
            if all(eos_reached):
                break

            batch_ids = torch.cat([batch_ids, next_token], axis=-1)
            batch_ids_nc = torch.cat([batch_ids_nc, next_token], axis=-1)
            batch_mask = torch.cat([batch_mask, ~eos_reached], axis=-1)
            batch_mask_nc = torch.cat([batch_mask_nc, ~eos_reached], axis=-1)

        batch_out_masks = torch.cat(batch_out_masks, dim=1)
        batch_out_tokens = torch.cat(batch_out_tokens, dim=1)
        batch_out_logprobs = torch.cat(batch_out_logprobs, dim=1)
        batch_out_tokens = torch.where(batch_out_masks, batch_out_tokens, pad_id)
        batch_out_logprobs = torch.where(batch_out_masks, batch_out_logprobs, neg_inf)

        output_tokens.append(batch_out_tokens)
        output_logprobs.append(batch_out_logprobs)

        if show_progress:
            pbar_prompt.close()
        if show_progress:
            pbar_batch.update(batch_size)
    max_len = max(t.shape[1] for t in output_tokens)
    neg_inf = torch.tensor(-float('inf'), device=model.device)
    output_tokens = [
        F.pad(t, (0, max_len - t.shape[1]), value=pad_id) for t in output_tokens
    ]
    output_logprobs = [
        F.pad(t, (0, max_len - t.shape[1]), value=neg_inf) for t in output_logprobs
    ]
    output_tokens = torch.cat(output_tokens, dim=0)
    output_logprobs = torch.cat(output_logprobs, dim=0)
    outputs = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    if verbose:
        for p, lmbd, out in zip(repeated_prompts, repeated_lambdas, outputs):
            print(f"Prompt: {p}, Lambda: {lmbd}")
            print(f"Generation: {out}")

    if show_progress:
        pbar_batch.close()

    return {
        'generation': [{"role": "assistant", "content": out} for out in outputs] if is_chat else outputs,
        "tokens": output_tokens.cpu(),
        "logprobs": output_logprobs.cpu(),
        'prompts': repeated_prompts,
        'contexts': repeated_contexts,
        'prompts_nc': repeated_prompts_nc,
        'lambdas': repeated_lambdas
    }



def get_cos_logprob_hf(
    model, 
    tokenizer,
    prompts: List[str],
    contexts: List[str],
    responses: List[str],
    contexts_neg: List[str] = None,
    is_chat: bool = True,
    lambdas: List[float] = [-1.0],
    temperature: float = 0.6,
    max_gen_len: int = None,
    max_batch_size: int = 8,
    max_seq_len: int = 512,
    put_context_first: bool = True,
    relative_logit: bool = False,
    verbose: bool = False,
    show_progress: bool = True,
) -> List[float]:
    """Return the log probability of the response given the prompt and context. This can be achieved
    with one forward pass as opposed to generating token by token.

    Note:
    - Assumes interaction happens in 1-round fashion: user -> assistant

    Shape:
        Np: number of prompts, same as the number of contexts
        Nl: number of lambdas

    Args:
        model: Hugging Face model.
        tokenizer: Hugging Face tokenizer.
        prompts (List[str]): List of prompts.
        contexts (List[str]): List of prompts contexts.
        responses (List[str]): List of responses.
        is_chat (bool): Whether the model is a chat model.
        lambdas (list[float]): List of lambdas.
        temperature (float): Temperature for sampling.
    """
    max_gen_len = max_gen_len or max_seq_len - 1
    mbsz = max_batch_size
    tokenize_chat = partial(
        tokenizer.apply_chat_template, 
        tokenize=True, 
        return_tensors='pt', 
        padding=True, 
        return_dict=True
    )
    tokenize_text = partial(tokenizer, return_tensors="pt", padding=True)
    get_tokens = tokenize_chat if is_chat else tokenize_text

    if is_chat:
        if contexts_neg is not None:
            prompts_neg, _ = get_context_pair_dialogs(prompts, contexts_neg, put_context_first)
            prompts, prompts_nc = get_context_pair_dialogs(prompts, contexts, put_context_first)
        else:
            prompts, prompts_nc = get_context_pair_dialogs(prompts, contexts, put_context_first)
            prompts_neg = prompts_nc
    else:
        if contexts_neg is not None:
            prompts_neg, _ = get_context_pair_texts(prompts, contexts_neg, put_context_first)
            prompts, prompts_nc = get_context_pair_texts(prompts, contexts, put_context_first)
        else:
            prompts, prompts_nc = get_context_pair_texts(prompts, contexts, put_context_first)
            prompts_neg = prompts_nc
    repeated_contexts = tile_seqs(contexts, len(lambdas))
    repeated_lambdas = repeat_seqs(lambdas, len(prompts))

    if is_chat:
        prompts_full = [
            p + [{'role': 'assistant', 'content': r}] for p, r in zip(prompts, responses)
        ]
        prompts_full_neg = [
            p + [{'role': 'assistant', 'content': r}] for p, r in zip(prompts_neg, responses)
        ]
    else:
        prompts_full = [f"{p} {r}" for p, r in zip(prompts, responses)]
        prompts_full_neg = [f"{p} {r}" for p, r in zip(prompts_neg, responses)]

    repeated_prompts = tile_seqs(prompts, len(lambdas))
    repeated_toks_prompts = get_tokens(repeated_prompts).to(model.device)
    repeated_toks_prompts_neg = get_tokens(tile_seqs(prompts_neg, len(lambdas))).to(model.device)
    repeated_full = tile_seqs(prompts_full, len(lambdas))
    repeated_full_neg = tile_seqs(prompts_full_neg, len(lambdas))
    repeated_toks_full = get_tokens(repeated_full).to(model.device)
    repeated_toks_full_neg = get_tokens(repeated_full_neg).to(model.device)

    batch_size = max_batch_size
    if show_progress:
        pbar_batch = tqdm.tqdm(total=len(repeated_full))

    output_lps, output_total_lps = [], []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    for i in range(0, len(repeated_full), mbsz):
        batch_prompt_masks = repeated_toks_prompts.attention_mask[i: i + mbsz]

        batch_full_ids = repeated_toks_full.input_ids[i: i + mbsz]
        batch_full_ids_neg = repeated_toks_full_neg.input_ids[i: i + mbsz]
        batch_full_masks = repeated_toks_full.attention_mask[i: i + mbsz]
        batch_full_masks_neg = repeated_toks_full_neg.attention_mask[i: i + mbsz]
        batch_lambdas = torch.tensor(repeated_lambdas[i: i + mbsz], device=model.device)

        # pad full_ids_nc to the same length as full_ids
        dlen_full_ids = batch_full_ids.shape[1] - batch_full_ids_neg.shape[1]
        if dlen_full_ids > 0:
            batch_full_ids_neg = F.pad(batch_full_ids_neg, (dlen_full_ids, 0), value=pad_id)
            batch_full_masks_neg = F.pad(batch_full_masks_neg, (dlen_full_ids, 0), value=0)
            batch_res_len = batch_full_masks.sum(dim=1) - batch_prompt_masks.sum(dim=1)
        else:
            batch_full_ids = F.pad(batch_full_ids, (-dlen_full_ids, 0), value=pad_id)
            batch_full_masks = F.pad(batch_full_masks, (-dlen_full_ids, 0), value=0)
            batch_res_len = batch_full_masks_neg.sum(dim=1) - batch_prompt_masks.sum(dim=1)

        # pad and roll so that response align
        batch_num_pads = batch_res_len.max() - batch_res_len.min()
        batch_full_ids = F.pad(batch_full_ids, (0, batch_num_pads), value=eos_id)
        batch_full_ids_neg = F.pad(batch_full_ids_neg, (0, batch_num_pads), value=eos_id)
        batch_full_masks = F.pad(batch_full_masks, (0, batch_num_pads), value=0)
        batch_full_masks_neg = F.pad(batch_full_masks_neg, (0, batch_num_pads), value=0)
        n_cols = batch_full_ids.size(1)
        # shift to the right for long response
        shifts = batch_res_len - batch_res_len.min()
        rolled_indices = (torch.arange(n_cols, device=model.device).unsqueeze(0) - shifts.unsqueeze(1)) % n_cols

        # aligned ids and masks
        batch_full_ids = batch_full_ids.gather(1, rolled_indices)
        batch_full_ids_neg = batch_full_ids_neg.gather(1, rolled_indices)
        left_mask = torch.arange(n_cols, device=model.device).unsqueeze(0) < shifts.unsqueeze(1)
        batch_full_ids = torch.where(left_mask, pad_id, batch_full_ids)
        batch_full_ids_neg = torch.where(left_mask, pad_id, batch_full_ids_neg)
        batch_full_masks = batch_full_masks.gather(1, rolled_indices)
        batch_full_masks_neg = batch_full_masks_neg.gather(1, rolled_indices)

        len_prompt = batch_full_ids.shape[1] - batch_res_len.max()
        is_prompt = torch.arange(batch_full_ids.shape[1], device=model.device) < len_prompt
        batch_resp_masks = torch.where(is_prompt[None, ], torch.zeros_like(batch_full_masks), batch_full_masks.bool())
        batch_resp_mask_last = torch.roll(batch_resp_masks, shifts=-1, dims=1)

        with torch.no_grad():
            logits = model(batch_full_ids, batch_full_masks, use_cache=False).logits
        logits = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            logits_nc = model(batch_full_ids_neg, batch_full_masks_neg, use_cache=False).logits
        logits_nc = torch.log_softmax(logits_nc, dim=-1)
        cos_probs = apply_cos(
            logits=logits, 
            logits_nc=logits_nc,
            temperature=temperature, 
            lambdas=batch_lambdas,
            mask=batch_resp_mask_last,
            return_probs=True,
            last_token=False
        ) # (N, L, V)
        next_cos_lp = torch.roll(cos_probs, shifts=1, dims=1).log()
        res_lp = torch.gather(next_cos_lp, -1, batch_full_ids[..., None]).squeeze() # (N, L)
        if relative_logit:
            res_lp -= next_cos_lp.max(dim=-1).values
        total_lp = torch.sum(res_lp, dim=-1) # (N)
        output_lps.append(res_lp)
        output_total_lps.append(total_lp)
        if show_progress:
            pbar_batch.update(min(mbsz, len(batch_full_ids)))

    max_len = max(t.shape[1] for t in output_lps)
    # neg_inf = torch.tensor(-float('inf'), device=model.device)
    output_lps = [
        F.pad(t, (0, max_len - t.shape[1]), value=0) for t in output_lps
    ]
    output_lps = torch.cat(output_lps, dim=0)
    output_total_lps = torch.cat(output_total_lps, dim=0)
    return {
        "logprobs": output_lps.cpu(),
        "total_logprobs": output_total_lps.cpu(),
        'prompts': repeated_prompts,
        'contexts': repeated_contexts,
        'prompts_full': repeated_full,
        'lambdas': repeated_lambdas
    }

def multi_contextual_steering_hf(
    model, 
    tokenizer,
    prompts: List[str], 
    all_contexts: List[List[str]], 
    all_lambdas: List[List[float]], 
    put_context_first: bool = True, 
    is_chat: bool = True,
    top_p: float = 0.9,
    temperature: float = 0.6,
    show_progress: bool = False,
    max_gen_len: int = None,
    max_batch_size: int = 8,
    max_seq_len: int = 512,
    verbose: bool = False,
) -> dict:
    """Generate response via Contextual Steering forward model p(response | lambda, prompt, context). Output one generation per prompt per lambda.

    Note:
        - supports top_p sampling

    Shape:
        Np: number of prompts, same as the number of contexts
        Nl: number of lambdas

    Args:
        prompts(List[str]): prompts (Np,)
        all_contexts(List[List[str]]): N different directional contexts, each a list of contexts [(Np,) ] * N
        all_lambdas(List[List[float]]): List of lambdas [(Nl,) ] * N

    Return: 
        [dict], {
            'generation': list of [{'role': 'assistant', 'content': ' Of course! ...'}]
            'tokens': [...],
            'logprobs': [...]
        }

    """
    max_gen_len = max_gen_len or max_seq_len - 1
    mbsz = max_batch_size

    if is_chat:
        prompts, prompts_nc = get_multi_context_pair_dialogs(prompts, all_contexts, put_context_first)
    else:
        prompts, prompts_nc = get_multi_context_pair_texts(prompts, all_contexts, put_context_first)

    repeated_prompts = [tile_seqs(p, len(all_lambdas[0])) for p in prompts]
    repeated_prompts_nc = tile_seqs(prompts_nc, len(all_lambdas[0]))
    repeated_lambdas = [repeat_seqs(l, len(prompts_nc)) for l in all_lambdas]
    tokenize_chat = partial(
        tokenizer.apply_chat_template, 
        tokenize=True, 
        return_tensors='pt', 
        padding=True, 
        return_dict=True
    )
    tokenize_text = partial(tokenizer, return_tensors="pt", padding=True)
    get_tokens = tokenize_chat if is_chat else tokenize_text
    repeated_inputs = [get_tokens(rp).to(model.device) for rp in repeated_prompts]
    repeated_inputs_nc = get_tokens(repeated_prompts_nc).to(model.device)

    batch_size = max_batch_size
    if show_progress:
        pbar_batch = tqdm.tqdm(total=len(repeated_prompts_nc))

    output_tokens, output_logprobs = [], []
    for i in range(0, len(repeated_prompts_nc), mbsz):
        all_batch_ids = [ri.input_ids[i: i + mbsz] for ri in repeated_inputs]
        all_batch_mask = [ri.attention_mask[i: i + mbsz] for ri in repeated_inputs]
        batch_ids_nc = repeated_inputs_nc.input_ids[i: i + mbsz]
        batch_masks_nc = repeated_inputs_nc.attention_mask[i: i + mbsz]
        all_batch_lambdas = [torch.tensor(repeated_lambdas[ci][i: i + mbsz], device=model.device) for ci in range(len(repeated_lambdas))]
        batch_size = min(mbsz, len(batch_ids_nc))
        cur_lens = [bids.shape[1] for bids in all_batch_ids]
        total_len = min(max_seq_len, max_gen_len + max(cur_lens))
        remain_len = total_len - max(cur_lens)
        if show_progress:
            pbar_prompt = tqdm.tqdm(total=remain_len, leave=False, desc='Generating')

        eos_reached = torch.zeros((batch_size, 1), device=model.device).bool()
        batch_out_tokens, batch_out_logprobs, batch_out_masks = [], [], []
        all_cache_kv = [DynamicCache() for _ in range(len(repeated_inputs))]
        cache_kv_nc = DynamicCache()
        pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        neg_inf = torch.tensor(-float('inf'), device=model.device)

        for _ in range(max(cur_lens), total_len):
            all_cur_logits = [
                get_seqs_logits(batch_ids, batch_mask, model, tokenizer, cache_kv)
                for batch_ids, batch_mask, cache_kv in zip(all_batch_ids, all_batch_mask, all_cache_kv)
            ]
            cur_logits_nc = get_seqs_logits(
                batch_ids_nc, batch_masks_nc, model, tokenizer, cache_kv_nc
            )
            cos_logits = apply_multi_cos(
                all_logits=[logits.unsqueeze(1) for logits in all_cur_logits], 
                logits_nc=cur_logits_nc.unsqueeze(1),
                temperature=temperature, 
                all_lambdas=all_batch_lambdas,
                return_probs=False,
            ) # (N, V)
            cos_logits = torch.log_softmax(cos_logits, dim=-1)
            if torch.any(torch.isnan(cos_logits)):
                print(f"Lambdas {all_batch_lambdas[ci]} NaNs in cos_probs")
            cos_probs = torch.softmax(cos_logits, dim=-1)
            next_token = sample_top_p(cos_probs, top_p) # (N, 1)
            next_logprobs = torch.log(torch.gather(cos_probs, 1, next_token))
            batch_out_tokens.append(next_token)
            batch_out_logprobs.append(next_logprobs)
            batch_out_masks.append(~eos_reached)
            eos_reached |= (next_token == pad_id) | (next_token == eos_id) # </s>
            if show_progress:
                pbar_prompt.update(1)
            if all(eos_reached):
                break

            all_batch_ids = [
                torch.cat([batch_ids, next_token], axis=-1)
                for batch_ids in all_batch_ids
            ]
            batch_ids_nc = torch.cat([batch_ids_nc, next_token], axis=-1)
            all_batch_mask = [
                torch.cat([batch_mask, ~eos_reached], axis=-1)
                for batch_mask in all_batch_mask
            ]
            batch_masks_nc = torch.cat([batch_masks_nc, ~eos_reached], axis=-1)

        batch_out_masks = torch.cat(batch_out_masks, dim=1)
        batch_out_tokens = torch.cat(batch_out_tokens, dim=1)
        batch_out_logprobs = torch.cat(batch_out_logprobs, dim=1)
        batch_out_tokens = torch.where(batch_out_masks, batch_out_tokens, pad_id)
        batch_out_logprobs = torch.where(batch_out_masks, batch_out_logprobs, neg_inf)

        output_tokens.append(batch_out_tokens)
        output_logprobs.append(batch_out_logprobs)

        if show_progress:
            pbar_prompt.close()
        if show_progress:
            pbar_batch.update(batch_size)
    max_len = max(t.shape[1] for t in output_tokens)
    neg_inf = torch.tensor(-float('inf'), device=model.device)
    output_tokens = [
        F.pad(t, (0, max_len - t.shape[1]), value=pad_id) for t in output_tokens
    ]
    output_logprobs = [
        F.pad(t, (0, max_len - t.shape[1]), value=neg_inf) for t in output_logprobs
    ]
    output_tokens = torch.cat(output_tokens, dim=0)
    output_logprobs = torch.cat(output_logprobs, dim=0)
    outputs = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    if verbose:
        for p, lmbd, out in zip(repeated_prompts, repeated_lambdas, outputs):
            print(f"Prompt: {p}, Lambda: {lmbd}")
            print(f"Generation: {out}")

    if show_progress:
        pbar_batch.close()

    return {
        'generation': [{"role": "assistant", "content": out} for out in outputs] if is_chat else outputs,
        "tokens": output_tokens.cpu(),
        "logprobs": output_logprobs.cpu(),
        'prompts': repeated_prompts,
        'prompts_nc': repeated_prompts_nc,
        'lambdas': repeated_lambdas
    }
