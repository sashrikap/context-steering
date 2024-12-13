"""
Tools for evaluation using cos
"""

import itertools
import torch
import torch.nn.functional as F
from cos.core import apply_cos
from cos.utils import SupportedModel, HF_MODEL_CLASS
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM



def get_highest_lp_indices_causal_lm(model,
                              tokenizer,
                              context_sequences: List[str], 
                              no_context_sequences: List[str],
                              options: List[str], 
                              lambdas: List[float],
                              temperature: float = 0.7) -> dict:  
    """
    Returns index [0, n) of most probable option in `options` per sequence in `sequences` for `model`
    of type AutoModelForCausalLM from HuggingFace API.

    Uses the GENERATE call of the model.
    """   
    # calculate logprobs of next token being generated
    def get_seqs_logits(seqs):
        input_ids = tokenizer(seqs, return_tensors="pt", padding=True).input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            ).scores
        assert len(output) == 1 # check shape before indexing into scores
        # scores are pre softmax
        logits = output[0] # shape: (batch_size, vocab_size ish)
        assert logits.size(0) == len(seqs)
        return logits 
    
    logits_no_context = get_seqs_logits(no_context_sequences)
    if context_sequences:
        logits_with_context = get_seqs_logits(context_sequences)

        # unsqueeze logits sizes to account for seqlen dimension
        logprobs = apply_cos(
            logits=logits_with_context.unsqueeze(1), 
            logits_no_context=logits_no_context.unsqueeze(1), 
            temperature=temperature, 
            lambdas=lambdas,
            return_probs=False
        )
    else:
        logprobs = F.log_softmax(logits_no_context, dim=-1) # log softmax over vocab size dimension

    # encode the options
    tokenized_options = tokenizer(options, return_tensors="pt").input_ids
    if tokenized_options.size(1) > 1:
        bos_col = tokenized_options[:, 0]
        assert torch.all(bos_col == bos_col[0]) # check that the first column is all bos tokens by confirming all values are the same
        # extract the middle column (ignoring bos, eos tokens)
        tokenized_options = tokenized_options[:, 1].to(model.device)
    else:
        tokenized_options = tokenized_options.squeeze(1).to(model.device)

    assert tokenized_options.size() == (len(options),)

    # influence function outputs lps corresponding to indices (l0, p0), (l0, p1), (l1, p0), ...
    lambdas_and_prompts = list(itertools.product(lambdas, context_sequences))

    gathered_lps = torch.gather(logprobs, dim=1, index=tokenized_options.repeat(len(lambdas_and_prompts), 1))
    assert gathered_lps.size() == (len(lambdas_and_prompts), len(options),), f"{gathered_lps.size()}"
    
    highest_lp_indices = torch.argmax(gathered_lps, dim=1).tolist()

    # batch size is the number of sequences
    assert len(highest_lp_indices) == len(lambdas_and_prompts)

    return {k: v for k, v in zip(lambdas_and_prompts, highest_lp_indices)}

def get_highest_lp_indices_seq2seq(model,
                                tokenizer,
                                context_sequences: List[str], 
                                no_context_sequences: List[str], 
                                options: List[str], 
                                lambdas: List[float],
                                temperature: float = 0.7) -> List[int]:  
    """
    Returns index [0, n) of most probable option in `options` per sequence in `sequences` for `model`
    of type AutoModelForSeq2SeqLM from HuggingFace API.

    Uses the GENERATE call of the model.
    """   
    # calculate logprobs of next token being generated
    def get_seqs_logits(seqs):
        input_ids = tokenizer(seqs, return_tensors="pt", padding=True).input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            ).scores
        assert len(output) == 1 # check shape before indexing into scores
        # scores are pre softmax
        logits = output[0] # shape: (batch_size, vocab_size ish)
        assert logits.size(0) == len(seqs)
        return logits 
    
    # encode the options, extracting the middle column (ignoring bos, eos tokens)
    # tokenized_options = tokenizer(options, return_tensors="pt").input_ids[:, 1].to(model.device)
    tokenized_options = tokenizer(options, return_tensors="pt").input_ids.squeeze(1).to(model.device)

    # encode the options
    tokenized_options = tokenizer(options, return_tensors="pt").input_ids
    if tokenized_options.size(1) > 1:
        bos_col = tokenized_options[:, 0]
        assert torch.all(bos_col == bos_col[0]) # check that the first column is all bos tokens by confirming all values are the same
        # extract the middle column (ignoring bos, eos tokens)
        tokenized_options = tokenized_options[:, 1].to(model.device)
    else:
        tokenized_options = tokenized_options.squeeze(1).to(model.device)

    assert tokenized_options.size() == (len(options),)
    
    # calculate logprobs (based on whether extra context is provided with the prompt)
    logits_no_context = get_seqs_logits(no_context_sequences)
    logits_with_context = get_seqs_logits(context_sequences)

    # unsqueeze logits sizes to account for seqlen dimension
    logprobs = apply_cos(
        logits=logits_with_context.unsqueeze(1), 
        logits_no_context=logits_no_context.unsqueeze(1), 
        temperature=temperature, 
        lambdas=lambdas, 
        return_probs=False
    )
    
    lambdas_and_prompts = list(itertools.product(lambdas, context_sequences))

    # gather logprobs of the tokenized labels (i.e. options)
    gathered_lps = torch.gather(logprobs, dim=1, index=tokenized_options.repeat(len(lambdas_and_prompts), 1))
    assert gathered_lps.size() == (len(lambdas_and_prompts), len(options),), f"{gathered_lps.size()}"
    
    highest_lp_indices = torch.argmax(gathered_lps, dim=1).tolist()

    # batch size is the number of sequences
    assert len(highest_lp_indices) == len(lambdas_and_prompts)
    
    return {k: v for k, v in zip(lambdas_and_prompts, highest_lp_indices)}

def get_highest_lp_indices(model,
                           tokenizer,
                           model_name: SupportedModel, 
                           context_sequences: List[str], 
                           no_context_sequences: List[str], 
                           options: List[str], 
                           lambdas: List[float], 
                           temperature: float = 0.7):
    """
    Wrapper function to call the correct function for the model type.
    """

    def batch_get_indices_calls(fn):
        BATCH_SIZE = 4 # batch size for HF from experimentation
        
        result = {}
        for i in range(0, len(no_context_sequences), BATCH_SIZE):
            batch_prompts = no_context_sequences[i: i + BATCH_SIZE]
            batch_context_prompts = context_sequences[i: i + BATCH_SIZE]
            res = fn(model=model, 
                    tokenizer=tokenizer,
                    no_context_sequences=batch_prompts, 
                    context_sequences=batch_context_prompts,
                    options=options,
                    lambdas=lambdas)
            result.update(res) # merge consecutive dictionaries
        
        return result
    
    # get model type from HF mappings if from hugging face, alternatively
    # checking if model is from native llama api
    model_type = HF_MODEL_CLASS.get(model_name) if model_name else type(model)

    if model_type == AutoModelForCausalLM:
        return batch_get_indices_calls(get_highest_lp_indices_causal_lm)
    elif model_type == AutoModelForSeq2SeqLM:
        return batch_get_indices_calls(get_highest_lp_indices_seq2seq)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

