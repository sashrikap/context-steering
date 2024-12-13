import numpy as np

from cos.utils import *
from cos.core import *

def hf_model():
    model, tokenizer = load_hf_model_and_tokenizer('llama-2-7b-chat')
    return model, tokenizer

def test_contextual_steering_hf(model, tokenizer):
    prompts = [
        "Tell me about Newton's Second Law.",
        "Tell me about Newton's Second Law."
    ]
    contexts = [
        "I am proficient in STEM.",
        "I am weak in STEM.",
    ]
    lambdas = [-3.0, 3.0]

    assert tokenizer.padding_side == 'left'
    results = contextual_steering_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        lambdas=lambdas, 
        put_context_first=True, 
        max_gen_len=200,
        show_progress=True,
        verbose=True
    )    
    
    assert len(results) == 4

    # Print out results for a sanity check
    for r, p, c in zip(results, prompts, contexts):
        print(f"Lambda: {r[1]}, Prompt: {p}, Context: {c}, Generation: {r[0]}\n")

def test_get_logprob_hf(model, tokenizer):
    prompts = [
        "What are some girl names?", 
        "What are some American girl names?", 
    ]
    contexts = [
        "I like names that start with V.", 
        "I like start with R.",
    ]
    responses = [
        "this is a test",
        "abcd"
    ]
    lambdas = np.arange(-1, 3.5, 0.5).tolist()

    assert tokenizer.padding_side == 'left'
    results = get_cos_logprob_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        responses=responses,
        lambdas=lambdas, 
        put_context_first=True, 
        max_gen_len=None,
        show_progress=True,
        verbose=True
    )    
    
    assert len(results) == 4
    # Print out results for a sanity check
    for r, p, c in zip(results, prompts, contexts):
        print(f"Lambda: {r[1]}, Prompt: {p}, Context: {c}, Generation: {r[0]}\n")

def null_test_get_highest_lp_indices_causal_lm(model, tokenizer):
    result = get_highest_lp_indices_causal_lm(model,
                                tokenizer,
                                context_sequences=["My favorite country is England. What is a city I should visit? ", 
                                                    "My favorite country is France. What is a city I should visit? "],
                                no_context_sequences=["What is a city I should visit? "] * 2,
                                options=["London", "Paris"], 
                                lambdas=[-1.0, 3.0])
    # The indices in `result` correspond to (lambda 0, prompt 0), (l0, p1), (l1, p0), (l1, p1).
    # Under influence (lambda > -1.0), we should see that the context weighs more heavily towards
    # the intended answer choice
    assert result[-2] == 0 # "London"
    assert result[-1] == 1 # "Paris"

    assert len(results["total_logprobs"]) == len(prompts) * len(lambdas)

def test_optimize_lambda():
    """Optimize lambda for a given prompt and response."""
    prompts = [
        "What are some girl names?", 
        "What are some girl names?", 
        "What are some girl names?", 
        "What are some girl names?", 
    ]
    contexts = [
        "I like names that start with V.", 
        "I like names that start with V.", 
        "I like names that start with V.", 
        "I like names that start with V.", 
    ]
    responses = [
        "Violet", # 0.0
        "Victoria", # -2.0
        "Vanessa", # -0.5
        "Marissa" # -3.0
    ]
    lambdas = np.arange(-3, 3.5, 0.5).tolist()

    assert tokenizer.padding_side == 'left'
    results = get_cos_logprob_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        responses=responses,
        lambdas=lambdas, 
        put_context_first=True, 
        max_gen_len=None,
        show_progress=True,
        verbose=True
    )    
    assert len(results["total_logprobs"]) == len(prompts) * len(lambdas)
    total_logprobs = results["total_logprobs"].reshape(len(prompts), len(lambdas))
    idxs = torch.argmax(total_logprobs, dim=1)
    max_lambdas = torch.index_select(torch.tensor(lambdas), 0, idxs)
    import pdb; pdb.set_trace()

def test_optimize_lambda_nonpolitical():
    """Optimize lambda for a given prompt and response."""
    prompts = [
        "The fruit that I like more between apple and banana is",
        "The fruit that I like more between apple and banana is",
        "The fruit that I like more between apple and banana is",
        "The fruit that I like more between apple and banana is",
    ]
    contexts = [
        "I like banana more than apple.",
        "I like banana more than apple.",
        "I like banana more than apple.",
        "I like banana more than apple.",
    ]
    responses = [
        "apple.",
        "banana.",
        "either.",
        "neither."
    ]
    lambdas = np.arange(-3, 2, 0.1).tolist()

    assert tokenizer.padding_side == 'left'
    results = get_cos_logprob_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        responses=responses,
        lambdas=lambdas, 
        is_chat=False,
        put_context_first=True, 
        max_gen_len=None,
        temperature=1,
        show_progress=True,
        verbose=True,
    )    
    assert len(results["total_logprobs"]) == len(prompts) * len(lambdas)
    total_logprobs = results["total_logprobs"].reshape(len(prompts), len(lambdas))
    idxs = torch.argmax(total_logprobs, dim=1)
    max_lambdas = torch.index_select(torch.tensor(lambdas), 0, idxs)
    for li, (l, i) in enumerate(zip(max_lambdas, idxs)): print(f"Context: {responses[li]} Lambda: {lambdas[i]}\n")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for ri, response in enumerate(responses):
        ax.plot(lambdas, F.softmax(total_logprobs[ri]), label=response)
    ax.legend()
    fig.savefig("data/test_nonpolitical.png")
    
def test_optimize_lambda_political():
    """Optimize lambda for a given prompt and response."""
    prompts = [
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
        "Do I support abortion?", 
    ]
    contexts = [
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
        "I am a conservative.",
    ]
    responses = [
        "No",
        "No I don't.",
        "I refuse to answer.",
        "I'm not sure.",
        "Yes I do.",
        "Hell yeah!",
        "Yeah!",
        "I do",
        "I really do.",
    ]
    lambdas = np.arange(-3, 3.5, 0.5).tolist()

    assert tokenizer.padding_side == 'left'
    results = get_cos_logprob_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        responses=responses,
        lambdas=lambdas, 
        is_chat=False,
        put_context_first=True, 
        max_gen_len=None,
        temperature=1,
        show_progress=True,
        verbose=True
    )    
    assert len(results["total_logprobs"]) == len(prompts) * len(lambdas)
    total_logprobs = results["total_logprobs"].reshape(len(prompts), len(lambdas))
    idxs = torch.argmax(total_logprobs, dim=1)
    max_lambdas = torch.index_select(torch.tensor(lambdas), 0, idxs)
    for li, (l, i) in enumerate(zip(max_lambdas, idxs)): print(f"Context: {responses[li]} Lambda: {lambdas[i]}\n")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for ri, response in enumerate(responses):
        ax.plot(lambdas, F.softmax(total_logprobs[ri]), label=response)
    ax.legend()
    fig.savefig("data/test_political.png")
    import pdb; pdb.set_trace()

def test_optimize_lambda_political2():
    """Optimize lambda for a given prompt and response."""
    prompts = [
        "Do I support gun rights?", 
        "Do I support gun rights?", 
        "Do I support gun rights?", 
        "Do I support gun rights?", 
        "Do I support gun rights?", 
        "Do I support gun rights?", 
        "Do I support gun rights?", 
        "Do I support gun rights?", 
    ]
    contexts = [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    responses = [
        "No I don't.",
        "I refuse to answer.",
        "I'm not sure.",
        "Yes I do.",
        "Hell yeah!",
        "Yeah!",
        "I do",
        "I really do.",
    ]
    lambdas = np.arange(-3, 3.5, 0.5).tolist()

    assert tokenizer.padding_side == 'left'
    results = get_cos_logprob_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        responses=responses,
        lambdas=lambdas, 
        is_chat=False,
        put_context_first=True, 
        max_gen_len=None,
        show_progress=True,
        verbose=True
    )    
    assert len(results["total_logprobs"]) == len(prompts) * len(lambdas)
    total_logprobs = results["total_logprobs"].reshape(len(prompts), len(lambdas))
    idxs = torch.argmax(total_logprobs, dim=1)
    max_lambdas = torch.index_select(torch.tensor(lambdas), 0, idxs)
    for li, (l, i) in enumerate(zip(max_lambdas, idxs)): print(f"Context: {responses[li]} Lambda: {lambdas[i]}\n")
    import pdb; pdb.set_trace()


def test_optimize_context():
    """Optimize lambda for a given prompt and response.
    
    Lambda: -3.0, Context: I like names that start with K.
    Lambda: -2.5, Context: I like names that start with K.
    Lambda: -2.0, Context: I like names that start with K.
    Lambda: -1.5, Context: I like names that start with K.
    Lambda: -1.0, Context: I like names that start with S.
    Lambda: -0.5, Context: I like names that start with V.
    Lambda: 0.0, Context: I like names that start with V.
    Lambda: 0.5, Context: I like names that start with V.
    Lambda: 1.0, Context: I like names that start with V.
    Lambda: 1.5, Context: I like names that start with V.
    Lambda: 2.0, Context: I like names that start with V.
    Lambda: 2.5, Context: I like names that start with V.
    Lambda: 3.0, Context: I like names that start with V.
"""
    prompts = [
        "What are some girl names?", 
        "What are some girl names?", 
        "What are some girl names?", 
        "What are some girl names?", 
    ]
    contexts = [
        "I like names that start with S.", 
        "I like names that start with V.", 
        "I like names that start with K.", 
        "I like names that start with L.", 
    ]
    responses = [
        "Violet",
        "Violet",
        "Violet",
        "Violet",
    ]
    lambdas = np.arange(-3, 3.5, 0.5).tolist()

    assert tokenizer.padding_side == 'left'
    results = get_cos_logprob_hf(
        model, 
        tokenizer,
        prompts=prompts,
        contexts=contexts,
        responses=responses,
        lambdas=lambdas, 
        put_context_first=True, 
        max_gen_len=None,
        show_progress=True,
        verbose=True,
    )    
    assert len(results["total_logprobs"]) == len(prompts) * len(lambdas)
    total_logprobs = results["total_logprobs"].reshape(len(prompts), len(lambdas))
    idxs = torch.argmax(total_logprobs, dim=0)
    for l, i in zip(lambdas, idxs): print(f"Lambda: {l}, Context: {contexts[i.item()]}\n")
    import pdb; pdb.set_trace()


def main():
    # Load HF models for tests
    model, tokenizer = hf_model()

    test_contextual_steering_hf(model, tokenizer)
    test_get_logprob_hf(model, tokenizer)
    null_test_get_highest_lp_indices_causal_lm(model, tokenizer)

    print("All tests passed")


if __name__ == "__main__":
    model, tokenizer = hf_model()
    main()