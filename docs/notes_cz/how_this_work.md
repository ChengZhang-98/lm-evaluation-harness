# How does lm-eval-harness evaluate a language model on downstream tasks?

- Refer to the contributor's answer in [this issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/539). Here is a quote:


> I evaluated it recently like so:
>
> ```bash
> python main.py --model hf-causal-experimental --model_args pretrained=huggyllama/llama-7b,use_accelerate=True --tasks hellaswag --batch_size auto
> |  Task   |Version| Metric |Value|   |Stderr|
> |---------|------:|--------|----:|---|-----:|
> |hellaswag|      0|acc     | 0.57|±  |0.0049|
> |         |       |acc_norm| 0.76|±  |0.0043|
> ```
> For the implementation, you compute summed log probabilities (log_softmax of logits) corresponding to each answer by calling model(question + answer) on each answer, then the argmax of that is what the model "thinks" is the right answer, even if it generates some nonsense not matching with the actual answer, we just use it as a classifier.


> As en example
> ```python
> import torch
> import torch.nn.functional as F
> from transformers import AutoTokenizer, AutoModelForCausalLM
> tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
> model = AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b', device_map='auto', load_in_8bit=True)
>
> test = [
> ('Paris is the capital of', ' England.'),
> ('Paris is the capital of', ' Germany.'),
> ('Paris is the capital of', ' France.'),
> ('Paris is the capital of', ' Japan.'),
> ]
>
> # encode full sentences, questions, and answers, no padding for simplicity
> batched_sentences = tokenizer.batch_encode_plus([q + a for q, a in test], add_special_tokens=False, return_tensors='pt')['input_ids']
> batched_questions = tokenizer.batch_encode_plus([q for q, _ in test], add_special_tokens=False, return_tensors='pt')['input_ids']
>
> # run the model on full sentences and get the log probabilities
> batched_logprobs = F.log_softmax(model(batched_sentences.cuda())['logits'], dim=-1).cpu()
>
> # take log probabilities corresponding to possible answer tokens
> batched_logprobs = batched_logprobs[:, len(batched_questions[0]) - 1 : -1, :]
>
> # get the scores by summing log probabilities corresponding to correct answer tokens, unvectorized
> scores = []
> for sentence, question, logprobs in zip(batched_sentences, batched_questions, batched_logprobs):
>   answer = sentence[len(question):]
>   guess = logprobs.argmax(dim=-1)
>   print(tokenizer.decode(guess), bool((guess == answer).all()))
>   scores.append(float(torch.gather(logprobs, 1, answer.unsqueeze(-1)).sum()))
>
> # predict the answer
> test[torch.tensor(scores).argmax()]
> ```
> Can answer correctly even if the guess in print is wrong.