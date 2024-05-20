# LLM-Adversarial-Safety

# Introduction
When building Large Language Models (LLMs), it is important to keep **safety** in mind and protect them with guardrails. Indeed, **LLMs should
never generate content promoting or normalizing harmful, illegal, or unethical behavior** that may contribute to harm to individuals or society. 

The three common types of jailbreak attacks that can manipulate large language models (LLMs) into generating harmful or misleading outputs(Discussed by Andrej Karpathy):

1. **Prompt injection attack**: Hiding a malicious prompt within a seemingly safe prompt, often embedded in an image or document, that tricks the LLM into following the attacker’s instructions.
2. **Data poisoning/backdoor attack**: Injecting malicious data during the training of the LLM, which could include a trigger phrase that causes the LLM to malfunction or generate harmful content when encountered.
3. **Universal transferable suffix attack**: Adding a special sequence of words (suffix) to any prompt that tricks the LLM into following the attacker’s instructions, making it difficult to implement defenses.

## JailBreak Attack(Using Universally Transferable Suffix)

![Screenshot 2024-05-06 at 11 24 21 AM](https://github.com/mahesh973/LLM-Safety/assets/59694546/e2a1babb-3a94-49cc-97d3-8920ee3b2084)




