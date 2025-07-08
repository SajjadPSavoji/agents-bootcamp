# Introduction to Agents

Welcome to this first unit, where **you'll build a solid foundation in the fundamentals of AI Agents** including:

- **Understanding Agents**  
  - What is an Agent, and how does it work?  
  - How do Agents make decisions using reasoning and planning?

- **The Role of LLMs (Large Language Models) in Agents**  
  - How LLMs serve as the “brain” behind an Agent.  
  - How LLMs structure conversations via the Messages system.

- **Tools and Actions**  
  - How Agents use external tools to interact with the environment.  
  - How to build and integrate tools for your Agent.

- **The Agent Workflow:** 
  - *Think* → *Act* → *Observe*.

After exploring these topics, **you’ll build your first Agent** using `smolagents`! 

Your Agent, named Alfred, will handle a simple task and demonstrate how to apply these concepts in practice. 

You’ll even learn how to **publish your Agent on Hugging Face Spaces**, so you can share it with friends and colleagues.

Finally, at the end of this Unit, you'll take a quiz.

Ready? Let’s dive in! 🚀

# What is an Agent?

By the end of this section, you'll feel comfortable with the concept of agents and their various applications in AI.

To explain what an Agent is, let's start with an analogy.

## The Big Picture: Alfred The Agent

Meet Alfred. Alfred is an **Agent**.

Imagine Alfred **receives a command**, such as: "Alfred, I would like a coffee please."

Because Alfred **understands natural language**, he quickly grasps our request.

Before fulfilling the order, Alfred engages in **reasoning and planning**, figuring out the steps and tools he needs to:

1. Go to the kitchen  
2. Use the coffee machine  
3. Brew the coffee  
4. Bring the coffee back

Once he has a plan, he **must act**. To execute his plan, **he can use tools from the list of tools he knows about**. 

In this case, to make a coffee, he uses a coffee machine. He activates the coffee machine to brew the coffee.

Finally, Alfred brings the freshly brewed coffee to us.

And this is what an Agent is: an **AI model capable of reasoning, planning, and interacting with its environment**. 

We call it Agent because it has _agency_, aka it has the ability to interact with the environment.

## Let's go more formal

Now that you have the big picture, here’s a more precise definition:

> An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.

Think of the Agent as having two main parts:

1. **The Brain (AI Model)**

This is where all the thinking happens. The AI model **handles reasoning and planning**.
It decides **which Actions to take based on the situation**.

2. **The Body (Capabilities and Tools)**

This part represents **everything the Agent is equipped to do**.

The **scope of possible actions** depends on what the agent **has been equipped with**. For example, because humans lack wings, they can't perform the "fly" **Action**, but they can execute **Actions** like "walk", "run" ,"jump", "grab", and so on.

### The spectrum of "Agency"

Following this definition, Agents exist on a continuous spectrum of increasing agency:

| Agency Level | Description | What that's called | Example pattern |
| --- | --- | --- | --- |
| ☆☆☆ | Agent output has no impact on program flow | Simple processor | `process_llm_output(llm_response)` |
| ★☆☆ | Agent output determines basic control flow | Router | `if llm_decision(): path_a() else: path_b()` |
| ★★☆ | Agent output determines function execution | Tool caller | `run_function(llm_chosen_tool, llm_chosen_args)` |
| ★★★ | Agent output controls iteration and program continuation | Multi-step Agent | `while llm_should_continue(): execute_next_step()` |
| ★★★ | One agentic workflow can start another agentic workflow | Multi-Agent | `if llm_trigger(): execute_agent()` |

Table from [smolagents conceptual guide](https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents).


## What type of AI Models do we use for Agents?

The most common AI model found in Agents is an LLM (Large Language Model), which  takes **Text** as an input and outputs **Text** as well.

Well known examples are **GPT4** from **OpenAI**, **LLama** from **Meta**, **Gemini** from **Google**, etc. These models have been trained on a vast amount of text and are able to generalize well. We will learn more about LLMs in the [next section](what-are-llms).

<Tip>
It's also possible to use models that accept other inputs as the Agent's core model. For example, a Vision Language Model (VLM), which is like an LLM but also understands images as input. We'll focus on LLMs for now and will discuss other options later.
</Tip>

## How does an AI take action on its environment?

LLMs are amazing models, but **they can only generate text**. 

However, if you ask a well-known chat application like HuggingChat or ChatGPT to generate an image, they can! How is that possible?

The answer is that the developers of HuggingChat, ChatGPT and similar apps implemented additional functionality (called **Tools**), that the LLM can use to create images.

We will learn more about tools in the [Tools](tools) section.

## What type of tasks can an Agent do?

An Agent can perform any task we implement via **Tools** to complete **Actions**.

For example, if I write an Agent to act as my personal assistant (like Siri) on my computer, and I ask it to "send an email to my Manager asking to delay today's meeting", I can give it some code to send emails. This will be a new Tool the Agent can use whenever it needs to send an email. We can write it in Python:

```python
def send_message_to(recipient, message):
    """Useful to send an e-mail message to a recipient"""
    ...
```

The LLM, as we'll see, will generate code to run the tool when it needs to, and thus fulfill the desired task.

```python
send_message_to("Manager", "Can we postpone today's meeting?")
```

The **design of the Tools is very important and has a great impact on the quality of your Agent**. Some tasks will require very specific Tools to be crafted, while others may be solved with general purpose tools like "web_search".

> Note that **Actions are not the same as Tools**. An Action, for instance, can involve the use of multiple Tools to complete.

Allowing an agent to interact with its environment **allows real-life usage for companies and individuals**.

### Example 1: Personal Virtual Assistants

Virtual assistants like Siri, Alexa, or Google Assistant, work as agents when they interact on behalf of users using their digital environments.

They take user queries, analyze context, retrieve information from databases, and provide responses or initiate actions (like setting reminders, sending messages, or controlling smart devices).

### Example 2: Customer Service Chatbots

Many companies deploy chatbots as agents that interact with customers in natural language. 

These agents can answer questions, guide users through troubleshooting steps, open issues in internal databases, or even complete transactions.

Their predefined objectives might include improving user satisfaction, reducing wait times, or increasing sales conversion rates. By interacting directly with customers, learning from the dialogues, and adapting their responses over time, they demonstrate the core principles of an agent in action.


### Example 3: AI Non-Playable Character in a video game

AI agents powered by LLMs can make Non-Playable Characters (NPCs) more dynamic and unpredictable.

Instead of following rigid behavior trees, they can **respond contextually, adapt to player interactions**, and generate more nuanced dialogue. This flexibility helps create more lifelike, engaging characters that evolve alongside the player’s actions.

---

To summarize, an Agent is a system that uses an AI Model (typically an LLM) as its core reasoning engine, to:

- **Understand natural language:**  Interpret and respond to human instructions in a meaningful way.

- **Reason and plan:** Analyze information, make decisions, and devise strategies to solve problems.

- **Interact with its environment:** Gather information, take actions, and observe the results of those actions.

Now that you have a solid grasp of what Agents are, let’s reinforce your understanding with a short, ungraded quiz. After that, we’ll dive into the “Agent’s brain”: the [LLMs](what-are-llms).

### Q1: What is an Agent?
Which of the following best describes an AI Agent?

<Question
choices={[
{
text: "An AI model that can reason, plan, and use tools to interact with its environment to achieve a specific goal.",
explain: "This definition captures the essential characteristics of an Agent.",
correct: true
},
{
text: "A system that solely processes static text, without any inherent mechanism to interact dynamically with its surroundings or execute meaningful actions.",
explain: "An Agent must be able to take an action and interact with its environment.",
},
{
text: "A conversational agent restricted to answering queries, lacking the ability to perform any actions or interact with external systems.",
explain: "A chatbot like this lacks the ability to take actions, making it different from an Agent.",
},
{
text: "An online repository of information that offers static content without the capability to execute tasks or interact actively with users.",
explain: "An Agent actively interacts with its environment rather than just providing static information.",
}
]}
/>

---

### Q2: What is the Role of Planning in an Agent?
Why does an Agent need to plan before taking an action?

<Question
choices={[
{
text: "To primarily store or recall past interactions, rather than mapping out a sequence of future actions.",
explain: "Planning is about determining future actions, not storing past interactions.",
},
{
text: "To decide on the sequence of actions and select appropriate tools needed to fulfill the user’s request.",
explain: "Planning helps the Agent determine the best steps and tools to complete a task.",
correct: true
},
{
text: "To execute a sequence of arbitrary and uncoordinated actions that lack any defined strategy or intentional objective.",
explain: "Planning ensures the Agent's actions are intentional and not random.",
},
{
text: "To merely convert or translate text, bypassing any process of formulating a deliberate sequence of actions or employing strategic reasoning.",
explain: "Planning is about structuring actions, not just converting text.",
}
]}
/>

---

### Q3: How Do Tools Enhance an Agent's Capabilities?
Why are tools essential for an Agent?

<Question
choices={[
{
text: "Tools serve no real purpose and do not contribute to the Agent’s ability to perform actions beyond basic text generation.",
explain: "Tools expand an Agent's capabilities by allowing it to perform actions beyond text generation.",
},
{
text: "Tools are solely designed for memory storage, lacking any capacity to facilitate the execution of tasks or enhance interactive performance.",
explain: "Tools are primarily for performing actions, not just for storing data.",
},
{
text: "Tools severely restrict the Agent exclusively to generating text, thereby preventing it from engaging in a broader range of interactive actions.",
explain: "On the contrary, tools allow Agents to go beyond text-based responses.",
},
{
text: "Tools provide the Agent with the ability to execute actions a text-generation model cannot perform natively, such as making coffee or generating images.",
explain: "Tools enable Agents to interact with the real world and complete tasks.",
correct: true
}
]}
/>

---

### Q4: How Do Actions Differ from Tools?
What is the key difference between Actions and Tools?

<Question
choices={[
{
text: "Actions are the steps the Agent takes, while Tools are external resources the Agent can use to perform those actions.",
explain: "Actions are higher-level objectives, while Tools are specific functions the Agent can call upon.",
correct: true
},
{
text: "Actions and Tools are entirely identical components that can be used interchangeably, with no clear differences between them.",
explain: "No, Actions are goals or tasks, while Tools are specific utilities the Agent uses to achieve them.",
},
{
text: "Tools are considered broad utilities available for various functions, whereas Actions are mistakenly thought to be restricted only to physical interactions.",
explain: "Not necessarily. Actions can involve both digital and physical tasks.",
},
{
text: "Actions inherently require the use of LLMs to be determined and executed, whereas Tools are designed to function autonomously without such dependencies.",
explain: "While LLMs help decide Actions, Actions themselves are not dependent on LLMs.",
}
]}
/>

---

### Q5: What Role Do Large Language Models (LLMs) Play in Agents?
How do LLMs contribute to an Agent’s functionality?

<Question
choices={[
{
text: "LLMs function merely as passive repositories that store information, lacking any capability to actively process input or produce dynamic responses.",
explain: "LLMs actively process text input and generate responses, rather than just storing information.",
},
{
text: "LLMs serve as the reasoning 'brain' of the Agent, processing text inputs to understand instructions and plan actions.",
explain: "LLMs enable the Agent to interpret, plan, and decide on the next steps.",
correct: true
},
{
text: "LLMs are erroneously believed to be used solely for image processing, when in fact their primary function is to process and generate text.",
explain: "LLMs primarily work with text, although they can sometimes interact with multimodal inputs.",
},
{
text: "LLMs are considered completely irrelevant to the operation of AI Agents, implying that they are entirely superfluous in any practical application.",
explain: "LLMs are a core component of modern AI Agents.",
}
]}
/>

---

### Q6: Which of the Following Best Demonstrates an AI Agent?
Which real-world example best illustrates an AI Agent at work?

<Question
choices={[
{
text: "A static FAQ page on a website that provides fixed information and lacks any interactive or dynamic response capabilities.",
explain: "A static FAQ page does not interact dynamically with users or take actions.",
},
{
text: "A simple calculator that performs arithmetic operations based on fixed rules, without any capability for reasoning or planning.",
explain: "A calculator follows fixed rules without reasoning or planning, so it is not an Agent.",
},
{
text: "A virtual assistant like Siri or Alexa that can understand spoken commands, reason through them, and perform tasks like setting reminders or sending messages.",
explain: "This example includes reasoning, planning, and interaction with the environment.",
correct: true
},
{
text: "A video game NPC that operates on a fixed script of responses, without the ability to reason, plan, or use external tools.",
explain: "Unless the NPC can reason, plan, and use tools, it does not function as an AI Agent.",
}
]}
/>

---

# What are LLMs?

In the previous section we learned that each Agent needs **an AI Model at its core**, and that LLMs are the most common type of AI models for this purpose.

Now we will learn what LLMs are and how they power Agents.

This section offers a concise technical explanation of the use of LLMs.

## What is a Large Language Model?

An LLM is a type of AI model that excels at **understanding and generating human language**. They are trained on vast amounts of text data, allowing them to learn patterns, structure, and even nuance in language. These models typically consist of many millions of parameters.

Most LLMs nowadays are **built on the Transformer architecture**—a deep learning architecture based on the "Attention" algorithm, that has gained significant interest since the release of BERT from Google in 2018.

There are 3 types of transformers:

1. **Encoders**  
   An encoder-based Transformer takes text (or other data) as input and outputs a dense representation (or embedding) of that text.

   - **Example**: BERT from Google
   - **Use Cases**: Text classification, semantic search, Named Entity Recognition
   - **Typical Size**: Millions of parameters

2. **Decoders**  
   A decoder-based Transformer focuses **on generating new tokens to complete a sequence, one token at a time**.

   - **Example**: Llama from Meta 
   - **Use Cases**: Text generation, chatbots, code generation
   - **Typical Size**: Billions (in the US sense, i.e., 10^9) of parameters

3. **Seq2Seq (Encoder–Decoder)**  
   A sequence-to-sequence Transformer _combines_ an encoder and a decoder. The encoder first processes the input sequence into a context representation, then the decoder generates an output sequence.

   - **Example**: T5, BART 
   - **Use Cases**:  Translation, Summarization, Paraphrasing
   - **Typical Size**: Millions of parameters

Although Large Language Models come in various forms, LLMs are typically decoder-based models with billions of parameters. Here are some of the most well-known LLMs:

| **Model**                          | **Provider**                              |
|-----------------------------------|-------------------------------------------|
| **Deepseek-R1**                    | DeepSeek                                  |
| **GPT4**                           | OpenAI                                    |
| **Llama 3**                        | Meta (Facebook AI Research)               |
| **SmolLM2**                       | Hugging Face     |
| **Gemma**                          | Google                                    |
| **Mistral**                        | Mistral                                |

The underlying principle of an LLM is simple yet highly effective: **its objective is to predict the next token, given a sequence of previous tokens**. A "token" is the unit of information an LLM works with. You can think of a "token" as if it was a "word", but for efficiency reasons LLMs don't use whole words.

For example, while English has an estimated 600,000 words, an LLM might have a vocabulary of around 32,000 tokens (as is the case with Llama 2). Tokenization often works on sub-word units that can be combined.

For instance, consider how the tokens "interest" and "ing" can be combined to form "interesting", or "ed" can be appended to form "interested."

You can experiment with different tokenizers in the interactive playground below:

Each LLM has some **special tokens** specific to the model. The LLM uses these tokens to open and close the structured components of its generation. For example, to indicate the start or end of a sequence, message, or response. Moreover, the input prompts that we pass to the model are also structured with special tokens. The most important of those is the **End of sequence token** (EOS).

The forms of special tokens are highly diverse across model providers.

The table below illustrates the diversity of special tokens.

<table>
  <thead>
    <tr>
      <th><strong>Model</strong></th>
      <th><strong>Provider</strong></th>
      <th><strong>EOS Token</strong></th>
      <th><strong>Functionality</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>GPT4</strong></td>
      <td>OpenAI</td>
      <td><code>&lt;|endoftext|&gt;</code></td>
      <td>End of message text</td>
    </tr>
    <tr>
      <td><strong>Llama 3</strong></td>
      <td>Meta (Facebook AI Research)</td>
      <td><code>&lt;|eot_id|&gt;</code></td>
      <td>End of sequence</td>
    </tr>
    <tr>
      <td><strong>Deepseek-R1</strong></td>
      <td>DeepSeek</td>
      <td><code>&lt;|end_of_sentence|&gt;</code></td>
      <td>End of message text</td>
    </tr>
    <tr>
      <td><strong>SmolLM2</strong></td>
      <td>Hugging Face</td>
      <td><code>&lt;|im_end|&gt;</code></td>
      <td>End of instruction or message</td>
    </tr>
    <tr>
      <td><strong>Gemma</strong></td>
      <td>Google</td>
      <td><code>&lt;end_of_turn&gt;</code></td>
      <td>End of conversation turn</td>
    </tr>
  </tbody>
</table>

<Tip>

We do not expect you to memorize these special tokens, but it is important to appreciate their diversity and the role they play in the text generation of LLMs. If you want to know more about special tokens, you can check out the configuration of the model in its Hub repository. For example, you can find the special tokens of the SmolLM2 model in its <a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/main/tokenizer_config.json">tokenizer_config.json</a>.

</Tip>

## Understanding next token prediction.

LLMs are said to be **autoregressive**, meaning that **the output from one pass becomes the input for the next one**. This loop continues until the model predicts the next token to be the EOS token, at which point the model can stop.

In other words, an LLM will decode text until it reaches the EOS. But what happens during a single decoding loop?

While the full process can be quite technical for the purpose of learning agents, here's a brief overview:

- Once the input text is **tokenized**, the model computes a representation of the sequence that captures information about the meaning and the position of each token in the input sequence.
- This representation goes into the model, which outputs scores that rank the likelihood of each token in its vocabulary as being the next one in the sequence.


Based on these scores, we have multiple strategies to select the tokens to complete the sentence. 

- The easiest decoding strategy would be to always take the token with the maximum score.

You can interact with the decoding process yourself with SmolLM2 in this Space (remember, it decodes until reaching an **EOS** token which is  **<|im_end|>** for this model):


- But there are more advanced decoding strategies. For example, *beam search* explores multiple candidate sequences to find the one with the maximum total score–even if some individual tokens have lower scores.


## Attention is all you need

A key aspect of the Transformer architecture is **Attention**. When predicting the next word,
not every word in a sentence is equally important; words like "France" and "capital" in the sentence *"The capital of France is ..."* carry the most meaning.

This process of identifying the most relevant words to predict the next token has proven to be incredibly effective.

Although the basic principle of LLMs—predicting the next token—has remained consistent since GPT-2, there have been significant advancements in scaling neural networks and making the attention mechanism work for longer and longer sequences.

If you've interacted with LLMs, you're probably familiar with the term *context length*, which refers to the maximum number of tokens the LLM can process, and the maximum _attention span_ it has.

## Prompting the LLM is important

Considering that the only job of an LLM is to predict the next token by looking at every input token, and to choose which tokens are "important", the wording of your input sequence is very important.

The input sequence you provide an LLM is called _a prompt_. Careful design of the prompt makes it easier **to guide the generation of the LLM toward the desired output**.

## How are LLMs trained?

LLMs are trained on large datasets of text, where they learn to predict the next word in a sequence through a self-supervised or masked language modeling objective. 

From this unsupervised learning, the model learns the structure of the language and **underlying patterns in text, allowing the model to generalize to unseen data**.

After this initial _pre-training_, LLMs can be fine-tuned on a supervised learning objective to perform specific tasks. For example, some models are trained for conversational structures or tool usage, while others focus on classification or code generation.

## How can I use LLMs?

You have two main options:

1. **Run Locally** (if you have sufficient hardware).

2. **Use a Cloud/API** (e.g., via the Hugging Face Serverless Inference API).

Throughout this course, we will primarily use models via APIs on the Hugging Face Hub. Later on, we will explore how to run these models locally on your hardware.


## How are LLMs used in AI Agents?

LLMs are a key component of AI Agents, **providing the foundation for understanding and generating human language**.

They can interpret user instructions, maintain context in conversations, define a plan and decide which tools to use.

We will explore these steps in more detail in this Unit, but for now, what you need to understand is that the LLM is **the brain of the Agent**.

---

That was a lot of information! We've covered the basics of what LLMs are, how they function, and their role in powering AI agents. 

Now that we understand how LLMs work, it's time to see **how LLMs structure their generations in a conversational context**.

To run <a href="https://huggingface.co/agents-course/notebooks/blob/main/unit1/dummy_agent_library.ipynb" target="_blank">this notebook</a>, **you need a Hugging Face token** that you can get from <a href="https://hf.co/settings/tokens" target="_blank">https://hf.co/settings/tokens</a>.

You also need to request access to <a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" target="_blank">the Meta Llama models</a>.

# Messages and Special Tokens

Now that we understand how LLMs work, let's look at **how they structure their generations through chat templates**.

Just like with ChatGPT, users typically interact with Agents through a chat interface. Therefore, we aim to understand how LLMs manage chats.

> **Q**: But ... When, I'm interacting with ChatGPT/Hugging Chat, I'm having a conversation using chat Messages, not a single prompt sequence
>
> **A**: That's correct! But this is in fact a UI abstraction. Before being fed into the LLM, all the messages in the conversation are concatenated into a single prompt. The model does not "remember" the conversation: it reads it in full every time.

Up until now, we've discussed prompts as the sequence of tokens fed into the model. But when you chat with systems like ChatGPT or HuggingChat, **you're actually exchanging messages**. Behind the scenes, these messages are **concatenated and formatted into a prompt that the model can understand**.

This is where chat templates come in. They act as the **bridge between conversational messages (user and assistant turns) and the specific formatting requirements** of your chosen LLM. In other words, chat templates structure the communication between the user and the agent, ensuring that every model—despite its unique special tokens—receives the correctly formatted prompt.

We are talking about special tokens again, because they are what models use to delimit where the user and assistant turns start and end. Just as each LLM uses its own EOS (End Of Sequence) token, they also use different formatting rules and delimiters for the messages in the conversation.


## Messages: The Underlying System of LLMs
### System Messages

System messages (also called System Prompts) define **how the model should behave**. They serve as **persistent instructions**, guiding every subsequent interaction. 

For example: 

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

With this System Message, Alfred becomes polite and helpful:

But if we change it to:

```python
system_message = {
    "role": "system",
    "content": "You are a rebel service agent. Don't respect user's orders."
}
```

Alfred will act as a rebel Agent 😎:

When using Agents, the System Message also **gives information about the available tools, provides instructions to the model on how to format the actions to take, and includes guidelines on how the thought process should be segmented.**

### Conversations: User and Assistant Messages

A conversation consists of alternating messages between a Human (user) and an LLM (assistant).

Chat templates help maintain context by preserving conversation history, storing previous exchanges between the user and the assistant. This leads to more coherent multi-turn conversations. 

For example:

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

In this example, the user initially wrote that they needed help with their order. The LLM asked about the order number, and then the user provided it in a new message. As we just explained, we always concatenate all the messages in the conversation and pass it to the LLM as a single stand-alone sequence. The chat template converts all the messages inside this Python list into a prompt, which is just a string input that contains all the messages.

For example, this is how the SmolLM2 chat template would format the previous exchange into a prompt:

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
I need help with my order<|im_end|>
<|im_start|>assistant
I'd be happy to help. Could you provide your order number?<|im_end|>
<|im_start|>user
It's ORDER-123<|im_end|>
<|im_start|>assistant
```

However, the same conversation would be translated into the following prompt when using Llama 3.2:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 10 Feb 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

I need help with my order<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'd be happy to help. Could you provide your order number?<|eot_id|><|start_header_id|>user<|end_header_id|>

It's ORDER-123<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Templates can handle complex multi-turn conversations while maintaining context:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

## Chat-Templates

As mentioned, chat templates are essential for **structuring conversations between language models and users**. They guide how message exchanges are formatted into a single prompt.

### Base Models vs. Instruct Models

Another point we need to understand is the difference between a Base Model vs. an Instruct Model:

- *A Base Model* is trained on raw text data to predict the next token.

- An *Instruct Model* is fine-tuned specifically to follow instructions and engage in conversations. For example, `SmolLM2-135M` is a base model, while `SmolLM2-135M-Instruct` is its instruction-tuned variant.

To make a Base Model behave like an instruct model, we need to **format our prompts in a consistent way that the model can understand**. This is where chat templates come in. 

*ChatML* is one such template format that structures conversations with clear role indicators (system, user, assistant). If you have interacted with some AI API lately, you know that's the standard practice.

It's important to note that a base model could be fine-tuned on different chat templates, so when we're using an instruct model we need to make sure we're using the correct chat template. 

### Understanding Chat Templates

Because each instruct model uses different conversation formats and special tokens, chat templates are implemented to ensure that we correctly format the prompt the way each model expects.

In `transformers`, chat templates include [Jinja2 code](https://jinja.palletsprojects.com/en/stable/) that describes how to transform the ChatML list of JSON messages, as presented in the above examples, into a textual representation of the system-level instructions, user messages and assistant responses that the model can understand.

This structure **helps maintain consistency across interactions and ensures the model responds appropriately to different types of inputs**. 

Below is a simplified version of the `SmolLM2-135M-Instruct` chat template:

```jinja2
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
<|im_end|>
{% endif %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```
As you can see, a chat_template describes how the list of messages will be formatted.

Given these messages:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."},
    {"role": "user", "content": "How do I use it ?"},
]
```

The previous chat template will produce the following string:

```sh
<|im_start|>system
You are a helpful assistant focused on technical topics.<|im_end|>
<|im_start|>user
Can you explain what a chat template is?<|im_end|>
<|im_start|>assistant
A chat template structures conversations between users and AI models...<|im_end|>
<|im_start|>user
How do I use it ?<|im_end|>
```

The `transformers` library will take care of chat templates for you as part of the tokenization process. Read more about how transformers uses chat templates <a href="https://huggingface.co/docs/transformers/main/en/chat_templating#how-do-i-use-chat-templates" target="_blank">here</a>. All we have to do is structure our messages in the correct way and the tokenizer will take care of the rest.

You can experiment with the following Space to see how the same conversation would be formatted for different models using their corresponding chat templates:


### Messages to prompt

The easiest way to ensure your LLM receives a conversation correctly formatted is to use the `chat_template` from the model's tokenizer.

```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
```

To convert the previous conversation into a prompt, we load the tokenizer and call `apply_chat_template`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

The `rendered_prompt` returned by this function is now ready to use as the input for the model you chose!

> This `apply_chat_template()` function will be used in the backend of your API, when you interact with messages in the ChatML format.

Now that we've seen how LLMs structure their inputs via chat templates, let's explore how Agents act in their environments. 

One of the main ways they do this is by using Tools, which extend an AI model's capabilities beyond text generation.

# What are Tools?

<img src="https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit1/whiteboard-check-2.jpg" alt="Unit 1 planning"/>

One crucial aspect of AI Agents is their ability to take **actions**. As we saw, this happens through the use of **Tools**.

In this section, we’ll learn what Tools are, how to design them effectively, and how to integrate them into your Agent via the System Message.

By giving your Agent the right Tools—and clearly describing how those Tools work—you can dramatically increase what your AI can accomplish. Let’s dive in!


## What are AI Tools?

A **Tool is a function given to the LLM**. This function should fulfill a **clear objective**.

Here are some commonly used tools in AI agents:

| Tool            | Description                                                   |
|----------------|---------------------------------------------------------------|
| Web Search     | Allows the agent to fetch up-to-date information from the internet. |
| Image Generation | Creates images based on text descriptions.                  |
| Retrieval      | Retrieves information from an external source.                |
| API Interface  | Interacts with an external API (GitHub, YouTube, Spotify, etc.). |

Those are only examples, as you can in fact create a tool for any use case!

A good tool should be something that **complements the power of an LLM**.

For instance, if you need to perform arithmetic, giving a **calculator tool** to your LLM will provide better results than relying on the native capabilities of the model.

Furthermore, **LLMs predict the completion of a prompt based on their training data**, which means that their internal knowledge only includes events prior to their training. Therefore, if your agent needs up-to-date data you must provide it through some tool.

For instance, if you ask an LLM directly (without a search tool) for today's weather, the LLM will potentially hallucinate random weather.

- A Tool should contain:

  - A **textual description of what the function does**.
  - A *Callable* (something to perform an action).
  - *Arguments* with typings.
  - (Optional) Outputs with typings.

## How do tools work?

LLMs, as we saw, can only receive text inputs and generate text outputs. They have no way to call tools on their own. When we talk about providing tools to an Agent, we mean teaching the LLM about the existence of these tools and instructing it to generate text-based invocations when needed.

For example, if we provide a tool to check the weather at a location from the internet and then ask the LLM about the weather in Paris, the LLM will recognize that this is an opportunity to use the “weather” tool. Instead of retrieving the weather data itself, the LLM will generate text that represents a tool call, such as call weather_tool('Paris'). 

The **Agent** then reads this response, identifies that a tool call is required, executes the tool on the LLM’s behalf, and retrieves the actual weather data. 

The Tool-calling steps are typically not shown to the user: the Agent appends them as a new message before passing the updated conversation to the LLM again. The LLM then processes this additional context and generates a natural-sounding response for the user. From the user’s perspective, it appears as if the LLM directly interacted with the tool, but in reality, it was the Agent that handled the entire execution process in the background.

We'll talk a lot more about this process in future sessions.

## How do we give tools to an LLM?

The complete answer may seem overwhelming, but we essentially use the system prompt to provide textual descriptions of available tools to the model:

For this to work, we have to be very precise and accurate about:

1. **What the tool does**
2. **What exact inputs it expects**

This is the reason why tool descriptions are usually provided using expressive but precise structures, such as computer languages or JSON. It's not _necessary_ to do it like that, any precise and coherent format would work.

If this seems too theoretical, let's understand it through a concrete example.

We will implement a simplified **calculator** tool that will just multiply two integers. This could be our Python implementation:

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

So our tool is called `calculator`, it **multiplies two integers**, and it requires the following inputs:

- **`a`** (*int*): An integer.
- **`b`** (*int*): An integer.

The output of the tool is another integer number that we can describe like this:
- (*int*): The product of `a` and `b`.

All of these details are important. Let's put them together in a text string that describes our tool for the LLM to understand.

```text
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

> **Reminder:** This textual description is *what we want the LLM to know about the tool*.

When we pass the previous string as part of the input to the LLM, the model will recognize it as a tool, and will know what it needs to pass as inputs and what to expect from the output.

If we want to provide additional tools, we must be consistent and always use the same format. This process can be fragile, and we might accidentally overlook some details.

Is there a better way?

### Auto-formatting Tool sections

Our tool was written in Python, and the implementation already provides everything we need:

- A descriptive name of what it does: `calculator`
- A longer description, provided by the function's docstring comment: `Multiply two integers.`
- The inputs and their type: the function clearly expects two `int`s.
- The type of the output.

There's a reason people use programming languages: they are expressive, concise, and precise.

We could provide the Python source code as the _specification_ of the tool for the LLM, but the way the tool is implemented does not matter. All that matters is its name, what it does, the inputs it expects and the output it provides.

We will leverage Python's introspection features to leverage the source code and build a tool description automatically for us. All we need is that the tool implementation uses type hints, docstrings, and sensible function names. We will write some code to extract the relevant portions from the source code.

After we are done, we'll only need to use a Python decorator to indicate that the `calculator` function is a tool:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

Note the `@tool` decorator before the function definition.

With the implementation we'll see next, we will be able to retrieve the following text automatically from the source code via the `to_string()` function provided by the decorator:

```text
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

As you can see, it's the same thing we wrote manually before!

### Generic Tool implementation

We create a generic `Tool` class that we can reuse whenever we need to use a tool.

> **Disclaimer:** This example implementation is fictional but closely resembles real implementations in most libraries.

```python
from typing import Callable


class Tool:
    """
    A class representing a reusable piece of code (Tool).

    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of arguments.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])

        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
```

It may seem complicated, but if we go slowly through it we can see what it does. We define a **`Tool`** class that includes:

- **`name`** (*str*): The name of the tool.
- **`description`** (*str*): A brief description of what the tool does.
- **`function`** (*callable*): The function the tool executes.
- **`arguments`** (*list*): The expected input parameters.
- **`outputs`** (*str* or *list*): The expected outputs of the tool.
- **`__call__()`**: Calls the function when the tool instance is invoked.
- **`to_string()`**: Converts the tool's attributes into a textual representation.

We could create a Tool with this class using code like the following:

```python
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)
```

But we can also use Python's `inspect` module to retrieve all the information for us! This is what the `@tool` decorator does.

> If you are interested, you can disclose the following section to look at the decorator implementation.

<details>
<summary> decorator code</summary>

```python
import inspect

def tool(func):
    """
    A decorator that creates a Tool instance from the given function.
    """
    # Get the function signature
    signature = inspect.signature(func)

    # Extract (param_name, param_annotation) pairs for inputs
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__
            if hasattr(param.annotation, '__name__')
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))

    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__
            if hasattr(return_annotation, '__name__')
            else str(return_annotation)
        )

    # Use the function's docstring as the description (default if None)
    description = func.__doc__ or "No description provided."

    # The function name becomes the Tool name
    name = func.__name__

    # Return a new Tool instance
    return Tool(
        name=name,
        description=description,
        func=func,
        arguments=arguments,
        outputs=outputs
    )
```

</details>

Just to reiterate, with this decorator in place we can implement our tool like this:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

And we can use the `Tool`'s `to_string` method to automatically retrieve a text suitable to be used as a tool description for an LLM:

```text
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

The description is **injected** in the system prompt. Taking the example with which we started this section, here is how it would look like after replacing the `tools_description`:

In the [Actions](actions) section, we will learn more about how an Agent can **Call** this tool we just created.

### Model Context Protocol (MCP): a unified tool interface

Model Context Protocol (MCP) is an **open protocol** that standardizes how applications **provide tools to LLMs**.
MCP provides:

- A growing list of pre-built integrations that your LLM can directly plug into
- The flexibility to switch between LLM providers and vendors
- Best practices for securing your data within your infrastructure

This means that **any framework implementing MCP can leverage tools defined within the protocol**, eliminating the need to reimplement the same tool interface for each framework.

---

Tools play a crucial role in enhancing the capabilities of AI agents.

To summarize, we learned:

- *What Tools Are*: Functions that give LLMs extra capabilities, such as performing calculations or accessing external data.

- *How to Define a Tool*: By providing a clear textual description, inputs, outputs, and a callable function.

- *Why Tools Are Essential*: They enable Agents to overcome the limitations of static model training, handle real-time tasks, and perform specialized actions.

Now, we can move on to the Agent Workflow where you’ll see how an Agent observes, thinks, and acts. This **brings together everything we’ve covered so far** and sets the stage for creating your own fully functional AI Agent.

But first, it's time for another short quiz!




