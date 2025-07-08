# Table of Contents

| Title | Description |
|-------|-------------|
| [Definition of an Agent](1_definition_of_an_agent.md) | General example of what agents can do without technical jargon. |
| [Explain LLMs](2_explain_llms.md) | Explanation of Large Language Models, including the family tree of models and suitable models for agents. |
| [Messages and Special Tokens](3_messages_and_special_tokens.md) | Explanation of messages, special tokens, and chat-template usage. |
| [Dummy Agent Library](4_dummy_agent_library.md) | Introduction to using a dummy agent library and serverless API. |
| [Tools](5_tools.md) | Overview of Pydantic for agent tools and other common tool formats. |
| [Agent Steps and Structure](6_agent_steps_and_structure.md) | Steps involved in an agent, including thoughts, actions, observations, and a comparison between code agents and JSON agents. |
| [Thoughts](7_thoughts.md) | Explanation of thoughts and the ReAct approach. |
| [Actions](8_actions.md) | Overview of actions and stop and parse approach. |
| [Observations](9_observations.md) | Explanation of observations and append result to reflect. |
| [Quizz](10_quizz.md) | Contains quizzes to test understanding of the concepts. |
| [Simple Use Case](11_simple_use_case.md) | Provides a simple use case exercise using datetime and a Python function as a tool. |

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

For more information on how to run Jupyter Notebooks, checkout <a href="https://huggingface.co/docs/hub/notebooks">Jupyter Notebooks on the Hugging Face Hub</a>.

You also need to request access to <a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" target="_blank">the Meta Llama models</a>.

