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
