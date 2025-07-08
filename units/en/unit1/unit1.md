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
