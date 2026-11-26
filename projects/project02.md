---
title: "Project 2 — Your Own Project or the Space Invaders Agent"
---

# 1. Goal

Project 2 is a **group project with two possible topics**. Each group chooses **one**:

1. **A project defined by the group.** The topic comes from your MSc, from another class, or from a personal interest. It must use machine learning in a way that fits the course topics (learning from data, optimization, decomposition, clustering, neural networks, reinforcement learning, ...).
2. **Space Invaders.** Fork the template repository [`detiuaveiro/ai-space-invader-template`](https://github.com/detiuaveiro/ai-space-invader-template) and build **your own agent** that plays the game.

The technique is yours to choose and to defend. For Space Invaders several routes are valid, and you may combine them:

* reinforcement learning (Class 13), on pixels or on a compact state;
* evolutionary or other blind optimization of a policy (Class 02);
* supervised or imitation learning from recorded play;
* search and planning, or a hand-designed controller, as a baseline or as part of the agent.

# 2. Groups and Topic

* Groups have **1 to 3 students**. They do **not** have to be the same as in Project 1.
* Each group **sends the chosen topic by e-mail** to the instructor. For topic 1, the e-mail gives a short description of the project (problem, data, planned methods).
* A topic of your own must be real, feasible on a laptop (or with the resources you already have) and agreed with the instructor through that e-mail.

# 3. What We Expect

For **both topics**:

* **A working solution**, with a clear definition of the problem, the data or observations, and the goal.
* **Baselines**: at least a trivial one (random, majority class, the mean) and a simple one (scripted, heuristic or classical model).
* **An evaluation you define and justify**: metrics that match the problem, many runs or folds with different seeds, the spread and not only the mean, no leakage.
* **An analysis**: what works and why, where and why it fails, and how the design choices change the result.
* **Reproducibility**: fixed seeds, and code that runs from the course environment (or from a documented environment of your own).

For **Space Invaders**, in addition:

* work in your **fork** of the template (`agents/` holds your agent; the game runs with `python3 -m server.server`, and `--seed` makes a run reproducible);
* the agent sees the state of the game (`player_x`, `aliens`, `lasers`, `valid_actions`, ...) and chooses `move WEST`, `move EAST`, `shoot` or nothing;
* evaluate the agent with the game score, the waves cleared and the lives lost, over several seeds, against the dummy (random) agent and a simple scripted one;
* a result that is not reproducible with its seed is not accepted.

**Space Invaders README.** The template `README.md` already has the *Group Identification* and *Agent Architecture* sections: fill them in, and add the evaluation, the plots and the issues below them.

**Issues in Space Invaders.** If you find an issue in the Space Invaders template or game (a bug, an exploit, an unfair or broken rule), report it and describe it in your `README.md`. A valid issue can be awarded **up to 1 additional point**.

# 4. Deliverables

Delivery is in **Moodle**, at the **last class of the course** (Class 14), as in Project 1: a repository (or a zip) with:

1. **The code**, a notebook or Python files, runnable from top to bottom. For Space Invaders, the fork of the template with your agent.
2. **`README.md`** with *all* the information of the project and the **identification of the group** (names and student numbers, course, date, topic):

| Section | Content |
|:------------|:--------------------------------------|
| **1. Identification** | names and student numbers, course, date, the topic and the project in one line |
| **2. Problem** | the problem or the game, the data or observations, the goal, and the justification of the topic |
| **3. Approach** | the models or the agent, the choices and their justification, hyper-parameters and search space |
| **4. Methodology** | the evaluation protocol (split, folds or runs, metrics, baselines, spread), preprocessing, and how leakage is avoided |
| **5. Evaluation and conclusion** | the results table (mean and spread, with the baselines), the plots, the analysis of errors or failures, the limits, and what you would do next |
| **6. Issues** *(Space Invaders)* | the issues found in the template, each with a description and how to reproduce it |

**Plots and diagrams are mandatory**: a diagram of the pipeline or of the agent, the data or the game behaviour, a comparison of the solution with the baselines with its spread, and a learning curve or a diagnostic plot. Every figure has a caption and is discussed in the text.

**Presentation.** There is **no** presentation by default. A group is called for an **in-person presentation only when the work has critical issues** (for example, doubts about authorship, a result that cannot be reproduced, or a serious methodological problem).

**Assessment.** We look at: the quality of the justifications; the correctness of the methodology (a baseline, a spread, no leakage); the quality and the use of the plots; the conclusions supported by the numbers; reproducibility.

# 5. Dates

* **Release:** Class 09 (12/13 November 2026)
* **Topic e-mail:** sent by each group to the instructor, before you start the work
* **Clinic:** Class 13 (10/11 December 2026)
* **Delivery in Moodle:** Class 14 (17/18 December 2026), the last class
* **Weight:** 25% of the final grade
