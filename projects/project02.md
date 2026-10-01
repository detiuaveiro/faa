---
title: "Project 2 — An Agent That Plays a Classic Game"
---

# 1. Goal

Build an **agent that plays a traditional arcade game** and learns, or is designed, to play it well. Choose one game, for example **Frogger** or **Space Invaders**, or a game of similar size agreed with your instructor (Pong, Breakout, Pac-Man, Asteroids, ...).

The project is **open on purpose**: the technique is yours to choose and to defend. Several routes are valid, and you may combine them:

* reinforcement learning (Class 13): value-based or policy-based, on pixels or on a compact state;
* search and planning, with a model of the game or a learned model of its dynamics;
* evolutionary or other blind optimization of a policy (Class 02);
* supervised or imitation learning from recorded play;
* a hand-designed controller or heuristics, as a baseline or as part of the agent.

# 2. What We Expect

* **A working agent**, with a clear definition of the game, the observations, the actions and the reward.
* **Baselines**: at least a random agent and a simple scripted or heuristic one.
* **An evaluation you define and justify**: score, survival, success rate, learning curves, many runs with different seeds, the spread and not only the mean.
* **An analysis**: what the agent learned or exploits, where and why it fails, and how the design choices (representation, reward, algorithm, hyper-parameters) change the result.

# 3. Deliverables

* The **code**, reproducible with the course environment.
* A **`README.md`** with the same structure as Project 1: identification, the game and the agent (choices and justification), methodology, evaluation and conclusion, with plots and diagrams.
* A **short live demonstration** of the agent, with the class presentation (Class 14).

# 4. To Be Defined at the Release

These points are not closed yet and will be fixed when the project is released (Class 09):

* the game environment and the library that provides it, and the compute limits;
* the exact evaluation protocol and the baseline scores;
* the assessment criteria, the group size and the format of the presentation.

# 5. Dates

* **Release:** Class 09 (12/13 November 2026)
* **Clinic:** Class 13 (10/11 December 2026)
* **Submission deadline and presentations:** Class 14 (17/18 December 2026)
* **Weight:** 25% of the final grade
