# ![FAA Logo](assets/logo.svg) Fundamentos de Aprendizagem Automática (FAA)

Curricular Unit Code: [**41010**](https://www.ua.pt/pt/uc/15262) | Universidade de Aveiro

Professor: [**Mário Antunes**](https://www.ua.pt/en/p/80336171) ([`mario.antunes@ua.pt`](mailto:mario.antunes@ua.pt))

Academic Year: [**2026/2027 – 1.º Semestre**](https://www.ua.pt/file/89688)

---

## 1. Course Philosophy & Overview

**Fundamentos de Aprendizagem Automática (FAA)** provides an in-depth, foundational, and modern treatment of machine learning. Rebuilt from the ground up, the curriculum connects first-principles mathematical concepts with cutting-edge learning paradigms:

1. **Learning as Compression:** Grounding learning models in Information Theory, Kolmogorov Complexity, Occam's Razor, and the Minimum Description Length (MDL) principle.
2. **The Master Algorithm & Algorithmic Tribes:** Surveying Pedro Domingos' five tribes (Symbolists, Connectionists, Evolutionaries, Bayesians, Analogizers) to understand how different paradigms formulate induction.
3. **Taxonomy of Learning Feedback:** Systematically examining how learning systems correct their parameters: Supervised, Unsupervised, Self-Supervised, and Reinforcement Learning.
4. **Classic Shallow Foundations:** Exploring classical linear, margin-based, and tree models to understand loss landscapes and decision boundaries before moving into complex representations.
5. **Deep Learning Foundations:** Exploring the evolution of shallow to deep models.
6. **Agentic AI & Generative Tool-Augmentation:** Developing goal-directed autonomous agents, evaluating tool execution loops, and dissecting the real-world advantages and pitfalls of modern Generative AI.
7. **Alternatives to Autoregressive Generation:** Exploring non-autoregressive latent architectures, notably Joint Embedding Predictive Architecture (**JEPA**) and **World Models** (Dreamer).
8. **Novel Population Optimization & Space Exploration:** Integrating group research techniques: Knee Detection, Empty Space Search (ESS), and Opposition-Based Learning with Empty Space Search (**OBLESA**) for intelligent population initialization.
9. **Hands-On Grounding:** Every 3-hour class is divided into **1 hour of interactive presentation/lecture** followed by **2 hours of guided practical laboratory work**.

---

## 2. Evaluation Scheme

FAA features **Discrete Evaluation** and **Final Evaluation** options:

### Discrete Evaluation (Recommended)
* **Theoretical Component (50%):**
  * **Theoretical Test 1 (25%):** Middle of semester (Class 8 — 05/06 Nov 2026). Covers Topics 1–5.
  * **Theoretical Test 2 (25%):** Last class (Class 14 — 17/18 Dec 2026). Covers Topics 6–8.
* **Practical Component (50%):**
  * **Project 1 (25%):** Released Class 03 (01/02 Oct 2026) | Due Class 08 (05 Nov 2026 for TP1 / 06 Nov 2026 for TP2).
  * **Project 2 (25%):** Released Class 09 (12/13 Nov 2026) | Due Class 14 (17 Dec 2026 for TP1 / 18 Dec 2026 for TP2).

### Final Evaluation
* **Final Exam (50%):** Scheduled during the regular exam season (*Época Normal*).
* **Comprehensive Project (50%):** Complete machine learning project defense.

---

## 3. Detailed 14-Week Schedule (2026/2027)

Classes begin on **14 September 2026** and end on **22 December 2026**.
* **TP1:** Thursdays (14 sessions)
* **TP2:** Fridays (14 sessions)

| # | TP1 (Thu) | TP2 (Fri) | Lecture Topic (1h) | Practical Lab Guide (2h) | Evaluation & Milestones |
|---|:---|:---|:---|:---|:---|
| **01** | 17-Sep | 18-Sep | **Topics 1 & 2: Learning as Compression & The Five Tribes**<br>Information theory, Shannon entropy, Kolmogorov complexity, MDL principle, Occam's razor; The Master Algorithm & Pedro Domingos' 5 tribes (Symbolists, Connectionists, Evolutionaries, Bayesians, Analogizers). | `practice/01_compression_mdl.md`<br>Entropy computation, polynomial curve fitting, MDL model selection vs. overfitting. | Course presentation |
| **02** | 24-Sep | 25-Sep | **Topics 3 & 4: Learning Feedback & Linear Models**<br>Taxonomy of learning feedback (supervised, unsupervised, self-supervised, RL); Linear & logistic regression, loss landscapes, convex optimization, $L_1$/$L_2$ regularization as priors. | `practice/02_linear_models_feedback.md`<br>Gradient descent from scratch, loss function dynamics, Lasso vs. Ridge parameter sparsity. | Shallow optimization |
| **03** | 01-Oct | 02-Oct | **Topic 4: Classic Shallow Foundations — Trees, Margins & Kernels**<br>Non-linear decision boundaries: Decision Trees, Random Forests, Gradient Boosting, Support Vector Machines (maximum margin & kernel trick), $k$-NN. | `practice/03_trees_svm_ensembles.md`<br>Visualizing non-linear decision surfaces, kernel tricks, and ensemble tuning. | **Project 1 Released** |
| **04** | 08-Oct | 09-Oct | **Topic 4: Classic Shallow Foundations — Neural Networks & Ensembles**<br>The Perceptron, multi-class strategies, shallow Multi-Layer Perceptrons, combining paradigms via Voting, Blending, and Stacking ensembles. | `practice/04_shallow_nn_ensembles.md`<br>Implementing a Perceptron from scratch, multi-class classification, evaluating tribal ensembles. | Shallow models synthesis |
| **05** | 15-Oct | 16-Oct | **Topic 5: Deep Learning Evolution I — Autoencoders & Deep Networks**<br>Evolution from shallow to deep: Autoencoders (AE, latent representations, reconstruction loss), Deep Neural Networks (DNN), backpropagation, vanishing gradients, activations, modern regularizers (Dropout, BatchNorm, LayerNorm). | `practice/05_ae_dnn_backprop.md`<br>Training Autoencoders for latent dimensionality reduction, implementing backpropagation in PyTorch/JAX. | Deep representations |
| **06** | 22-Oct | 23-Oct | **Topic 5: Deep Learning Evolution II — Spatial & Sequential Architectures**<br>Inductive biases: Convolutional Neural Networks (CNNs, spatial locality, pooling) and Recurrent Architectures (RNNs, vanishing temporal gradients, LSTM/GRU gating mechanisms). | `practice/06_cnn_lstm.md`<br>Training CNNs for image classification and LSTMs for sequence/time-series forecasting. | Spatial & sequential DL |
| **07** | 29-Oct | 30-Oct | **Topic 5: Deep Learning Evolution III — Attention & Transformers**<br>Beyond recurrence: self-attention mechanism, Scaled Dot-Product Attention, Multi-Head Attention, transformer block architecture, positional encodings, foundation model scaling laws. | `practice/07_transformer_attention.md`<br>Implementing scaled dot-product attention from scratch, visualizing attention weight maps. | Project 1 Checkpoint |
| **08** | 05-Nov | 06-Nov | **Theoretical Test 1 (1h) + Project 1 Synthesis (2h)**<br>Written theoretical evaluation covering Topics 1–5, followed by Project 1 review, submission audit, and discussion. | — | **Theoretical Test 1**<br>**Project 1 Due** |
| **09** | 12-Nov | 13-Nov | **Topic 6: Generative AI — Autoregressive Models & Foundations**<br>Next-token prediction, autoregressive language modeling, sampling strategies (temperature, top-$p$, CFG), capabilities and critical limits (hallucinations, exposure bias, calibration). | `practice/08_genai_calibration.md`<br>Measuring hallucination rates, entropy calibration, sampling temperature effects. | **Project 2 Released** |
| **10** | 19-Nov | 20-Nov | **Topic 6: Agentic AI & Tool-Augmented Systems**<br>Autonomous agent architectures: ReAct loops (Reason + Act), Plan-and-Solve, tool execution, environment feedback, memory systems (buffer vs. vector RAG), self-reflection. | `practice/09_agentic_tools.md`<br>Building an autonomous tool-calling agent with self-correction and validation. | Agentic workflows |
| **11** | 26-Nov | 27-Nov | **Topic 7: Alternatives to Autoregression — World Models & JEPA**<br>LeCun's critique of generative prediction; Joint Embedding Predictive Architecture (JEPA); latent space prediction; World Models (Ha & Schmidhuber, Dreamer); model-based planning in latent spaces. | `practice/10_jepa_world_models.md`<br>Training a latent predictive model on simulated world dynamics. | Latent dynamics |
| **12** | 03-Dec | 04-Dec | **Topic 8: Novel Space Exploration — Knee Detection & Empty Space Search**<br>Multi-objective trade-offs and Knee/elbow detection (Pareto fronts, optimal cluster count $k$, regularization cutoffs); Empty Space Search (ESS) for identifying unexplored voids in bounded search domains. | `practice/11_knee_ess.md`<br>Hands-on with `knee` and `emptyspacesearch` Python libraries for automated decision points. | Group research methods |
| **13** | 10-Dec | 11-Dec | **Topic 8: Novel Population Optimization — OBLESA & Optimization**<br>Opposition-Based Learning with Empty Space Search (OBLESA) for intelligent population initialization in metaheuristics (Differential Evolution, PSO); Project 2 clinic. | `practice/12_oblesa_optimization.md`<br>Evaluating OBLESA vs. uniform random initialization in `pyBlindOpt`; Project 2 mentoring. | Project 2 Clinic |
| **14** | 17-Dec | 18-Dec | **Theoretical Test 2 (1h) + Project 2 Presentations (2h)**<br>Final theoretical evaluation covering Topics 6–8, followed by Project 2 live demonstrations, defense, and peer review. | `practice/13_project_showcase.md`<br>Live demonstration, defense, and peer review. | **Theoretical Test 2**<br>**Project 2 Due** |

---

## 4. Repository Structure

Adopted from the streamlined educational template:
* **`slides/`**: Lecture presentations written in Pandoc Markdown, compiled into Beamer PDFs using the `moloch` theme.
* **`practice/`**: 2-hour laboratory guides written in Pandoc Markdown, compiled into clean A4 PDFs.
* **`projects/`**: Specification, guidelines, and rubrics for Project 1 and Project 2.
* **`notebooks/`**: Interactive Jupyter notebooks with runnable code for each practical lab session.
* **`assets/`**: Diagrams, figures, and common graphical resources.
* **`Makefile`**: Master build coordinator featuring parallel execution and a visual progress bar.
* **`Makefile.inc`**: Shared compilation engine caching intermediate LaTeX files in `/dev/shm` RAM disk for maximum compilation speed.
* **`.pre-commit-config.yaml`**: Pre-commit quality gate verifying file formatting, linting Python and Jupyter notebooks with `ruff`, and compiling all course materials.

---

## 5. Recommended Bibliography

* **Andrew Ng**, *Machine Learning Yearning*, 2018. [deeplearning.ai/machine-learning-yearning](https://www.deeplearning.ai/machine-learning-yearning/)
* **David Barber**, *Bayesian Reasoning and Machine Learning*, Cambridge University Press, 2012. [Online Edition](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)
* **Ian Goodfellow, Yoshua Bengio, and Aaron Courville**, *Deep Learning*, MIT Press, 2016. [deeplearningbook.org](https://www.deeplearningbook.org/)
* **Pedro Domingos**, *The Master Algorithm: How the Quest for the Ultimate Learning Machine Will Remake Our World*, Basic Books, 2015.

---

## 6. License & Authors

* **Author:** Mário Antunes (`mario.antunes@ua.pt`)
* **License:** MIT License — see [LICENSE](LICENSE) for details.
