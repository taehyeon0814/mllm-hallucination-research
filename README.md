# MLLM Hallucination Research and Benchmarks

A curated list of research papers and benchmarks for Multimodal Large Language Models (MLLMs) Hallucination.

## Research Papers

### 1. OPERA: Over-Trust Penalty and Retrospection-Allocation
- **Motivation:** MLLMs tend to cause hallucinations by over-relying on specific summary tokens rather than looking at the entire image.
- **Method:**
    - **Over-Trust Penalty:** Imposes a penalty to reduce excessive reliance on specific tokens during beam search decoding.
    - **Retrospection-Allocation:** A rollback strategy that reviews whether summary tokens are present in the generated tokens and re-adjusts token selection if necessary.
- **Analysis:** Demonstrates that hallucinations can be mitigated by decoding strategy alone without additional data or training, proving its effectiveness in various MLLMs.
- **Contribution:** Proposes a new decoding method applicable without training, offering an efficient solution to the hallucination problem.

### 2. Data-augmented Phrase-level Alignment (DPA)
- **Motivation:** Focuses on the 'object hallucination' problem, where information about an object not present in the image is generated.
- **Method:**
    - **Data Augmentation:** Generates 'hallucinated' and 'correct' response pairs by intentionally altering parts of the ground-truth text.
    - **DPA Loss:** Applies a new loss function that trains the model to lower the probability of hallucinatory phrases compared to correct ones.
- **Analysis:** A model fine-tuned with DPA (HALVA) significantly improved F1 scores in hallucination-related VQA and reduced hallucination rates in image captioning.
- **Contribution:** Proposes a method to mitigate hallucinations in existing MLLMs and maintain general vision-language capabilities through data augmentation and phrase-level alignment.

### 3. ConVis: Contrastive Decoding with Hallucination Visualization
- **Motivation:** Aims to solve the hallucination problem where the generated response does not accurately reflect the given image.
- **Method:**
    - **Contrastive Decoding:** Uses a T2I (Text-to-Image) model to reconstruct an image from a hallucinatory caption and suppresses hallucinations by comparing the probability distributions of the original and reconstructed images.
- **Analysis:** Proved through 5 benchmarks that it can effectively reduce hallucinations across various MLLMs with only the decoding process, without additional data or model updates.
- **Contribution:** Proposes a new training-free contrastive decoding method using a T2I model to enhance model reliability.

### 4. Woodpecker: Hallucination Correction for Multimodal Large Language Models
- **Motivation:** While most research focuses on detecting or mitigating hallucinations, there was a lack of research on 'correcting' hallucinations in already generated results.
- **Method:** Proposes a post-hoc correction framework called 'Woodpecker'. It operates in 5 steps: (1) Decompose the generated text into key claim units, (2) Generate questions to verify each claim, (3) Use a VQA model to answer the questions based on the image, (4) Check for discrepancies between the original claim and the VQA answer, (5) Correct the discrepant claims.
- **Analysis:** Effectively corrects various types of hallucinations, including objects, attributes, and relations, and shows higher performance than existing methods on several benchmarks.
- **Contribution:** Presents a new paradigm for verifying MLLM outputs and automatically correcting hallucinations without training.

### 5. Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating Visual Hallucination (VIC)
- **Motivation:** The existing CoT (Chain-of-Thought) method can cause hallucinations due to misleading image information by processing visual information and reasoning simultaneously.
- **Method:** Proposes the 'Visual Inference Chain (VIC)' framework. It first generates a reasoning chain with only the text context and then introduces visual information to derive the final answer. This separates text-based logical reasoning from visual perception.
- **Analysis:** VIC significantly improved zero-shot performance across various vision-language tasks by reducing cross-modal biases.
- **Contribution:** Proposes a new framework that mitigates hallucinations and improves the reasoning accuracy of MLLMs by separating the reasoning process from visual input.

### 6. Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation (MemVR)
- **Motivation:** Hallucinations in MLLMs often stem from the text decoder 'forgetting' visual information (amnesia).
- **Method:** Proposes a new decoding paradigm called 'MemVR'. When the model shows high uncertainty during inference, it re-injects visual tokens into the Feed Forward Network (FFN) as 'key-value memory'. This mimics the human behavior of 'looking twice' to re-check something.
- **Analysis:** Showed a significant reduction in hallucinations across various MLLMs without additional inference time.
- **Contribution:** Improves the model's fact-based response capability through a tuning-free decoding method called 'MemVR' that mimics human cognitive processes.

### 7. VCD: A General Decoding Strategy for Mitigating Hallucination in MLLMs
- **Motivation:** There is a need for a general, training-free decoding strategy to mitigate hallucinations that can be universally applied to various MLLMs.
- **Method:** Proposes 'Visual Contrastive Decoding (VCD)'. It contrasts the output probability distribution of the original MLLM with that of a 'degraded' model without visual information. This difference is used to penalize text generation that is not based on visual information.
- **Analysis:** Effectively reduced hallucinations in a number of MLLMs on benchmarks like POPE and MME.
- **Contribution:** Presents a universal decoding strategy (VCD) that mitigates hallucinations without training by utilizing visual contrast.

### 8. Mitigating Object-Related Hallucination in Large Vision-Language Models (LURE-Correct)
- **Motivation:** 'Object-related hallucinations', such as mentioning non-existent objects or misdescribing object attributes, are a major problem in MLLMs.
- **Method:** Proposes a fine-tuning method called 'LURE-Correct'. It automatically builds a dataset from image-caption data to train the model to be based on visual evidence. The training proceeds by contrasting correct descriptions with those containing hallucinations.
- **Analysis:** A model fine-tuned with LURE-Correct significantly reduced object-related hallucinations without degrading general performance.
- **Contribution:** Proposes an effective and automated fine-tuning method specialized for reducing object-related hallucinations.

### 9. Object-level Hallucination in Vision-Language Models (POPE)
- **Motivation:** There was a need to systematically study and quantify the problem of object-level hallucinations.
- **Method:** Proposed the 'POPE' benchmark, which is specialized for measuring object hallucinations. It consists of simple "yes/no" questions about the presence of a specific object in an image.
- **Analysis:** Analyzed several state-of-the-art VLMs (Vision-Language Models) and showed that most of them frequently produce object hallucinations. It particularly found a trade-off between the ability to follow instructions well and the occurrence of hallucinations.
- **Contribution:** A pioneering study that presented a basic analysis of the object hallucination problem and the POPE benchmark, which is still widely used today.

### 10. LURE: A Benchmark for Hallucination Evaluation and Reasoning in Large Vision-Language Models
- **Motivation:** Existing benchmarks focus only on hallucinations themselves and fail to evaluate the complex reasoning processes that cause them.
- **Method:** Proposes the 'LURE' benchmark. It consists of a 'Diagnosing' set to diagnose various types of hallucinations and a 'Reasoning' set that requires complex multi-step visual reasoning where hallucinations can easily occur.
- **Analysis:** Showed that even state-of-the-art models like GPT-4V struggle with the reasoning tasks in LURE, revealing a deep connection between reasoning failure and hallucinations.
- **Contribution:** Provides a new benchmark (LURE) that allows for a deeper analysis of the failure causes of MLLMs by evaluating hallucinations and complex reasoning abilities together.

### 11. HaELM: Hallucination-aware Ensemble of Experts for Mitigating Hallucinations in MLLMs
- **Motivation:** Having a single MLLM handle all kinds of tasks can cause hallucinations, so a specialized approach to tasks is needed.
- **Method:** Proposes 'HaELM' based on the Mixture of Experts (MoE) architecture. A 'hallucination-aware router' determines the type of question and sends the task to a specialized 'expert' model for factual description or another 'expert' model for creative tasks.
- **Analysis:** HaELM reduced hallucinations compared to general models while maintaining high performance in other tasks.
- **Contribution:** Proposes a new model architecture that mitigates hallucinations by routing tasks to specialized 'experts' using the MoE architecture.

### 12. Hallucination-in-Context: A New Benchmark and A Simple Method (HIC)
- **Motivation:** Existing benchmarks evaluate hallucinations in a single question-answer pair, but not in a multi-turn conversational context.
- **Method:** Proposes the 'HIC' benchmark for evaluating hallucinations in a multi-turn conversational context. It also presents a simple mitigation method using a 'visual-aware prompt' that explicitly instructs to "answer based on the image".
- **Analysis:** Showed that MLLMs are more likely to hallucinate in a conversational context and that the proposed prompting method can effectively reduce this.
- **Contribution:** Presents a new benchmark (HIC) for evaluating hallucinations in a conversational context and a simple yet effective prompting mitigation strategy.

### 13. GAVIE: A General and Efficient Tuning-Free Approach for Vision-Language Instruction Following
- **Motivation:** An efficient approach is needed to solve the problem of MLLMs not faithfully following visual instructions and producing hallucinations.
- **Method:** Proposes a tuning-free method called 'GAVIE'. It uses a 'Grounded Attention' mechanism that dynamically adjusts the contribution between internal knowledge (LLM) and external knowledge (image). This encourages focusing more on visual evidence during text generation.
- **Analysis:** GAVIE significantly improved the model's faithfulness, reducing hallucinations and enabling it to follow various instructions more accurately.
- **Contribution:** Proposes a general and efficient method to improve the visual instruction following ability of MLLMs and mitigate hallucinations without training by controlling the attention mechanism.

## Benchmarks

### 1. POPE (Polling-based Object Probing Evaluation)
- **Composition:** Consists of questions that ask for a "yes/no" answer about the presence of a specific object in an image. Hallucinations are mainly measured by False Positives, where the model incorrectly answers "yes" for an object that is not in the image.
- **Evaluation Method:** The model's answers are compared with the ground truth to calculate accuracy, precision, recall, F1 score, etc. It is particularly effective for measuring object hallucinations.
- **Dataset Location:** [https://github.com/A-Lin-X/POPE](https://github.com/A-Lin-X/POPE)

### 2. MME (Multimodal Model Evaluation)
- **Composition:** A comprehensive benchmark consisting of 14 sub-tasks that evaluate both perception and cognition abilities. Hallucinations are mainly evaluated in tasks such as 'existence', 'count', and 'position'.
- **Evaluation Method:** Accuracy is measured for each task to identify the model's overall performance and hallucination tendencies in specific areas.
- **Dataset Location:** [https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/master/MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/master/MME)

### 3. AMBER (An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation)
- **Composition:** Includes both generative and discriminative tasks and is designed to evaluate multi-dimensional hallucinations such as existence, attribute, and relation hallucinations.
- **Evaluation Method:** Provides a low-cost and efficient evaluation pipeline that can be used without an external LLM like GPT-4.
- **Dataset Location:** [https://github.com/junyangwang0410/AMBER](https://github.com/junyangwang0410/AMBER)