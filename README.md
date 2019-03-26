----

### Artificial Intelligence

The objective of this course is to introduce the basic concepts of artificially intelligent systems. Topics covered include knowledge representation, search strategies and machine learning. An emphasis is placed on developing intelligent agents that can interact with their environment using machine learning techniques. Modern machine learning techniques for supervised, unsupervised, and reinforcement learning are introduced. The role of AI in engineering and computing systems is presented. Students complete exercises that allow them to apply AI in diverse problem settings including search, constraint satisfaction, game play, navigation, and natural language processing.

Prerequisites: MA-262 Probability and Statistics; programming maturity, and the ability to program in Python.  

Helpful: CS3851 Algorithms, MA-383 Linear Algebra.  

ABET: Math/Science, Engineering Topics.

2-2-3 (class hours/week, laboratory hours/week, credits)

Lectures are augmented with hands-on tutorials using Jupyter Notebooks. Laboratory assignments will be completed using Python and related data science and machine learning libraries: NumPy, Pandas, Scikit-learn, Matplotlib, TensorFlow, Keras, PyTorch.

Outcomes:   
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Demonstrate an understanding of the principles of formal logic including propositional calculus, and first order logic.  
- Conduct proofs of correctness in reasoning systems using the methods of Unification and Resolution.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.   
- Understand the limitations and future challenges of AI.

References:  

Artificial Intelligence, A Modern Approach. Stuart Russell and Peter Norvig. Third Edition. Pearson Education. Resources:  http://aima.cs.berkeley.edu/

Reinforcement Learning: An Introduction Richard S. Sutton and Andrew G. Barto. Second Edition, in progress. MIT Press, Cambridge, MA, 2017.
https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view

[Deep Learning with Python, François Chollet. Manning, 2017.](https://www.manning.com/books/deep-learning-with-python)

[Python Data Science Handbook, Jake VanderPlas, O'Reilly.](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

### Week 1: Introduction to AI, AI History, Intelligent Agents

#### Lecture:    
0. [Syllabus](syllabus.pdf)  

1. [Introduction to Artificial Intelligence](slides/m1-intro.pdf)   
AI Demonstrations    
Reading: AIMA: Chapter 1   


2. [Intelligent Agents](slides/m2-agents.pdf)  
Reading: AIMA: Chapter 2  

#### Lab:
- [Juputer Notebook Tutorial](https://github.com/jayurbain/machine-learning/blob/master/notebooks/lab_0_python/lab_0_jupyter.ipynb) *Optional*    
- [Problem set 1](labs/CS4881-problemset1.pdf) *Submission required*
  
- [Python Programming](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/lab_0_python/lab_0_python.ipynb)  *Submission required*  

- Reference: [AI Grand Challenges](http://www.engineeringchallenges.org/challenges.aspx)  

Outcomes addressed in week 1:   
- Understand the concepts of an intelligent agent and their environment.  

---

### Week 2: Search

#### Lecture:   

1. [Uninformed Search](slides/m3-search.pdf)   
Reading: AIMA: Chapter 3.1-3.5     

2. [A* Search and heuristic functions](slides/m4-heuristics.pdf)    
Reading: AIMA: Chapter 4  

#### Lab Notebooks:
- [Problem set 2](labs/CS4881-problemset2.pdf)  *2-week lab*

Outcomes addressed in week 2:   
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.   

---

### Week 3: Constraint Satisfaction Problems, Games  

#### Lecture:   

1. [Online Search](slides/m4.5-onlinesearch.pdf)  **postpone**  
Reading: AIMA Ch. 4.5  

#### Lab Notebooks:
- [Problem set 2: Logical Agents](labs/CS4881-problemset2.pdf)  *continued*

Outcomes addressed in week 3:    
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  

---

#### Week 4: Logic and Inference   

#### Lecture:
<!--
1. [Logical Agents, Propositional Logic](slides/m7-logic.pdf)  
Reading: AIMA Ch. 7  
[New chapter 7](http://jayurbain.com/msoe/cs4881/newchap07.pdf)  
[Forward chaining, backward chaining, inference](slides/Propositional%20Theorem%20Prooving%20-%20Notes.pdf)    

2. [First Order Logic](slides/m8-fol.pdf)      
Reading: AIMA Ch. 8           

3. [First Order Inference](slides/m8-fol.pdf)      
Reading: AIMA Ch. 9.4    
[Logical Agents Review Notes](http://jayurbain.com/msoe/cs4881/logicalagentsreviewnotes.pdf)
-->

1. [Constraint Satisfaction](slides/berkley/lecture4_CSPs.pdf)  
<!-- 1. [Constraint Satisfaction](slides/m5-csp.pdf)  -->
Reading: AIMA Ch. 5  

2. [Game Play, Adversarial Search, Minimax](slides/berkley/m6-game.pdf)  
Reading: AIMA Ch. 6.1-6.3  

3. [Game Play Uncertainty]()  *not yet posted*   
Reading: AIMA Ch. 6.1-6.3  

#### Lab Notebooks:  
- [Problem set 3: Pacman Multi-Agent](labs/pacman_multi_agent/Multi-Agent-Search.ipynb)  

Outcomes addressed in week 4:     
- Demonstrate an understanding of the principles of formal logic including propositional calculus, and first order logic.  
- Conduct proofs of correctness in reasoning systems using the methods of Unification and Resolution.  

---

### Week 5: Uncertainty

#### Lecture:

1. [Acting under uncertainty, probability, joint distributions](slides/m13-uncertainty.pdf)  
Reading: AIMA Ch. 13   

2. [Naive Bayes](slides/naievbayesx2.pdf)

3. [Bayesian Networks](slides/m14-bayesian.pdf)  *Optional*
- Reading: AIMA Ch. 14   

#### Lab Notebooks:  
- [Problem set 3: Pacman Muilti-Agent](labs/pacman_multi_agent/Multi-Agent-Search.ipynb)  *continued*     

Outcomes addressed in week 5:    
- Understand the techniques involved with reasoning in the presence of uncertainty.  

---

#### Week 6: Midterm, Introduction to Machine Learning

#### Lecture:
1. **Midterm**

2. [Introduction to Machine Learning](https://github.com/jayurbain/machine-learning/blob/master/slides/IntroMachineLearning.pdf)   
ML Demonstrations  

#### Lab Notebooks:   
- [Jupyter Notebooks](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/lab_0_python/lab_0_jupyter.ipynb)
- [Python Machine Learning Environment](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/lab_0_python/python_programming_style.ipynb)
- [Clustering](https://github.com/jayurbain/machine-learning/blob/master/notebooks/Clustering.ipynb)  *Submission required*

Outcomes addressed in week 6:
- Understand the concepts of an intelligent agent and their environment.
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.    

---

### Week 7: Linear Regression, Logistic Regression

1. [Linear Regression](https://github.com/jayurbain/machine-learning/blob/master/slides/LinearRegressionML_Jay.pdf)  

2. [Logistic Regression](https://github.com/jayurbain/machine-learning/blob/master/slides/LogisticRegressionML_Jay.pdf)

#### Lab Notebooks:

- [Supervised Learning - Linear Regression](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/Lab3_LinearRegression/Supervised%20Learning%20-%20%20Linear%20Regression.ipynb)   *Submission required 2-weeks*   
- [Supervised Learning - Logistic Regression](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/Lab5_Logistic_Regression/Supervised%20Learning%20-%20Logistic%20Regression.ipynb)  *Submission required 2-weeks*   

*Note: need to reduce quantity of work in this week*  

Outcomes addressed in week 1:    
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.   

---

#### Week 8: Reinforcement Learning  

#### Lecture:

1. [Reinforcement Learning 1](slides/rl_intro.pdf)   
Reading: Reinforcement Learning: An Introduction (RL) - Ch. 1   

2. [Reinforcement Learning 2](slides/rl_intro_2.pdf)  
Reading: Reinforcement Learning: An Introduction (RL) - Ch. 2  

#### Lab Notebooks:  
- Complete the [OpenAI Gym Tutorial](https://gym.openai.com/docs)  *Submission required*

References:
- [Reinforcement Learning extra slides](slides/reinforcementlearning.pdf)   
- [Q Learning Spreadsheet](http://jayurbain.com/msoe/cs4881/RL.xls)  

Outcomes addressed in week 7:
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.   

---

#### Week 8: Deep Learning  

#### Lecture:

1. [Deep Learning Introduction 1](https://github.com/jayurbain/machine-learning/blob/master/slides/Deep%20Learning%20Introduction.pdf)

2. [Deep Learning Introduction 2](https://github.com/jayurbain/machine-learning/blob/master/slides/dli/Lecture-2-1-dl-intro-urbain.pdf)

3. [Backpropagation](https://github.com/jayurbain/machine-learning/blob/master/slides/dli/Lecture-2-3-dl-backprop2-urbain.pdf)   *Optional*

[YouTube: Deep Learning Revolution](https://www.youtube.com/watch?v=Dy0hJWltsyE)

#### Lab Notebooks:  
- [Keras Intro](https://github.com/jayurbain/machine-learning/blob/master/notebooks/deep_learning_intro/Keras-task.ipynb) *Submission required*   
- [Image Classification](https://github.com/jayurbain/machine-learning/blob/master/notebooks/computer_vision/cnn_cifar10.ipynb) *Submission required*   

Outcomes addressed in week 8:
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.       

#### Week 9: Deep Reinforcement Learning

#### Lecture:

1. [Deep Learning for Computer Vision](https://github.com/jayurbain/machine-learning/blob/master/slides/dli/Lecture-3-1-convnets-history-urbain.pdf)   

2. [Convnets](https://github.com/jayurbain/machine-learning/blob/master/slides/dli/Lecture-3-2-convnets-intro-urbain.pdf)  *optional*

3. [Deep Reinforcement Learning](slides/RL_deep.pdf)  

#### Lab Notebooks:
- [Deep Q Learning](labs/Deep_Reinforcement_Learning/DQN.ipynb) *Submission required*

Outcomes addressed in week 9:
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.     

---

#### Week 10: Deep Learning for NLP

#### Lecture:

1. [NLP Classification](https://github.com/jayurbain/machine-learning/blob/master/slides/nlp/2%20NLP%20Text%20Classification.pdf)  *optional*  

2. [NLP Translation](https://github.com/jayurbain/machine-learning/blob/master/slides/nlp/3%20NLP%20Text%20Translations.pdf)  *optional*   

3. [Artificial General Intelligence (MIT)](slides/lecture_2018_01_22.pdf)

4. [Artificial General Intelligence (Machine Learning Summer School)](slides/AGILecture.pdf)

#### Lab Notebooks:     
- [NLP Classification](notebooks/nlp/Text&#32;Classification/SentimentClassification.ipynb)   *optional*
- [NLP Translation ](https://github.com/jayurbain/machine-learning/blob/master/slides/nlp/3%20NLP%20Text%20Translations.pdf)   *optional*

*Note: need to prune answers from notebooks*   

Outcomes addressed in week 10:   
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.    

---
