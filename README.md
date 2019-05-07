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
Reading: AIMA Ch. 6.1-6.3, Sutton Ch. 3-4  

3. [Game Play Uncertainty, Expectimax](slides/berkley/lecture7_expectimax_search_and_utilities.pdf)    
Reading: AIMA Ch. 6.1-6.3, Sutton Ch. 3-4.

#### Lab Notebooks:  
- [Problem set 3: Pacman Multi-Agent](labs/pacman_multi_agent/Multi-Agent-Search.ipynb)  

Outcomes addressed in week 4:     
- Demonstrate an understanding of the principles of formal logic including propositional calculus, and first order logic.  
- Conduct proofs of correctness in reasoning systems using the methods of Unification and Resolution.  

---

### Week 5: Markov Decision Processes

#### Lecture:

1. [MDP 1](slides/berkley/lecture8_MDPs_I.pdf)  
Reading: AIMA Ch. 13-17, Sutton Ch. 3-4   

2. [MDP 2](slides/berkley/lecture8_MDPs_II.pdf)  
Reading: AIMA Ch. 13-17, Sutton Ch. 3-4   

3. [Midterm Study Guide](cs4881-midterm-studyguide.pdf)

<!-- 
1. [Acting under uncertainty, probability, joint distributions](slides/berkley/m13-uncertainty.pdf)  
Reading: AIMA Ch. 13   

2. [Naive Bayes](slides/naievbayesx2.pdf)

3. [Bayesian Networks](slides/m14-bayesian.pdf)  *Optional*
- Reading: AIMA Ch. 14   
-->

#### Lab Notebooks:  
- Midterm review   
- [Problem set 3: Pacman Muilti-Agent](labs/pacman_multi_agent/Multi-Agent-Search.ipynb)  *continued*     

Outcomes addressed in week 5:    
- Understand the techniques involved with reasoning in the presence of uncertainty.  

---

#### Week 6: Midterm, Introduction to Reinforcement Learning

#### Lecture:
1. **Midterm**

2. [Reinforcement Learning I](slides/berkley/lecture10_reinforcement_learning_I.pdf) 

#### Lab Notebooks:  
- [Problem set 4: Reinforcement Learning](labs/reinforcement_learning/reinforcement-learning.ipynb) 

Outcomes addressed in week 6:
- Understand the concepts of an intelligent agent and their environment.
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.    

---

### Week 7: Reinforcement Learning, Value Functions, Intro to Machine Learning, Supervised Learning

1. [Reinforcement Learning II](slides/berkley/lecture11_reinforcement_learning_II.pdf) 

2. [Introduction to Machine Learning with KNN](https://github.com/jayurbain/machine-learning/blob/master/slides/06_machine_learning_knn.pdf)  *review on your own *
- Reading: ISLR Ch. 4.6.5  

3. [Linear Regression 1](https://github.com/jayurbain/machine-learning/blob/master/slides/08_linear_regression.pdf)
- Reading: PDSH Ch. 5 p. 331-375, 390-399  
- Reading: ISLR Ch. 1, 2 

4. [Logistic Regression Classification](https://github.com/jayurbain/machine-learning/blob/master/slides/09_logistic_regression_classification.pdf)  
- Reading: ISLR Ch. 4  

#### Lab Notebooks:
<!--  
- [Jupyter Notebooks](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/lab_0_python/lab_0_jupyter.ipynb)
- [Python Machine Learning Environment](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/lab_0_python/python_programming_style.ipynb)
- [Clustering](https://github.com/jayurbain/machine-learning/blob/master/notebooks/Clustering.ipynb)  *Submission required*
-->
- Problem set 4: Reinforcement Learning continued   
- [Linear Regression Notebook](https://github.com/jayurbain/machine-learning/blob/master/notebooks/08_linear_regression.ipynb) *Tutorial: Submission required*   

References:
- [Gradient Descent](slides/LogisticRegressionML_Jay.pdf)   
- [Gradient Descent notebook](notebooks/GradientDescent.ipynb)  
- [Reinforcement Learning extra slides](slides/reinforcementlearning.pdf)   
- [Q Learning Spreadsheet](http://jayurbain.com/msoe/cs4881/RL.xls) 

<!--
- [Supervised Learning - Linear Regression](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/Lab3_LinearRegression/Supervised%20Learning%20-%20%20Linear%20Regression.ipynb)   *Submission required 2-weeks*   
- [Supervised Learning - Logistic Regression](https://github.com/jayurbain/DataScienceIntro/blob/master/labs/Lab5_Logistic_Regression/Supervised%20Learning%20-%20Logistic%20Regression.ipynb)  *Submission required 2-weeks*   
-->

Outcomes addressed in week 7:    
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.   

---

#### Week 8: Macine Learning - Deep Learning   

#### Lecture: 
1. [Deep Learning Introduction 1](slides/Deep&#32;Learning&#32;Introduction.pdf)  
Reference for earlier in class:    

2. [Deep Learning Introduction 2](slides/Deep&#32;Learning&#32;Introduction.pdf) *Optional*   

3. [Backpropagation](slides/backpropagation.pdf) *Optional*       

[YouTube: Deep Learning Revolution](https://www.youtube.com/watch?v=Dy0hJWltsyE)

#### Lab Notebooks:  
- [Multinomial Image Classification](https://github.com/jayurbain/machine-learning/blob/master/notebooks/multinomial_classification.ipynb) *submission required*     
-- If you're having trouble reading MNIST from mldata use the following notebook to load the data:  
-- [Write Read MNIST](https://github.com/jayurbain/machine-learning/blob/master/notebooks/write_read_MNIST.ipynb)   
-- [mnist_data.csv](https://drive.google.com/open?id=1p-K8JpCEiATE7VrBNsfruYtn4ivorceW)  
-- [mnist_target.csv](https://drive.google.com/open?id=1TnARZEzS0CPhco_MxpkVbf31zYqkuFbL) 
<!--
- [Image Classification](https://github.com/jayurbain/machine-learning/blob/master/notebooks/computer_vision/cnn_cifar10.ipynb) *Submission required*   
-->

Complete the following Get Started with TensorFlow tutorials. *Optional, but recommended*   

https://www.tensorflow.org/tutorials/  Train your first neural network: basic classification  
https://www.tensorflow.org/tutorials/keras/basic_classification  Explore overfitting and underfitting  
https://www.tensorflow.org/tutorials/keras/overfit_and_underfit  

Outcomes addressed in week 8:
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.   

---

#### Week 9: Convnets, Deep RL  

#### Lecture:

1. [Convolutional Neural Networks](https://github.com/jayurbain/machine-learning/blob/master/slides/cnn_1.pdf)  

2. [Visualizing what ConvNets Learn](http://localhost:8888/notebooks/machine-learning/machine-learning/notebooks/dlp/visualizing-what-convnets-learn.ipynb)  

3. [Deep Q-Learning]() *available Thursday*   


Reading:  
- DLP Ch. 5 

#### Lab Notebooks:  
- [RL with OpenAI Gym](deep_rl/reinforcement-learning-with-openai-gym.ipynb)  *Submission required.*   
Use the following link https://github.com/jayurbain/artificial-intelligence/blob/master/deep_rl/reinforcement-learning-with-openai-gym.ipynb on Google Collab https://colab.research.google.com/notebooks/welcome.ipynb.   
- Complete the [OpenAI Gym Tutorial](https://gym.openai.com/docs)  *Optional*
- [Keras Intro](https://github.com/jayurbain/machine-learning/blob/master/notebooks/deep_learning_intro/Keras-task.ipynb) *Submission required*   
- [Deep Q-Learning with OpenAI Gym]()  *Submission required - available Thursday*

Outcomes addressed in week 9:
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.       

#### Week 10: Advanced Topics

#### Lecture:

<!--
1. [Deep Learning for Computer Vision](https://github.com/jayurbain/machine-learning/blob/master/slides/dli/Lecture-3-1-convnets-history-urbain.pdf)   

2. [Convnets](https://github.com/jayurbain/machine-learning/blob/master/slides/dli/Lecture-3-2-convnets-intro-urbain.pdf)  *optional*

3. [Deep Reinforcement Learning](slides/RL_deep.pdf)  

#### Lab Notebooks:

- [Deep Q Learning](labs/Deep_Reinforcement_Learning/DQN.ipynb) *Submission required*
-->
Outcomes addressed in week 10:
- Understand the concepts of an intelligent agent and their environment.
- Be able to address problems related to search, and its application to intelligent systems, including: game playing, decision making, and adversarial search.  
- Understand the techniques involved with reasoning in the presence of uncertainty.  
- Understand and apply modern machine learning techniques for supervised, unsupervised, and reinforcement learning.     

---

#### Optional material: Deep Learning for NLP

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

#### Week 11: Final Exam

Wednesday 11:00 AM - 1:00 PM S243