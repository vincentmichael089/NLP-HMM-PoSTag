# Hidden Markov Model - Viterbi Algorithm implementation for Part-of-Speech Tagging

As per the title, this is an HMM model with Viterbi Algorithm which aims to perform Part-of-Speech Tagging. This model is built from scratch, so it may not be optimal and has some bugs (which I am not aware of).

## Requirements
1. &nbsp;Pandas
2. &nbsp;Swifter
3. &nbsp;NumPy

## Functions
<div>
  <b> 1. Train model: </b>
  To train the HMM model, the fit() function is called from the HMM class. the fit() function accepts training data in pandas dataframe format as shown below. From the picture below, it can also be seen that the training data has a sequence of {word}_{part-of-speech} format.
</div>
<br>
<img src="https://github.com/vincentmichael089/NLP-HMM-PoSTag/blob/main/asset/disp-01.png" width="600" />
<br>
<div>
  <b> 2. Predict: </b>
  predicting data can be executed by calling the predict() function, which takes 4 inputs namely:
  
  > 1.	data: can be a dataframe or a single sentence.
  > 2.  dataFrame (default = True): must be changed to False when the data is in the form of a single sentence. Returns accuracy when value is True.
  > 3.	printStep (default = False) : determines whether to print Viterbi algorithm steps.
  > 4.	getResult (default = False): returns a dataframe containing the predicted result.

  The following is an example of using the predict() function and predicting the results of testing data. Also attached is the use of the predict() function when the input is a single sentence and when returning the steps of the Viterbi algorithm.

</div>
<br>
<img src="https://github.com/vincentmichael089/NLP-HMM-PoSTag/blob/main/asset/disp-02.png" width="600" />
<br>

<br>
<img src="https://github.com/vincentmichael089/NLP-HMM-PoSTag/blob/main/asset/disp-03.png" width="600" />
<br>

<div>
  <b> Please go to HMM_example.ipynb for more details. </b>
</div>

<hr>