# text-generator
Text generation with Recurrent Neural Networks. We train a Recurrent Neural Network with vocabulary (source TBD) so that we obtain a generative language model. This model will be publicly exposed as an API, deployed on a cloud service provider (cloud provider TBD). Once deployed, users can input a single word and the model will generate some (hopefully funny) sentence with it.

## Things to-do 

* ~~Decide on the vocabulary to use.~~ Chosen vocab: 
Clickbait titles from The Examiner (extracted from https://www.kaggle.com/datasets/therohk/examine-the-examiner?resource=download)
* Decide on cloud provider to deploy model.
* Code 
  * Build training module
  * Build inference module 
