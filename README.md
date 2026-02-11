# BERT_Fine_Tuning
Fine-tuned Google’s BERT base uncased model on the NYT dataset with an A100 GPU on Colab for training to categorize news articles into 3 imbalanced classes (business, sports, and politics) with a validation f1 score of 98%. Also implemented Word2Vec, GloVe, and other vector embedding algorithms.

# Summary of results 

I had initially started off by splitting the data into the training, validation and test sets without stratification, but then the macro f1 scores in Q1 seemed off. That's when it occurred to me that the dataset might be imbalanced. Upon verifying this, I stratified the splits to get significant representations from each class. 

In Q1, the tf-idf vector performed the best with a macro f1 score of 0.9798, as compared to binary vector's 0.9690 and frequency vector's 0.9724. I chose the macro f1 score as the statistic for ranking the models, as it prioritises the underrepresented classes as much as the significantly represented ones, due to its unweighted averaging. 

In Q2, the model trained on GloVe embedding vectors performed better (f1 macro = 0.9706) than the model trained on Word2Vec vectors (f1 macro = 0.9594). This is likely due to the vast difference in the sizes of the training data for the two models. GloVe was trained on Wikipedia + Gigaword corpus, whereas our Word2Vec model was trained only on our limited training data, thus limiting it from achieving as rich word representations as the GloVe model.

In Q3, I printed the performance stats of the BERT-based classifier for each epoch, and saved the best performing model for testing. It is interesting to note that while the total loss for the model decreases in each epoch, the macro f1 score worsens instead of improving in the third epoch (0.9528 to 0.9392). This is likely due to overfitting of the model on the training set. The best performing model (which is from the seond epoch) yields a macro f1 score of 0.9634 on the test set.

Overall performance ranking: 

1. TF-IDF (0.9798)

2. frequency vector (0.9724)

3. GloVe (0.9706)

4. binary vector (0.9690)

5. BERT fine-tuned (0.9634)

6. Word2Vec (0.9594)
