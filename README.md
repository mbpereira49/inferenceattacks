# inferenceattacks

## Ideas
- In "On Inferring Training Data Attributes in Machine Learning Models", they get class labels with k-means clustering. But that's going to make the labels very smooth, i.e. there's no randomness or "unexpected" class labels. Because MIA/AIA techniques rely on confidence of the model's output, those "unexpected" labels might be harder to infer in an attack. So, let's try to **introduce randomness in the k-means labelling of the data**.
- Extend to entropy along with maximum confidence

## To-do
Follow the description in section Setup of the paper.
All the following will go in Classifier.ipynb:
### Pre-processing
- k-means clustering for Purchase
- train-test split
### Target models
- PyTorch for 5-layer network
### Attack models
- Single-layer
### Evaluation
- Figure out how to compute ROC and then AUC
### Methods for synthetic data generation
