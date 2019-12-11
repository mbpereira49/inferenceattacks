# inferenceattacks

## Ideas
- In "On Inferring Training Data Attributes in Machine Learning Models", they get class labels with k-means clustering. But that's going to make the labels very smooth, i.e. there's no randomness or "unexpected" class labels. Because MIA/AIA techniques rely on confidence of the model's output, those "unexpected" labels might be harder to infer in an attack. So, let's try to **introduce randomness in the k-means labelling of the data**.
- Extend to entropy along with maximum confidence

## To-do

### Methods for synthetic data generation

### Different classifiers
- CNN : find appropriate basic image set. Euclidean metric
- Random forest: should we do a different dataset?

### https://arxiv.org/pdf/1912.02292.pdf
- Follow architecture to look for double descent
- Explore MIA effectiveness starting at top of descent
