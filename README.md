# inferenceattacks

## Ideas
- In "On Inferring Training Data Attributes in Machine Learning Models", they get class labels with k-means clustering. But that's going to make the labels very smooth, i.e. there's no randomness or "unexpected" class labels. Because MIA/AIA techniques rely on confidence of the model's output, those "unexpected" labels might be harder to infer in an attack. So, let's try to **introduce randomness in the k-means labelling of the data**.
- Extend to entropy along with maximum confidence

## To-do

### Tasks
- Jordan: implement CNN in pytorch as done in https://arxiv.org/pdf/1912.02292.pdf with param for modifying model width. Verify it works in our pipeline with CIFAR10 and get a sense for how long it takes to run.
- Alex: Extend pipeline to work with multiple trials
- Rithvik: Jaccard distance for comparing images & Figure out how to add a variable amount of noise to k-means labels and then run some preliminary experiments


### Methods for synthetic data generation

### Different classifiers
- CNN : find appropriate basic image set. Euclidean metric
  - if we're doing Jaccard distance, should we only compare it with "real" images or should we make noisy images that might not have semantic meaning
- Random forest: should we do a different dataset?

### https://arxiv.org/pdf/1912.02292.pdf
- Follow architecture to look for double descent
- Explore MIA effectiveness starting at top of descent

### Add noise to labels
