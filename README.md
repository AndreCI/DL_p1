# DL_p1

Repo for the first mini project of the deep learning course.

The goal of this project is to implement a neural network to predict the laterality of finger movement (left or right) from the EEG recording. This is a standard two-class classification problem.

### TODO
- [ ] data analysis
    - [x] data handler
    - [ ] data visualisation
    - [ ] data checking
    - [ ] PCA?
    - [ ] data preparation (train, val, dev)

- [ ] Models Implementations
    - [ ] simple first model (linear predictor)
    - [ ] brainstorm others ideas

- [ ] Testing & others
    - [ ] loss evolutions curves
    - [ ] logging


### Brainstorming
Data set is clearly time dependant.
Data set is composed of 316 examples of size 28 * 50 ?

### Important
The expected error, if classification is random, is 50%

There are a lot of possible pitfalls in processing the test data which may lead to bad or even "random" results.
references:
http://www.bbci.de/competition/ii/results/