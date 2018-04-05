# DL_p1

Repo for the first mini project of the deep learning course.

The goal of this project is to implement a neural network to predict the laterality of finger movement (left or right) from the EEG recording. This is a standard two-class classification problem.

### TODO
- [ ] data analysis
    - [x] data handler
    - [x] data visualisation
    - [ ] data checking
    - [ ] PCA?
    - [ ] data preparation (train, val, dev)

- [ ] Models Implementations
    - [ ] simple first model (linear predictor)
    - [ ] brainstorm others ideas
        - [ ] channel level/time step level CNN
        - [ ] LSTMs with attention?
        - [ ] adding dense layer on one of the axis (input reprensentation)
        - [ ] More simple stuff: SVM?
        - [ ] Deep FFNN with lot of dropout


- [ ] Testing & others
    - [ ] loss evolutions curves
    - [ ] logging


### Brainstorming
Data set is composed of 316 examples of size 28 * 50. Each example has 28 channels, sample during 0.5 sec.
The data set is thus time dependant, which is hard for classic neural net. This means that the solution should probably be able to handle time dependency, which is the reason why LSTMS or time level CNNs could be good.
The dataset is really small.

Main issue: dataset size. ~300ish data examples is not enough..

-Q about dataset: is it really that small? How can we use deep learning? What is it actuall? Laterity of finger mvt?
Can we use winners from the BCI competition as inspiration?
In what format do you want the report? Paper style?

### Important
The expected error, if classification is random, is 50%

There are a lot of possible pitfalls in processing the test data which may lead to bad or even "random" results.
references:
http://www.bbci.de/competition/ii/results/

## Journal

#### 3/04
After looking at the dataset, I decided to first implement a simple model (as asked) to see how it succeds

I also want to implement a simple pipeline to automatically take cares of the train and test steps. I will probably design first or in conionction to the simple model an abstract class and the architecture to handle these steps.

Finally, I may also want a way to save networks configuration. This could be done by allowing the user to store and load .xml files containing diverses infos on the networks.

I'll start by thinking of the way the project should be constructed.

#### 05/04
I start working on a first simple model. From this, I plan to developp the pipeline for futur models.

First model implemented. The architecture is not nice, but the idea is to have a somewhat working model to build around. The next step will be focused on rearrange this model to allow the construction of an abstractive class and to refine the pipeline of events. By the way, the model is as expected terrible: it does not perform better than random chance.