# What Is Learned, When? | EleutherAI Community Project

reference thesis: https://nsaphra.net/uploads/thesis.pdf

older work (e.g. that thesis) uses lstm, is unclear if any of that transfers to a modern architecture, pythia checkpoints give a good reference point

## Goals for the project
As per JDC has mentioned:
1. Establish things that pythia-12b embeddings have learned when fully trained.
1. Look through the checkpoints to see at what point the model learns those things/how quickly it learns those things/what the learning curve looks like for all those things the fully trained model learns
1. See how this extends to other pythia models/sizes.

## TODO List:
- [ ] Upload all(or a meaningful subset, e.g. every power of 2) pythia-12b checkpoints to HF
- [ ] Analysis of token meanings/categories in the fully trained pythia-12b model
- [ ] Analysis of what meanings show up when in training pythia-12b
- [ ] Potentially expand this to other pythia model sizes to see if this is true across scales?

## Links to stuff that's useful
GSON has uploaded weights here: https://huggingface.co/amphora/pythia-12b-weights

- [Representation Degeneration Problem in Training Natural Language Generation Models](https://openreview.net/forum?id=SkEYojRqtm) & [Is anisotropy really the cause of BERT embeddings not being semantic?](https://aclanthology.org/2022.findings-emnlp.314/) - anisotropic and hypercone behaviour of token embeddings and the latter paper links them to known biases such as frequency, subword, punctuation, and case
- [Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse](https://openreview.net/forum?id=FxVH7iToXS) - probably only relevant part fig1 evolution of cosine angle between tokens, maybe as another way to quantify embedding quality, or some behaviour that might be worth keeping in mind 
- [Interpreting Word Embeddings with Eigenvector Analysis](https://openreview.net/forum?id=rJfJiR5ooX) - embedding svd
