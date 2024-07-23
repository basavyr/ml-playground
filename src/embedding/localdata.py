
train_data = "Machine learning (ML) continues to propel a broad range of applications in image classification [28], voice recognition [12], precision medicine [17], malware/intrusion detection [41], autonomous vehicles [42], and so much more. ML models are vulnerable to adversarial examples minimally perturbed legitimate inputs that fool models to make incorrect predictions [4, 19]. Given an input 𝑥 (e.g., an image) correctly classified by a model 𝑓 , an adversary performs a small perturbation 𝛿 and obtains 𝑥 ′ = 𝑥 + 𝛿 that is indistinguishable from 𝑥 to a human analyst, yet the model misclassifies 𝑥′. Adversarial examples pose realistic threats on domains such as self-driving cars and malware detection for the consequences of incorrect predictions are highly likely to cause real harm [16, 23, 26]. To defend against adversarial examples, previous work took multiple directions each with its pros and cons. Early attempts [20, 24] to harden ML models provided only marginal robustness improvements against adversarial examples. Heuristic defenses based on defensive distillation [37], data transformation [3, 13, 21, 34, 51, 52], and gradient masking [5, 46] were subsequently broken [2, 7, 8, 22]. While adversarial training [19] remains effective against known classes of attacks, robustness comes at a cost of accuracy penalty on clean data. Similarly, data transformation-based defenses also penalize prediction accuracy on legitimate inputs. Certified defenses [10, 32, 33] provide formal robustness guarantee, but are limited to a class of attacks constrained to LP-norms [32, 50]. As outlined by Goodfellow [18], a shared limitation of all prior defense techniques is the static and fixed target nature of the de- ployed ML model. We argue that, although defended by methods such as adversarial training, the very fact that a ML model is a fixed target that continuously responds to prediction queries makes it a prime target for repeated correlated adversarial attacks. As a result, given enough time, an adversary will have the advantage to repeatedly query the prediction API and build enough knowledge about the ML model and eventually fool it. Once the adversary launches a successful attack, it will be always effective since the model is not moving from its compromised “location”. In this paper, we introduce Morphence, an approach that makes a ML model a moving target in the face of adversarial example attacks. By regularly moving the decision function of a model, Morphence makes it significantly challenging for an adversary to successfully fool the model through adversarial examples. Particularly, Morphence significantly reduces the effectiveness of once successful and repeated attacks and attacks that succeed after patiently prob- ing a fixed target model. Morphence deploys a pool of 𝑛 models generated from a base model in a manner that introduces sufficient randomness when it selects the most suitable model to respond to prediction queries. The selection of the most suitable model is governed by a scheduling strategy that relies on the prediction confidence of each model on a given query input. To ensure repeated or correlated attacks fail, the deployed pool of 𝑛 models automatically expires after a query budget is reached. The model pool is then seamlessly replaced by a new pool of 𝑛 models generated and queued in advance. To be practical, Morphence’s moving target defense (MTD) strategy needs to address the following challenges: Challenge-1: Significantly improving Morphence's robustness to adversarial examples across white-box and black-box attacks. Challenge-2: Maintaining accuracy on clean data as close to that of the base model as possible. Challenge-3: Significantly increasing diversity among models in the pool to reduce adversarial example transferability among them. Morphence addresses Challenge-1 by enhancing the MTD aspect through: larger model pool size, a model selection scheduler, and dynamic pool renewal (Sections 3.2 and 3.3). Challenge-2 is addressed by re-training each generated model to regain accu- racy loss caused by perturbations (Section 3.2: step-2). Morphence addresses Challenge-3 by making the individual models distant"

test_data = "Robustness to adversarial examples of machine learning models remains an open topic of research. Attacks often succeed by repeatedly probing a fixed target model with adversarial examples purposely crafted to fool it. In this paper, we introduce Morphence, an approach that shifts the defense landscape by making a model a moving target against adversarial examples. By regularly moving the decision function of a model, Morphence makes it significantly challenging for repeated or correlated attacks to succeed. Morphence deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence on two benchmark image classification datasets (MNIST and CIFAR10) against five reference attacks (2 white-box and 3 black-box). In all cases, Morphence consistently outperforms the thus-far effective defense, adversarial training, even in the face of strong white-box attacks, while preserving accuracy on clean data."

train_data_2 = """
Word Embeddings: Encoding Lexical Semantics
===========================================

Word embeddings are dense vectors of real numbers, one per word in your
vocabulary. In NLP, it is almost always the case that your features are
words! But how should you represent a word in a computer? You could
store its ascii character representation, but that only tells you what
the word *is*, it doesn't say much about what it *means* (you might be
able to derive its part of speech from its affixes, or properties from
its capitalization, but not much). Even more, in what sense could you
combine these representations? We often want dense outputs from our
neural networks, where the inputs are :math:`|V|` dimensional, where
:math:`V` is our vocabulary, but often the outputs are only a few
dimensional (if we are only predicting a handful of labels, for
instance). How do we get from a massive dimensional space to a smaller
dimensional space?

How about instead of ascii representations, we use a one-hot encoding?
That is, we represent the word :math:`w` by

.. math::  \overbrace{\left[ 0, 0, \dots, 1, \dots, 0, 0 \right]}^\text{|V| elements}

where the 1 is in a location unique to :math:`w`. Any other word will
have a 1 in some other location, and a 0 everywhere else.

There is an enormous drawback to this representation, besides just how
huge it is. It basically treats all words as independent entities with
no relation to each other. What we really want is some notion of
*similarity* between words. Why? Let's see an example.

Suppose we are building a language model. Suppose we have seen the
sentences

* The mathematician ran to the store.
* The physicist ran to the store.
* The mathematician solved the open problem.

in our training data. Now suppose we get a new sentence never before
seen in our training data:

* The physicist solved the open problem.

Our language model might do OK on this sentence, but wouldn't it be much
better if we could use the following two facts:

* We have seen  mathematician and physicist in the same role in a sentence. Somehow they
  have a semantic relation.
* We have seen mathematician in the same role  in this new unseen sentence
  as we are now seeing physicist.

and then infer that physicist is actually a good fit in the new unseen
sentence? This is what we mean by a notion of similarity: we mean
*semantic similarity*, not simply having similar orthographic
representations. It is a technique to combat the sparsity of linguistic
data, by connecting the dots between what we have seen and what we
haven't. This example of course relies on a fundamental linguistic
assumption: that words appearing in similar contexts are related to each
other semantically. This is called the `distributional
hypothesis <https://en.wikipedia.org/wiki/Distributional_semantics>`__.


Getting Dense Word Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How can we solve this problem? That is, how could we actually encode
semantic similarity in words? Maybe we think up some semantic
attributes. For example, we see that both mathematicians and physicists
can run, so maybe we give these words a high score for the "is able to
run" semantic attribute. Think of some other attributes, and imagine
what you might score some common words on those attributes.

If each attribute is a dimension, then we might give each word a vector,
like this:

.. math::

    q_\text{mathematician} = \left[ \overbrace{2.3}^\text{can run},
   \overbrace{9.4}^\text{likes coffee}, \overbrace{-5.5}^\text{majored in Physics}, \dots \right]

.. math::

    q_\text{physicist} = \left[ \overbrace{2.5}^\text{can run},
   \overbrace{9.1}^\text{likes coffee}, \overbrace{6.4}^\text{majored in Physics}, \dots \right]

Then we can get a measure of similarity between these words by doing:

.. math::  \text{Similarity}(\text{physicist}, \text{mathematician}) = q_\text{physicist} \cdot q_\text{mathematician}

Although it is more common to normalize by the lengths:

.. math::

    \text{Similarity}(\text{physicist}, \text{mathematician}) = \frac{q_\text{physicist} \cdot q_\text{mathematician}}
   {\| q_\text{physicist} \| \| q_\text{mathematician} \|} = \cos (\phi)

Where :math:`\phi` is the angle between the two vectors. That way,
extremely similar words (words whose embeddings point in the same
direction) will have similarity 1. Extremely dissimilar words should
have similarity -1.


You can think of the sparse one-hot vectors from the beginning of this
section as a special case of these new vectors we have defined, where
each word basically has similarity 0, and we gave each word some unique
semantic attribute. These new vectors are *dense*, which is to say their
entries are (typically) non-zero.

But these new vectors are a big pain: you could think of thousands of
different semantic attributes that might be relevant to determining
similarity, and how on earth would you set the values of the different
attributes? Central to the idea of deep learning is that the neural
network learns representations of the features, rather than requiring
the programmer to design them herself. So why not just let the word
embeddings be parameters in our model, and then be updated during
training? This is exactly what we will do. We will have some *latent
semantic attributes* that the network can, in principle, learn. Note
that the word embeddings will probably not be interpretable. That is,
although with our hand-crafted vectors above we can see that
mathematicians and physicists are similar in that they both like coffee,
if we allow a neural network to learn the embeddings and see that both
mathematicians and physicists have a large value in the second
dimension, it is not clear what that means. They are similar in some
latent semantic dimension, but this probably has no interpretation to
us.


In summary, **word embeddings are a representation of the *semantics* of
a word, efficiently encoding semantic information that might be relevant
to the task at hand**. You can embed other things too: part of speech
tags, parse trees, anything! The idea of feature embeddings is central
to the field.


Word Embeddings in Pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we get to a worked example and an exercise, a few quick notes
about how to use embeddings in Pytorch and in deep learning programming
in general. Similar to how we defined a unique index for each word when
making one-hot vectors, we also need to define an index for each word
when using embeddings. These will be keys into a lookup table. That is,
embeddings are stored as a :math:`|V| \times D` matrix, where :math:`D`
is the dimensionality of the embeddings, such that the word assigned
index :math:`i` has its embedding stored in the :math:`i`'th row of the
matrix. In all of my code, the mapping from words to indices is a
dictionary named word\_to\_ix.

The module that allows you to use embeddings is torch.nn.Embedding,
which takes two arguments: the vocabulary size, and the dimensionality
of the embeddings.

To index into this table, you must use torch.LongTensor (since the
indices are integers, not floats).

"""
