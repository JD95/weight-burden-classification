# Weight Burden Classification

We are going to attempt to classify animations of movements while holding different weights from motion capture data.

## Plan of Attack

- Use positional information of skeleton to augment initial motion capture dataset
- Train an auto encoder to reduce dimensionality of data to 7
- Train a reccurent neural network to classify sequences of reduced motion capture frames into light, normal, or heavy weight burden classes.
