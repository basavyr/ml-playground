
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

dim = 2

extra = 10
dim += extra
embedding_size = 10

words = ["gicu", "home", "house", "stocks"]

word_to_ix = {}

for id in range(len(words)):
    word_to_ix.update({f'{words[id-1]}': id})


test_word = "gicu"
embeds = nn.Embedding(dim, embedding_size)

if test_word in words:
    test_embed = embeds(torch.tensor(
        [word_to_ix[test_word]], dtype=torch.long))
    print(test_embed)
else:
    test_embed = embeds(torch.tensor([word_to_ix["gicu"]], dtype=torch.long))
    print(test_embed)

exit(1)


class Model(nn.Module):
    def __init__(self, embedding_size: int):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding(1, embedding_size)


net = Model(5)


print(net)
