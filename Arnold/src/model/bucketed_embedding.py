import torch.nn as nn
import torch

class BucketedEmbedding(nn.Embedding):

    def __init__(self, bucket_size, num_embeddings, *args, **kwargs):
        self.bucket_size = bucket_size
        real_num_embeddings = (num_embeddings + bucket_size - 1) // bucket_size
        super(BucketedEmbedding, self).__init__(real_num_embeddings, *args, **kwargs)

    def forward(self, indices):
        print ("Shravs type", indices.div(self.bucket_size).dtype)
        return super(BucketedEmbedding, self).forward(indices.div(self.bucket_size).to(torch.int64))
