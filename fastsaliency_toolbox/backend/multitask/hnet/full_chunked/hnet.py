import torch
from typing import List, Union
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP

from backend.multitask.hnet.hnet_interface import AHNET

class HNET(AHNET):
    def __init__(self, target_shapes : List[torch.Size], hnet_conf : dict):
        super().__init__(target_shapes)

        self.hnet = ChunkedHMLP(
            target_shapes=target_shapes, 
            chunk_size=hnet_conf["chunk_size"],
            layers=hnet_conf["hidden_layers"], # the sizes of the hidden layers (excluding the last layer that generates the weights)
            cond_in_size=hnet_conf["embedding_size"], # the size of the embeddings
            num_cond_embs=hnet_conf["task_cnt"], # the number of embeddings we want to learn
            cond_chunk_embs = hnet_conf["chunk_emb_per_task"], # chunk embeddings depend on task id
            chunk_emb_size=hnet_conf["chunk_emb_size"] # size of the chunk embeddings
        )

    def forward(self, task_id : Union[int, List[int]]):
        return self.hnet.forward(cond_id = task_id)

    def freeze_hnet_for_catchup(self):
        print([n for n,p in self.hnet.named_parameters()])
        
    def unfreeze_hnet_from_catchup(self):
        pass

    def get_gradients_on_outputs(self) -> List[torch.Tensor]:
        print("Warning: observing gradients of ChunkedHLMP is currently not yet supported!")
        return []