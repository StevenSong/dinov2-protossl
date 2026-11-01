# ProtoViT-style prototype layers shared by ProtoSSL pretraining and downstream (SupProto) models
from ._bank import PrototypeBank, subtract_cls
from ._greedy import GreedyMatch, check_feasible, greedy_match, neighbor_table
