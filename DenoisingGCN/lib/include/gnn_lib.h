#ifndef GNN_LIB_H
#define GNN_LIB_H

#include "/home/xujiarong/baseline/pytorch_DGCNN/lib/include/config.h"

extern "C" int Init(const int argc, const char **argv);

extern "C" void *GetGraphStruct();

extern "C" int PrepareBatchGraph(void *_batch_graph,
                                 const int num_graphs,
                                 const int *num_nodes,
                                 const int *num_edges,
                                 const int *num_noises,
                                 void **list_of_edge_pairs,
                                 void **list_of_noise_pairs,
                                 int is_directed);

extern "C" int PrepareSparseMatrices(void *_batch_graph,
                                     void **list_of_idxes,
                                     void **list_of_vals);

extern "C" int NumEdgePairs(void *_graph);

#endif
