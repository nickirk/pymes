/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TOPOLOGY_H__
#define __INT_TOPOLOGY_H__

#include "../interface/common.h"

namespace CTF_int {
  class mapping;
  enum TOPOLOGY { TOPOLOGY_GENERIC, TOPOLOGY_BGP, TOPOLOGY_BGQ,
                  TOPOLOGY_8D, NO_TOPOLOGY };

  /* \brief mesh/torus topology configuration */
  class topology {
    public:
      // number of dimensions in torus
      int        order;
      // lengths of dimensions
      int *      lens;
      // lda[i] = lens[i-1] * ... * lens[0]
      int *      lda;
      // global communicator is reordered if intra-node grid is provided
      int        is_reordered;
      // whether dim_comm communicators have been activated
      bool       is_activated;

      // list of communicators along fibers of each dimension of torus 
      CommData * dim_comm;
      // global communicator, ordered as in torus given by dim_comm
      CommData   glb_comm;
      // global communicator, ordered as given, assuming processors are ordered as [processes in node 1], [processes in node 2], etc.
      //CommData   input_comm;

      //topology();
      ~topology();

      /** 
       * \brief copy constructor
       * \param[in] other topology to copy
       */
      topology(topology const & other);

      /**
       * \brief constructs torus topology, if intra_node_lens is NULL, the p processors are folded into a torus, otherwise, the each set of prod(intra_node_lens) processors is mapped to different modes of the processor grid, e.g., if lens_ = [6,4] and intra_node_lens=[3,2] (6 processes per node), the processors are assiged as
       * [[ 0  1  2  6  7  8 ],
       *  [ 3  4  5  9  10 11],
       *  [ 12 13 14 18 19 20],
       *  [ 15 16 17 21 22 23]]
       * \param[in] order_ number of torus dimensions
       * \param[in] lens_ lengths of torus dimensions
       * \param[in] cdt communicator for whole torus 
       * \param[in] activate whether to create MPI_Comms
       * \param[in] intra_node_lens lengths of intra-node processor grid
       */
      topology(int         order_,
               int const * lens_,
               CommData    cdt,
               bool        activate=false,
               int const * intra_node_lens=NULL);
     
      /* \brief create (split off) MPI communicators, re-entrant */ 
      void activate();

      /* \breif free MPI communicators, re-entrant */
      void deactivate();
  };

  /**
   * \brief determine this processors rank in the global communicator given by reordering nodes so that they adhere to the assignment described in the constructor of the topology() object, assuming initial order is node by node
   *
   * \param[in] order_ number of torus dimensions
   * \param[in] lens_ lengths of torus dimensions
   * \param[in] intra_node_lens lengths of intra-node processor grid
   */
  int get_topo_reorder_rank(int order, int const * lens, int const * intra_node_lens);


  /**
   * \brief get dimension and torus lengths of specified topology
   *
   * \param[in] glb_comm communicator
   * \param[in] mach specified topology
   */
  topology * get_phys_topo(CommData glb_comm,
                           TOPOLOGY mach);

  /**
   * \brief computes all topology configurations given undelying physical topology information
   * \param[in] cdt global communicator
   */
  std::vector< topology* > get_generic_topovec(CommData   cdt);

  /**
   * \brief folds specified topology and all of its permutations into all configurations of lesser dimensionality
   * \param[in] phys_topology topology to fold
   * \param[in] cdt  global communicator
   */
  std::vector< topology* > peel_perm_torus(topology * phys_topology,
                                           CommData   cdt);
   /**
   * \brief folds specified topology into all configurations of lesser dimensionality
   * \param[in] topo topology to fold
   * \param[in] glb_comm  global communicator
   */
  std::vector< topology* > peel_torus(topology const * topo,
                                      CommData         glb_comm);

  /**
   * \brief searches for an equivalent topology in avector of topologies
   * \param[in] topo topology to match
   * \param[in] topovec vector of existing parameters
   * \return -1 if not found, otherwise index of first found topology
   */
  int find_topology(topology const *           topo,
                    std::vector< topology* > & topovec);

 
  /**
   * \brief get the best topologoes (least nvirt) over all procs
   * \param[in] nvirt best virtualization achieved by this proc
   * \param[in] topo topology index corresponding to best virtualization
   * \param[in] global_comm is the global communicator
   * \param[in] bcomm_vol best comm volume computed
   * \param[in] bmemuse best memory usage computed
   * return virtualization factor
   */
   int get_best_topo(int64_t  nvirt,
                     int      topo,
                     CommData global_comm,
                     int64_t  bcomm_vol=0,
                     int64_t  bmemuse=0);
   

  /**
   * \brief extracts the set of physical dimensions still available for mapping
   * \param[in] topo topology
   * \param[in] order_A dimension of A
   * \param[in] edge_map_A mapping of A
   * \param[in] order_B dimension of B
   * \param[in] edge_map_B mapping of B
   * \param[out] num_sub_phys_dims number of free torus dimensions
   * \param[out] psub_phys_comm the torus dimensions
   * \param[out] pcomm_idx index of the free torus dimensions in the origin topology
   */
  void extract_free_comms(topology const *  topo,
                          int               order_A,
                          mapping const *   edge_map_A,
                          int               order_B,
                          mapping const *   edge_map_B,
                          int &             num_sub_phys_dims,
                          CommData *  *     psub_phys_comm,
                          int **            pcomm_idx);


  /**
   * \brief determines if two topologies are compatible with each other
   * \param topo_keep topology to keep (larger dimension)
   * \param topo_change topology to change (smaller dimension)
   * \return true if its possible to change
   */
  int can_morph(topology const * topo_keep,
                topology const * topo_change);

  /**
   * \brief morphs a tensor topology into another
   * \param[in] new_topo topology to change to
   * \param[in] old_topo topology we are changing from
   * \param[in] order number of tensor dimensions
   * \param[in,out] edge_map mapping whose topology mapping we are changing
   */
  void morph_topo(topology const * new_topo,
                  topology const * old_topo,
                  int              order,
                  mapping *        edge_map);
}

#endif
