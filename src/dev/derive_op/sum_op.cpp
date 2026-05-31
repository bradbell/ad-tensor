// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/size_ptr2array_ref.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // op_enum
    op_enum_t sum_op_t::op_enum(void) const {
        return op_enum_t::sum;
    }
    // ------------------------------------------------------------------------
    // forward_par
    void sum_op_t::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const {
        //
        // arg_index
        size_t    arg_index = agraph.m_arg_start[op_index];
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_index];
        assert( ad_type  == ad_type_t::parameter );
        //
        // n_dim
        size_t n_dim = agraph.m_arg_value[arg_index + 1];
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_index + 1] );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 + n_dim );
#endif
        //
        // dim
        c10::ArrayRef<long int> dim = size_ptr2array_ref(
            agraph.m_arg_value.data() + arg_index + 1
        );
        //
        // par_vec
        size_t par_index  = agraph.m_arg_value[arg_index];
        par_vec[op_index] = par_vec[par_index].sum(dim);
    }
    // ------------------------------------------------------------------------
    // forward_var
    void sum_op_t::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const {
        //
        // arg_index
        size_t    arg_index = agraph.m_arg_start[op_index];
        //
        // n_dim
        size_t n_dim = agraph.m_arg_value[arg_index + 1];
        assert( ad_type_t::none ==  agraph.m_arg_type[arg_index + 1] );
        //
#ifndef NDEBUG
        //
        // ad_type
        ad_type_t ad_type   = agraph.m_arg_type[arg_index];
        assert( ad_type  == ad_type_t::variable );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 + n_dim );
#endif
        // par_vec
        size_t var_index  = agraph.m_arg_value[arg_index];
        if( n_dim == 0 ) {
            var_vec[op_index] = var_vec[var_index].sum();
        } else {
            //
            // dim
            c10::ArrayRef<long int> dim = size_ptr2array_ref(
                agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // par_vec
            var_vec[op_index] = var_vec[var_index].sum(dim);
        }
    }
    // ------------------------------------------------------------------------
    // forward_der
    void sum_op_t::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          for_der
    ) const {
        assert(false && "sum: forward_der not implemented");
    }
    // ------------------------------------------------------------------------
    // reverse_der
    void sum_op_t::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const {
        assert(false && "sum: reverse_der not implemented");
    }
} }
