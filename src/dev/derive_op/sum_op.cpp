// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/size_ptr2array_ref.hpp>
#include <ad_tensor/dev/rev_sum_reshape.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    void sum_op_t::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const {
        //
        // lock
        bool lock;
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
        assert( ad_type  == ad_type_t::parameter );
        //
        // n_arg
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 2 + n_dim );
#endif
        // par_index
        size_t par_index  = agraph.m_arg_value[arg_index];
        //
        if( n_dim == 0 ) {
            // par_vec
            par_vec[op_index] = par_vec[par_index].sum();
        } else {
            //
            // dim
            lock = true;
            c10::ArrayRef<long> dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            //
            // par_vec
            par_vec[op_index] = par_vec[par_index].sum(dim);
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
        }
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
        // lock
        bool lock;
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
        // var_index
        size_t var_index  = agraph.m_arg_value[arg_index];
        //
        if( n_dim == 0 ) {
            // var_vec
            var_vec[op_index] = var_vec[var_index].sum();
        } else {
            //
            // dim
            lock = true;
            c10::ArrayRef<long> dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // var_vec
            var_vec[op_index] = var_vec[var_index].sum(dim);
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
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
        //
        // lock
        bool lock;
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
        // var_index
        size_t var_index  = agraph.m_arg_value[arg_index];
        //
        if( n_dim == 0 ) {
            // for_der
            for_der[op_index] = for_der[var_index].sum();
        } else {
            //
            // dim
            lock = true;
            c10::ArrayRef<long> dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // for_der
            for_der[op_index] = for_der[var_index].sum(dim);
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
        }
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
        //
        // check for case where this operation is not connected to the range
        if( rev_der[op_index].numel() == 0 ) {
            return;
        }
        //
        // lock
        bool lock;
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
        // lhs_index, lhs_shape
        size_t              lhs_index  = agraph.m_arg_value[arg_index];
        c10::ArrayRef<long> lhs_shape  = var_vec[lhs_index].sizes();
        //
        // rev_der[lhs_index]
        if( rev_der[op_index].numel() == 1 ) {
            if( rev_der[lhs_index].numel() == 0 ) {
                at::Tensor zeros   = torch::zeros(lhs_shape);
                rev_der[lhs_index] = zeros + rev_der[op_index];
            } else {
                rev_der[lhs_index] += rev_der[op_index];
            }
        } else {
            assert( n_dim != 0 );
            //
            // dim
            lock = true;
            c10::ArrayRef<long> dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // res_shape
            lock = true;
            c10::ArrayRef<long> res_shape = rev_sum_reshape(
                lock, dim, var_vec[op_index], var_vec[lhs_index]
            );
            if( rev_der[lhs_index].numel() == 0 ) {
                at::Tensor zeros   = torch::zeros(lhs_shape);
                rev_der[lhs_index] = zeros + \
                    rev_der[op_index].reshape(res_shape);
            } else {
                rev_der[lhs_index] += rev_der[op_index].reshape(res_shape);
            }
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
            rev_sum_reshape(lock);
        }
    }
} }
