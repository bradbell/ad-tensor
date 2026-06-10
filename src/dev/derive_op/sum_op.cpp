// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/size_ptr2array_ref.hpp>
#include <ad_tensor/dev/rev_sum_view.hpp>
//
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void sum_op_t<TensorType>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
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
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        if( n_dim == 0 ) {
            // par_vec
            par_vec[op_index] = par_vec[operand_index].sum();
        } else {
            //
            // dim
            lock = true;
            c10::IntArrayRef dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            //
            // par_vec
            par_vec[op_index] = par_vec[operand_index].sum(dim);
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
        }
    }
    template void sum_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void sum_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void sum_op_t<TensorType>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
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
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        if( n_dim == 0 ) {
            // var_vec
            var_vec[op_index] = var_vec[operand_index].sum();
        } else {
            //
            // dim
            lock = true;
            c10::IntArrayRef dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // var_vec
            var_vec[op_index] = var_vec[operand_index].sum(dim);
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
        }
    }
    template void sum_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void sum_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    template<class TensorType>
    void sum_op_t<TensorType>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
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
        // operand_index
        size_t operand_index  = agraph.m_arg_value[arg_index];
        //
        if( n_dim == 0 ) {
            // for_der
            for_der[op_index] = for_der[operand_index].sum();
        } else {
            //
            // dim
            lock = true;
            c10::IntArrayRef dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // for_der
            for_der[op_index] = for_der[operand_index].sum(dim);
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
        }
    }
    template void sum_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void sum_op_t<at::Tensor>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          for_der
    ) const;
    // ------------------------------------------------------------------------
    // reverse_der
    template<class TensorType>
    void sum_op_t<TensorType>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
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
        // operand_index, operand_shape
        size_t           operand_index  = agraph.m_arg_value[arg_index];
        c10::IntArrayRef operand_shape  = var_vec[operand_index].sizes();
        //
        // rev_der[operand_index]
        if( rev_der[op_index].numel() == 1 ) {
            if( rev_der[operand_index].numel() == 0 ) {
                TensorType zeros   = TensorType( torch::zeros(operand_shape) );
                rev_der[operand_index] = zeros + rev_der[op_index];
            } else {
                rev_der[operand_index] += rev_der[op_index];
            }
        } else {
            assert( n_dim != 0 );
            //
            // dim
            lock = true;
            c10::IntArrayRef dim = size_ptr2array_ref(
                lock, agraph.m_arg_value.data() + arg_index + 1
            );
            assert( dim.size() == n_dim );
            //
            // res_shape
            lock = true;
            c10::IntArrayRef res_shape = rev_sum_view(
                lock,
                dim,
                var_vec[op_index].sizes(),
                var_vec[operand_index].sizes()
            );
            if( rev_der[operand_index].numel() == 0 ) {
                TensorType zeros   = TensorType( torch::zeros(operand_shape) );
                rev_der[operand_index] = zeros + \
                    rev_der[op_index].view(res_shape);
            } else {
                rev_der[operand_index] += rev_der[op_index].view(res_shape);
            }
            //
            // dim
            lock = false;
            size_ptr2array_ref(lock);
            rev_sum_view(lock);
        }
    }
    template void sum_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void sum_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
