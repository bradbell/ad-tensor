// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/derive_op.hpp>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/broadcast.hpp>
#include <ad_tensor/dev/plus_minus_equal.hpp>
#include <ad_tensor/dev/tensor_at_index.hpp>
//
namespace {
    using at::linalg_solve;         // used for at::Tensor equations
    using ad_tensor::linalg_solve;  // user for ad_tensor::adten_t equations
}
namespace ad_tensor { namespace dev {
    // ------------------------------------------------------------------------
    // forward_par
    template<class TensorType>
    void solve_op_t<TensorType>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<TensorType>&          par_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
        // left
        bool left = bool( agraph.m_arg_value[arg_index + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 3  );
        assert( agraph.m_arg_type[arg_index+2] == ad_type_t::none );
# endif
        //
        // square_ten, rhs_ten
        TensorType square_ten  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec
        );
        TensorType rhs_ten  = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec[op_index] = linalg_solve(square_ten, rhs_ten, left);
    }
    template void solve_op_t<adten_t>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<adten_t>&             par_vec
    ) const;
    template void solve_op_t<at::Tensor>::forward_par(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        ad_tensor::vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void solve_op_t<TensorType>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        ad_tensor::vector<TensorType>&          var_vec
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
        // left
        bool left = bool( agraph.m_arg_value[arg_index + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 3 );
        assert( agraph.m_arg_type[arg_index+2] == ad_type_t::none );
# endif
        //
        // square_ten, rhs_ten
        TensorType square_ten  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        );
        TensorType rhs_ten  = tensor_at_arg_index(
            arg_index + 1, agraph, con_vec, par_vec, var_vec
        );
        //
        // var_vec
        var_vec[op_index] = linalg_solve(square_ten, rhs_ten, left);
    }
    template void solve_op_t<adten_t>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<adten_t>&       par_vec     ,
        ad_tensor::vector<adten_t>&             var_vec
    ) const;
    template void solve_op_t<at::Tensor>::forward_var(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        ad_tensor::vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    // see Guiles-2008 Section 2.3.1
    // square * solution_dot = rsh_dot - square_dot * solution   (left)
    // solution_dot * square = rsh_dot - solution * square_dot   (! left)
    template<class TensorType>
    void solve_op_t<TensorType>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          for_der
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
        // left, transpose
        bool left      = bool( agraph.m_arg_value[arg_index + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 3 );
        assert( agraph.m_arg_type[arg_index+2] == ad_type_t::none );
# endif
        //
        // square_type, rhs_type
        ad_type_t square_type     = agraph.m_arg_type[arg_index];
        ad_type_t rhs_type        = agraph.m_arg_type[arg_index + 1];
        //
        // square_index, rhs_index
        size_t square_index = agraph.m_arg_value[arg_index];
        size_t rhs_index = agraph.m_arg_value[arg_index + 1];
        //
        // square
        TensorType square = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        );
        //
        // prod
        TensorType prod = TensorType( torch::empty( {0} ) );
        if( left && square_type == ad_type_t::variable ) {
            prod = for_der[square_index].matmul( var_vec[op_index] );
        } else if( ! left && square_type == ad_type_t::variable ) {
            prod = var_vec[op_index].matmul( for_der[square_index] );
        }
        // diff
        TensorType diff = TensorType( torch::empty( {0} ) );
        if( rhs_type == ad_type_t::variable ) {
            diff = for_der[rhs_index];
        }
        minus_equal(diff, prod);
        assert( diff.numel() != 0 );
        //
        // solution_dot
        for_der[op_index] = linalg_solve(square, diff, left);
    }
    template void solve_op_t<adten_t>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<adten_t>&       par_vec     ,
        const ad_tensor::vector<adten_t>&       var_vec     ,
        ad_tensor::vector<adten_t>&             for_der
    ) const;
    template void solve_op_t<at::Tensor>::forward_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          for_der
    ) const;
    // ------------------------------------------------------------------------
    // reverse_der
    // see Guiles-2008 Section 2.3.1
    template<class TensorType>
    void solve_op_t<TensorType>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<TensorType>&    par_vec     ,
        const ad_tensor::vector<TensorType>&    var_vec     ,
        ad_tensor::vector<TensorType>&          rev_der
    ) const {
        //
        // arg_index
        size_t arg_index = agraph.m_arg_start[op_index];
        //
        // left, transpose
        bool left      = bool( agraph.m_arg_value[arg_index + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_index;
        assert( n_arg == 3 );
        assert( agraph.m_arg_type[arg_index+2] == ad_type_t::none );
# endif
        //
        // square_type, rhs_type
        ad_type_t square_type     = agraph.m_arg_type[arg_index];
        ad_type_t rhs_type        = agraph.m_arg_type[arg_index + 1];
        //
        // square_index, rhs_index
        size_t square_index = agraph.m_arg_value[arg_index];
        size_t rhs_index = agraph.m_arg_value[arg_index + 1];
        //
        // square_ten
        TensorType square_ten  = tensor_at_arg_index(
            arg_index, agraph, con_vec, par_vec, var_vec
        );
        //
        //
        if( left ) {
            //
            // square_tra
            TensorType square_tra = square_ten.transpose(0, 1);
            //
            // square_tra * rhs_bar = solution_bar
            TensorType rhs_bar =
                linalg_solve(square_tra, rev_der[op_index], left);
            if( rhs_type == ad_type_t::variable ) {
                rev_der[rhs_index] = rhs_bar;
            }
            //  square_bar = - rhs_bar * solution
            if( square_type == ad_type_t::variable ) {
                rev_der[square_index] = - rhs_bar.matmul( var_vec[op_index] );
            }
        } else {
            // solution_bar_tra
            TensorType solution_bar_tra = rev_der[op_index].transpose(0, 1);
            //
            // square_ten * rhs_bar = solution_bar_tra
            TensorType rhs_bar =
                linalg_solve(square_ten, solution_bar_tra, left);
            if( rhs_type == ad_type_t::variable ) {
                rev_der[rhs_index] = rhs_bar;
            }
            // square_bar = - rhs_bar * solution
            if( square_type == ad_type_t::variable ) {
                rev_der[square_index] = - rhs_bar.matmul( var_vec[op_index] );
            }
        }
    }
    template void solve_op_t<adten_t>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<adten_t>&       par_vec     ,
        const ad_tensor::vector<adten_t>&       var_vec     ,
        ad_tensor::vector<adten_t>&             rev_der
    ) const;
    template void solve_op_t<at::Tensor>::reverse_der(
        size_t                                  op_index    ,
        const agraph_t&                         agraph      ,
        const ad_tensor::vector<at::Tensor>&    con_vec     ,
        const ad_tensor::vector<at::Tensor>&    par_vec     ,
        const ad_tensor::vector<at::Tensor>&    var_vec     ,
        ad_tensor::vector<at::Tensor>&          rev_der
    ) const;
} }
