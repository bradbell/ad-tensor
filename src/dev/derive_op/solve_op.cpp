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
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<TensorType>&          par_vec
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
        // left
        bool left = bool( agraph.m_arg_value[arg_start + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3  );
        assert( agraph.m_arg_type[arg_start+2] == adtype_t::none );
# endif
        //
        // linear_ten, rhs_ten
        TensorType linear_ten  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec
        );
        TensorType rhs_ten  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_vec
        );
        //
        // par_vec
        par_vec[op_index] = linalg_solve(linear_ten, rhs_ten, left);
    }
    template void solve_op_t<adten_t>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<adten_t>&             par_vec
    ) const;
    template void solve_op_t<at::Tensor>::forward_par(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        vector<at::Tensor>&          par_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_var
    template<class TensorType>
    void solve_op_t<TensorType>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        vector<TensorType>&          var_vec
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
        // left
        bool left = bool( agraph.m_arg_value[arg_start + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
        assert( agraph.m_arg_type[arg_start+2] == adtype_t::none );
# endif
        //
        // linear_ten, rhs_ten
        TensorType linear_ten  = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        TensorType rhs_ten  = tensor_at_arg_index(
            arg_start + 1, agraph, con_vec, par_vec, var_vec
        );
        //
        // var_vec
        var_vec[op_index] = linalg_solve(linear_ten, rhs_ten, left);
    }
    template void solve_op_t<adten_t>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        vector<adten_t>&             var_vec
    ) const;
    template void solve_op_t<at::Tensor>::forward_var(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        vector<at::Tensor>&          var_vec
    ) const;
    // ------------------------------------------------------------------------
    // forward_der
    // see Guiles-2008 Section 2.3.1
    // linear * solution_dot = rsh_dot - linear_dot * solution   (left)
    // solution_dot * linear = rsh_dot - solution * linear_dot   (! left)
    template<class TensorType>
    void solve_op_t<TensorType>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          for_der
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
        // left, transpose
        bool left      = bool( agraph.m_arg_value[arg_start + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
        assert( agraph.m_arg_type[arg_start+2] == adtype_t::none );
# endif
        // variable
        adtype_t variable = adtype_t::variable;
        //
        // linear_index, rhs_index
        size_t linear_index = agraph.m_arg_value[arg_start];
        size_t rhs_index = agraph.m_arg_value[arg_start + 1];
        //
        // linear_type, rhs_type
        adtype_t linear_type = agraph.m_arg_type[arg_start];
        adtype_t rhs_type = agraph.m_arg_type[arg_start + 1];
        if( linear_type == variable && for_der[linear_index].numel() == 0 ) {
            linear_type = adtype_t::constant;
        }
        if( rhs_type == variable && for_der[rhs_index].numel() == 0 ) {
            rhs_type = adtype_t::constant;
        }
        //
        // linear
        TensorType linear = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        //
        // prod
        TensorType prod = TensorType( torch::empty( {0} ) );
        if( left && linear_type == adtype_t::variable ) {
            prod = for_der[linear_index].matmul( var_vec[op_index] );
        } else if( ! left && linear_type == adtype_t::variable ) {
            prod = var_vec[op_index].matmul( for_der[linear_index] );
        }
        // diff
        TensorType diff = TensorType( torch::empty( {0} ) );
        if( rhs_type == adtype_t::variable ) {
            diff = for_der[rhs_index];
        }
        minus_equal(diff, prod);
        assert( diff.numel() != 0 );
        //
        // solution_dot
        for_der[op_index] = linalg_solve(linear, diff, left);
    }
    template void solve_op_t<adten_t>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             for_der
    ) const;
    template void solve_op_t<at::Tensor>::forward_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          for_der
    ) const;
    /* ------------------------------------------------------------------------
    reverse_der
    see Guiles-2008 Section 2.3.1
    left:
        rhs_bar    = linear^-T * solution_bar
        linear_bar = - rhs_bar * solution^T
    ! left:
        rhs_bar     = solution_bar * linear^-T
        linear_bar  = - solution^T * rhs_bar
    */
    template<class TensorType>
    void solve_op_t<TensorType>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<TensorType>&    par_vec     ,
        const vector<TensorType>&    var_vec     ,
        vector<TensorType>&          rev_der
    ) const {
        //
        // arg_start
        size_t arg_start = agraph.m_arg_start[op_index];
        //
        // left, transpose
        bool left      = bool( agraph.m_arg_value[arg_start + 2] );
        //
#ifndef NDEBUG
        size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
        assert( n_arg == 3 );
        assert( agraph.m_arg_type[arg_start+2] == adtype_t::none );
# endif
        //
        // linear_type, rhs_type
        adtype_t linear_type     = agraph.m_arg_type[arg_start];
        adtype_t rhs_type        = agraph.m_arg_type[arg_start + 1];
        //
        // linear_index, rhs_index
        size_t linear_index = agraph.m_arg_value[arg_start];
        size_t rhs_index = agraph.m_arg_value[arg_start + 1];
        //
        // linear
        TensorType linear = tensor_at_arg_index(
            arg_start, agraph, con_vec, par_vec, var_vec
        );
        // linear_tra
        size_t n_lin = linear.sizes().size();
        TensorType linear_tra   = linear.transpose(n_lin-2, n_lin-1);
        //
        // solution_tra
        size_t n_sol = var_vec[op_index].sizes().size();
        TensorType solution_tra = var_vec[op_index].transpose(n_sol-2, n_sol-1);
        //
        // rhs_bar
        TensorType rhs_bar = linalg_solve(linear_tra, rev_der[op_index], left);
        if( rhs_type == adtype_t::variable ) {
            plus_equal(rev_der[rhs_index], rhs_bar);
        }
        //
        // linear_bar
        if( linear_type == adtype_t::variable ) {
            if( left ) {
                TensorType prod = rhs_bar.matmul(solution_tra);
                minus_equal(rev_der[linear_index], prod);
            } else {
                TensorType prod = solution_tra.matmul( rhs_bar );
                minus_equal(rev_der[linear_index], prod);
            }
        }
    }
    template void solve_op_t<adten_t>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<adten_t>&       par_vec     ,
        const vector<adten_t>&       var_vec     ,
        vector<adten_t>&             rev_der
    ) const;
    template void solve_op_t<at::Tensor>::reverse_der(
        size_t                       op_index    ,
        const agraph_t&              agraph      ,
        const vector<at::Tensor>&    con_vec     ,
        const vector<at::Tensor>&    par_vec     ,
        const vector<at::Tensor>&    var_vec     ,
        vector<at::Tensor>&          rev_der
    ) const;
} }
