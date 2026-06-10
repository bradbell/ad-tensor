// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin base_op_static dev}
{xrst_spell
    agraph
}

tensor_at_arg_index
###################

Prototype
*********
{xrst_literal ,
    BEGIN_TENSOR_AT_ARG_INDEX_PAR, END_TENSOR_AT_ARG_INDEX_PAR
    BEGIN_TENSOR_AT_ARG_INDEX_VAR, END_TENSOR_AT_ARG_INDEX_VAR
}

arg_index
*********
This is the index in agraph.m_arg_value and agraph.m_arg_type
for this operator usage.

agraph
******
is the acyclic graph that contains the information for this operator usage.

m_arg_type
==========
If var_vec is present, m_arg_type[arg_index] must be
constant, parameter, or variable.
If var_vec is not present, m_arg_type[arg_index] must be
constant or parameter.

con_vec
*******
is the vector of constant tensors.

par_vec
*******
is the vector of parameter tensors.

var_vec
*******
This argument should (should not) be present when
agraph is a variable (parameter) acyclic graph.

return
******
The return is the tensor corresponding to this argument and operator usage.

{xrst_end base_op_static}
*/
#include <cassert>
#include <torch/torch.h>
//
#include <ad_tensor/adten.hpp>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/dev/base_op.hpp>
//
namespace ad_tensor { namespace dev {
    // ----------------------------------------------------------------------
    // BEGIN_TENSOR_AT_ARG_INDEX_PAR
    template<class TensorType>
    TensorType tensor_at_arg_index(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<TensorType>& par_vec   )
    // END_TENSOR_AT_ARG_INDEX_PAR
    {   size_t    index   = agraph.m_arg_value[arg_index];
        ad_type_t ad_type = agraph.m_arg_type[arg_index];
        switch( ad_type ) {
            //
            // constant
            case ad_type_t::constant:
            return TensorType( con_vec[index] );
            //
            // parameter
            case ad_type_t::parameter:
            return par_vec[index];
            //
            // default
            default:
            assert( false && "tensor_at_arg_index: "
                "expected a constant or parameter"
            );
        }
        // should not get here
        return par_vec[0];
    }
    template adten_t tensor_at_arg_index<adten_t>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<adten_t>&    par_vec
    );
    template at::Tensor tensor_at_arg_index<at::Tensor>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<at::Tensor>& par_vec
    );
    // ----------------------------------------------------------------------
    // BEGIN_TENSOR_AT_ARG_INDEX_VAR
    template<class TensorType>
    TensorType tensor_at_arg_index(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<TensorType>& par_vec   ,
        const vector<TensorType>& var_vec   )
    // END_TENSOR_AT_ARG_INDEX_VAR
    {   size_t    index   = agraph.m_arg_value[arg_index];
        ad_type_t ad_type = agraph.m_arg_type[arg_index];
        switch( ad_type ) {
            //
            // constant
            case ad_type_t::constant:
            return TensorType( con_vec[index] );
            //
            // parameter
            case ad_type_t::parameter:
            return par_vec[index];
            //
            // variable
            case ad_type_t::variable:
            return var_vec[index];
            //
            // default
            default:
            assert( false && "tensor_at_arg_index:"
                "expected a constant, parameter or variable"
            );
        }
        // should not get here
        return var_vec[0];
    }
    template adten_t tensor_at_arg_index<adten_t>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<adten_t>&    par_vec   ,
        const vector<adten_t>&    var_vec
    );
    template at::Tensor tensor_at_arg_index<at::Tensor>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<at::Tensor>& par_vec   ,
        const vector<at::Tensor>& var_vec
    );
} }
