// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin base_op_static dev}
{xrst_spell
    agraph
}

The base_op Static Functions
############################

tensor_at_arg_index
*******************

Prototype
=========
{xrst_literal ,
    include/ad_tensor/dev/base_op.hpp
    BEGIN_TENSOR_AT_ARG_INDEX, END_TENSOR_AT_ARG_INDEX
}

arg_index
=========
This is the index in agraph.m_arg_value and agraph.m_arg_type
for this operator usage.

agraph
======
is the acyclic graph that contains the information for this operator usage.

m_arg_type
----------
If var_vec is present, m_arg_type[arg_index] must be
constant, parameter, or variable.
If var_vec is not present, m_arg_type[arg_index] must be
constant or parameter.

con_vec
=======
is the vector of constant tensors.

par_vec
=======
is the vector of parameter tensors.

var_vec
=======
This argument should (should not) be present when
agraph is a variable (parameter) acyclic graph.

return
******
The return is the tensor corresponding to this argument and operator usage.

{xrst_end base_op_static}
*/
#include <cassert>
#include <ad_tensor/vector.hpp>
#include <torch/torch.h>
//
#include <ad_tensor/dev/base_op.hpp>
//
namespace ad_tensor { namespace dev {
    //
    const at::Tensor& base_op_t::tensor_at_arg_index(
        size_t                        arg_index ,
        const agraph_t&               agraph    ,
        const ad_tensor::vector<at::Tensor> con_vec   ,
        const ad_tensor::vector<at::Tensor> par_vec   )
    {   size_t    index   = agraph.m_arg_value[arg_index];
        ad_type_t ad_type = agraph.m_arg_type[arg_index];
        switch( ad_type ) {
            //
            // constant
            case ad_type_t::constant:
            return con_vec[index];
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
    const at::Tensor& base_op_t::tensor_at_arg_index(
        size_t                        arg_index ,
        const agraph_t&               agraph    ,
        const ad_tensor::vector<at::Tensor> con_vec   ,
        const ad_tensor::vector<at::Tensor> par_vec   ,
        const ad_tensor::vector<at::Tensor> var_vec   )
    {   size_t    index   = agraph.m_arg_value[arg_index];
        ad_type_t ad_type = agraph.m_arg_type[arg_index];
        switch( ad_type ) {
            //
            // constant
            case ad_type_t::constant:
            return con_vec[index];
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
} }
