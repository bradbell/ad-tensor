// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin shape_at_index dev}
{xrst_spell
    agraph
}

Get Shape Corresponding to an Operator Argument Index
#####################################################

Prototype
*********
{xrst_literal ,
    BEGIN_SHAPE_AT_ARG_INDEX_PAR, END_SHAPE_AT_ARG_INDEX_PAR
    BEGIN_SHAPE_AT_ARG_INDEX_VAR, END_SHAPE_AT_ARG_INDEX_VAR
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
If var_all is present, m_arg_type[arg_index] must be
constant, parameter, or variable.
If var_all is not present, m_arg_type[arg_index] must be
constant or parameter.

con_vec
*******
is the vector of constant tensors.

par_all
*******
is the vector of parameter tensors.

var_all
*******
This argument should (should not) be present when
agraph is a variable (parameter) acyclic graph.

return
******
The return is the shape of the tensor corresponding to this
argument and operator usage.

{xrst_end shape_at_index}
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
    // BEGIN_SHAPE_AT_ARG_INDEX_PAR
    template<class TensorType>
    c10::IntArrayRef shape_at_arg_index(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<TensorType>& par_all   )
    // END_SHAPE_AT_ARG_INDEX_PAR
    {   size_t    index   = agraph.m_arg_value[arg_index];
        vitype_t vitype = agraph.m_arg_type[arg_index];
        switch( vitype ) {
            //
            // constant
            case vitype_t::constant:
            return con_vec[index].sizes();
            //
            // parameter
            case vitype_t::parameter:
            return par_all[index].sizes();
            //
            // default
            default:
            assert( false && "shape_at_arg_index: "
                "expected a constant or parameter"
            );
        }
        // should not get here
        return par_all[0].sizes();
    }
    template c10::IntArrayRef shape_at_arg_index<adten_t>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<adten_t>&    par_all
    );
    template c10::IntArrayRef shape_at_arg_index<at::Tensor>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<at::Tensor>& par_all
    );
    // ----------------------------------------------------------------------
    // BEGIN_SHAPE_AT_ARG_INDEX_VAR
    template<class TensorType>
    c10::IntArrayRef shape_at_arg_index(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<TensorType>& par_all   ,
        const vector<TensorType>& var_all   )
    // END_SHAPE_AT_ARG_INDEX_VAR
    {   size_t    index   = agraph.m_arg_value[arg_index];
        vitype_t vitype = agraph.m_arg_type[arg_index];
        switch( vitype ) {
            //
            // constant
            case vitype_t::constant:
            return con_vec[index].sizes();

            //
            // parameter
            case vitype_t::parameter:
            return par_all[index].sizes();
            //
            // variable
            case vitype_t::variable:
            return var_all[index].sizes();
            //
            // default
            default:
            assert( false && "shape_at_arg_index:"
                "expected a constant, parameter or variable"
            );
        }
        // should not get here
        return var_all[0].sizes();
    }
    template c10::IntArrayRef shape_at_arg_index<adten_t>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<adten_t>&    par_all   ,
        const vector<adten_t>&    var_all
    );
    template c10::IntArrayRef shape_at_arg_index<at::Tensor>(
        size_t                    arg_index ,
        const agraph_t&           agraph    ,
        const vector<at::Tensor>& con_vec   ,
        const vector<at::Tensor>& par_all   ,
        const vector<at::Tensor>& var_all
    );
} }
