#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin op_base dev}

The Operator Base Class
#######################

Prototype
*********
{xrst_literal ,
    BEGIN_OP_BASE, END_OP_BASE
}

Derived Classes
***************
All the member functions for this class are public and virtual.
Their member functions are like static functions
that are grouped together by operator; i.e.,
they have no member data and are const.

op_enum
*******
This function returns the :ref:`op_enum-name` value
for the corresponding operator.
{xrst_literal ,
    BEGIN_OP_ENUM, END_OP_ENUM
}

Common Arguments
****************
For each argument below, its documentation is common for
all the member functions where the corresponding argument appears:

op_index
========
is the index of this operator in the operation sequence.
It is also the index of the result for this operator.
It must be greater than zero, because the first result is the
independent parameter tensor.

agraph
======
Is the :ref:`agraph-name` for the operation sequence that we are using.

con_vec
*******
is the vector containing all the constant tensors.

par_vec
*******
is the vector containing all the parameter tensors.

var_vec
*******
is the vector containing all the variable tensors..

tensor_at_index
***************
This returns a reference to the tensor corresponding to
the specified type and index.
This is a reference to an element in the corresponding vector
and hence its lifetime is restricted to the life of the vector.
{xrst_literal ,
    BEGIN_TENSOR_AT_INDEX, END_TENSOR_AT_INDEX
    BEGIN_WITH_VAR_TENSOR_AT_INDEX, END_WITH_VAR_TENSOR_AT_INDEX
}

forward_par
***********
For this function, ad_type_all[arg_index] is never variable.
In addition, par_vec[index] for index < op_index is an input for this function
and par_vec[op_index] is an output.
{xrst_literal ,
    BEGIN_FORWARD_PAR, END_FORWARD_PAR
}

{xrst_end op_base}
*/
#include <cassert>
#include <vector>
#include <torch/torch.h>
//
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/ad_type.hpp>
#include <ad_tensor/devel/agraph.hpp>
//
// BEGIN_OP_BASE
namespace ad_tensor { namespace devel { struct op_base_t
// END_OP_BASE
{
    // BEGIN_OP_ENUM
    virtual op_enum_t op_enum(void) const = 0;
    // END_OP_ENUM
    //
    // BEGIN_TENSOR_AT_INDEX
    static inline const at::Tensor& tensor_at_index(
        ad_type_t                        ad_type ,
        size_t                           index   ,
        const std::vector<at::Tensor> con_vec ,
        const std::vector<at::Tensor> par_vec )
    // END_TENSOR_AT_INDEX
    {   switch( ad_type ) {
            //
            // constant
            case ad_type_t::constant:
            return con_vec.at(index);
            //
            // parameter
            case ad_type_t::parameter:
            return par_vec.at(index);
            //
            // default
            default:
            assert( false && "expected a constant or parameter" );
        }
        // should not get here
        return par_vec.at(0);
    }
    //
    // BEGIN_WITH_VAR_TENSOR_AT_INDEX
    static inline const at::Tensor& tensor_at_index(
        ad_type_t                        ad_type ,
        size_t                           index   ,
        const std::vector<at::Tensor>    con_vec ,
        const std::vector<at::Tensor>    par_vec ,
        const std::vector<at::Tensor>    var_vec )
    // END_WITH_VAR_TENSOR_AT_INDEX
    {   switch( ad_type ) {
            //
            // constant
            case ad_type_t::constant:
            return con_vec.at(index);
            //
            // parameter
            case ad_type_t::parameter:
            return par_vec.at(index);
            //
            // variable
            case ad_type_t::variable:
            return var_vec.at(index);
            //
            // default
            default:
            assert( false && "expected a constant, parameter or variable" );
        }
        // should not get here
        return var_vec.at(0);
    }
    //
    // BEGIN_FORWARD_PAR
    virtual void forward_par(
        bool                              trace       ,
        size_t                            op_index    ,
        const agraph_t&                   agraph      ,
        const std::vector<at::Tensor>&    con_vec     ,
        std::vector<at::Tensor>&          par_vec     ) const = 0;
    // END_FORWARD_PAR

};
} }
