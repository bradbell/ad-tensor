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

arg_start
=========
Let n_arg = arg_start[op_index+1] - arg_start[op_index].
For i_arg = 0, ... n_arg,
arg_index = arg_start[op_index] + i_arg is the index in arg_all and ad_type_all
of the corresponding argument and ad_type for this operator usage.

arg_all
=======
is a vector containing the arguments for all the operator uses.

ad_type_all
===========
is a vector containing the AD type for all the operator uses.
For each arg_index = arg[op_index] + i_arg,
if ad_type_all[arg_index] is constant (parameter) [variable],
val_index is an index in con_vec (par_vec) [var_vec].

con_vec
*******
is the vector containing all the constants.

par_vec
*******
is the vector containing all the parameters.

var_vec
*******
is the vector containing all the variables.

forward_par
***********
For this function, ad_type_all[arg_index] is not variable.
In addition, par_vec[index] for index < op_index is an input for this function
and par_vec[op_index] is an output.
{xrst_literal ,
    BEGIN_FORWARD_PAR, END_FORWARD_PAR
}

{xrst_end op_base}
*/
#include <vector>
#include <torch/torch.h>
//
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/ad_type.hpp>
//
// BEGIN_OP_BASE
namespace ad_tensor { namespace devel { class op_base_t
// END_OP_BASE
{
public:
    // BEGIN_OP_ENUM
    virtual op_enum_t op_enum(void) const = 0;
    // END_OP_ENUM
    //
    // BEGIN_FORWARD_PAR
    virtual void forward_par(
        bool                              trace       ,
        size_t                            op_index    ,
        const std::vector<size_t>&        arg_start   ,
        const std::vector<size_t>&        arg_all     ,
        const std::vector<ad_type_t>&     ad_type_all ,
        const std::vector<torch::Tensor>& con_vec     ,
        std::vector<torch::Tensor>&       par_vec     ) const = 0;
    // END_FORWARD_PAR

};
} }
