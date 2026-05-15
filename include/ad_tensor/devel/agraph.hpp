#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin agraph dev}

Acyclic Graph Representation of an Operation Sequence
#####################################################
{xrst_literal ,
    BEGIN_AGRAPH, END_AGRAPH
}

op_vec
******
This vector has one element for each operation in the operation sequence.
The value op_vec[op_index] is the :ref:`op_enum-name` for the
use of the operator corresponding to this operator use.

arg_start
*********
Let n_arg = arg_start[op_index+1] - arg_start[op_index].
For i_arg = 0, ... n_arg,
arg_index = arg_start[op_index] + i_arg is the index in arg_all and ad_type_all
of the corresponding argument and ad_type for this operator usage.

arg_all
*******
is a vector containing the arguments for all the operator uses.

ad_type_all
***********
is a vector containing the AD type for all the operator uses;
see :ref:`ad_type-name` .
For each arg_index = arg[op_index] + i_arg,
if ad_type_all[arg_index] is constant (parameter) [variable],
val_index is an index in con_vec (par_vec) [var_vec].

default constructor
*******************
This constructor creates an empty graph (all its vectors are empty).

empty
*****
This member function returns true if all the vectors in the graph are empty.


{xrst_end agraph}
*/
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/ad_type.hpp>

// BEGIN_AGRAPH
namespace ad_tensor { namespace devel { struct agraph_t {
    std::vector<op_enum_t> op_vec;
    std::vector<size_t>    arg_start;
    std::vector<size_t>    arg_all;
    std::vector<ad_type_t> ad_type_all;
    //
    // default constructor
    agraph_t() : op_vec() , arg_start() , arg_all(), ad_type_all()
    { }
    //
    // empty
    bool empty(void) {
        return
            op_vec.empty() &&
            arg_start.empty() &&
            arg_all.empty() &&
            ad_type_all.empty();
    }
}; } }
// END_AGRAPH
