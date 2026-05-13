#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin agraph dev}

An Operation Sequence Acyclic Graph
###################################
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


{xrst_end agraph}
*/
#include <ad_tensor/devel/ad_type.hpp>

// BEGIN_AGRAPH
namespace ad_tensor { namespace devel { class agraph_t {
public:
    std::vector<op_enum_t> op_vec;
    std::vector<size_t>    arg_start;
    std::vector<size_t>    arg_all;
    std::vector<ad_type_t> ad_type_all;
    //
    // ctor
    agraph_t() :
        op_vec() ,
        arg_start() ,
        arg_all(),
        ad_type_all()
    { }
}; } }
// END_AGRAPH
