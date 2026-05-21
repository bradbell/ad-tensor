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

n_dom_
******
is the number of domain tensors in this tape.

op_vec_
*******
This vector has one element for each operation in the operation sequence.
The value op_vec_[op_index] is the :ref:`op_enum-name` for the
use of the operator corresponding to this operator use.

arg_start_
**********
Let n_arg = arg_start_[op_index+1] - arg_start_[op_index].
For i_arg = 0, ... n_arg,
arg_index = arg_start_[op_index] + i_arg is the index in arg_all_ and ad_type_all_
of the corresponding argument and ad_type for this operator usage.

arg_all_
********
is a vector containing the arguments for all the operator uses.

ad_type_all_
************
is a vector containing the AD type for all the operator uses;
see :ref:`ad_type-name` .
For each arg_index = arg[op_index] + i_arg,
if ad_type_all_[arg_index] is constant (parameter) [variable],
val_index is an index in con_vec (par_vec) [var_vec].

default constructor
*******************
This constructor creates an empty graph (all its vectors are empty).

swap
****
Exchange the contents between two acyclic graphs.

is_empty
********
This member function returns true if all the vectors in the graph are empty.

this_threads_tape
*****************
is the tape used to record AD operations on this thread.

{xrst_end agraph}
*/
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>

// BEGIN_AGRAPH
namespace ad_tensor { namespace devel { class agraph_t {
public:
    size_t                 n_dom_;
    std::vector<op_enum_t> op_vec_;
    std::vector<size_t>    arg_start_;
    std::vector<size_t>    arg_all_;
    std::vector<ad_type_t> ad_type_all_;
    //
    // default constructor
    agraph_t() : n_dom_(0), op_vec_(), arg_start_() , arg_all_(), ad_type_all_()
    { }
    //
    // swap
    void swap(agraph_t& other) noexcept
    {   std::swap( n_dom_, other.n_dom_ );
        op_vec_.swap(      other.op_vec_);
        arg_start_.swap(   other.arg_start_);
        arg_all_.swap(     other.arg_all_);
        ad_type_all_.swap( other.ad_type_all_);
    }
    //
    // is_empty
    bool is_empty(void) const {
        return
            op_vec_.empty() &&
            arg_start_.empty() &&
            arg_all_.empty() &&
            ad_type_all_.empty();
    }
}; } }
// END_AGRAPH
