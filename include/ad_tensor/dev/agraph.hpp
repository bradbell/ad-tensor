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

m_op_vec
********
This vector has one element for each operation in the operation sequence.
The value m_op_vec[op_index] is the :ref:`op_enum-name` for the
use of the operator corresponding to this operator use.

m_arg_start
***********
Let n_arg = m_arg_start[op_index+1] - m_arg_start[op_index].
For i_arg = 0, ... n_arg,
arg_index = m_arg_start[op_index] + i_arg
is the index in m_arg_all and m_ad_type_all
of the corresponding argument and ad_type for this operator usage.

m_arg_all
*********
is a vector containing the arguments for all the operator uses.
For each arg_index = m_arg_start[op_index] + i_arg,
m_arg_all[arg_index] is the index for the corresponding argument.

m_ad_type_all
*************
is a vector containing the AD type for all the operator uses;
see :ref:`ad_type-name` .
For each arg_index = m_arg_start[op_index] + i_arg,
if m_ad_type_all[arg_index] is constant (parameter) [variable],
the corresponding index is in con_vec (par_vec) [var_vec].

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
#include <ad_tensor/dev/op_enum.hpp>

// BEGIN_AGRAPH
namespace ad_tensor { namespace devel { class agraph_t {
public:
    size_t                 n_dom_;
    std::vector<op_enum_t> m_op_vec;
    std::vector<size_t>    m_arg_start;
    std::vector<size_t>    m_arg_all;
    std::vector<ad_type_t> m_ad_type_all;
    //
    // default constructor
    agraph_t() : n_dom_(0), m_op_vec(), m_arg_start() , m_arg_all(), m_ad_type_all()
    { }
    //
    // swap
    void swap(agraph_t& other) noexcept
    {   std::swap( n_dom_, other.n_dom_ );
        m_op_vec.swap(      other.m_op_vec);
        m_arg_start.swap(   other.m_arg_start);
        m_arg_all.swap(     other.m_arg_all);
        m_ad_type_all.swap( other.m_ad_type_all);
    }
    //
    // is_empty
    bool is_empty(void) const {
        return
            m_op_vec.empty() &&
            m_arg_start.empty() &&
            m_arg_all.empty() &&
            m_ad_type_all.empty();
    }
}; } }
// END_AGRAPH
