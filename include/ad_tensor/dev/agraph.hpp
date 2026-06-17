#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin agraph dev}
{xrst_spell
    seq
    int
    bool
}

Acyclic Graph Representation of an Operation Sequence
#####################################################
{xrst_literal ,
    BEGIN_AGRAPH, END_AGRAPH
}

m_dom_shapes
************
is a vector with size equal to the number of domain tensors
and each element is the shape of the corresponding domain tensor.

m_op_seq
********
This vector has one element for each operation in the operation sequence.
The value m_op_seq[op_index]
is the :ref:`op_enum-name` for this operator usage.

m_arg_start
***********
This vector has the same length as m_op_seq.
The number of arguments for the op_index operator usage is

    n_arg = m_arg_start[op_index+1] - m_arg_start[op_index].


m_arg_value
***********
For i_arg = 0, ... n_arg,

    m_arg_value[ m_arg_start[op_index] + i_arg ]

is he value of the i_arg argument for the op_index operator usage.


m_arg_type
**********
This vector has the same length as m_arg_value.

    m_arg_type[ m_arg_start[op_index] + i_arg ]

is the AD type for the i_arg argument for the op_index operator usage.

m_int64
*******
If an operation uses int64_t values, then there is an i_arg such that

    m_arg_value[ m_arg_start[op_index] + i_arg ]

is the start index in m_int64 of the int64_t values for this operation.

m_bool
******
If an operation uses bool values, then there is an i_arg such that

    m_arg_value[ m_arg_start[op_index] + i_arg ]

is the start index in m_bool of the bool values for this operation.


default constructor
*******************
This constructor creates an empty graph (all its vectors are empty).

swap
****
Exchange the contents between two acyclic graphs.

is_empty
********
This member function returns true if all the vectors in the graph are empty.

{xrst_end agraph}
*/
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/op_enum.hpp>

// BEGIN_AGRAPH
namespace ad_tensor { namespace dev { class agraph_t {
public:
    vector< vector<int64_t> >    m_dom_shapes;
    vector<op_enum_t>            m_op_seq;
    vector<size_t>               m_arg_start;
    vector<int64_t>              m_arg_value;
    vector<ad_type_t>            m_arg_type;
    vector<int64_t>              m_int64;
    vector<bool>                 m_bool;
    //
    // default constructor
    agraph_t()
    : m_dom_shapes()
    , m_op_seq()
    , m_arg_start()
    , m_arg_value()
    , m_arg_type()
    , m_int64()
    , m_bool()
    { }
    //
    // swap
    void swap(agraph_t& other) noexcept
    {   m_dom_shapes.swap(  other.m_dom_shapes);
        m_op_seq.swap(      other.m_op_seq);
        m_arg_start.swap(   other.m_arg_start);
        m_arg_value.swap(   other.m_arg_value);
        m_arg_type.swap(    other.m_arg_type);
        m_int64.swap(       other.m_int64);
        m_bool.swap(        other.m_bool);
    }
    //
    // is_empty
    bool is_empty(void) const {
        return
            m_dom_shapes.empty() &&
            m_op_seq.empty() &&
            m_arg_start.empty() &&
            m_arg_value.empty() &&
            m_arg_type.empty() &&
            m_int64.empty() &&
            m_bool.empty();
    }
}; } }
// END_AGRAPH
