// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_where usr}
{xrst_spell
    cond
}

Element-Wise Conditional Assignment
###################################

Syntax
******
{xrst_code cpp}
    result = where(cond, true_case, false_case)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_WHERE, END_WHERE
}

condition
*********
is an AD tensor with boolean scalar type.

true_case
*********
AD tensor with the values that are chosen for elements where cond is true.
It must have the same shape as cond.

false_case
**********
AD tensor with the values that are chosen for elements where cond is false.
It must have the same shape as cond.

result
******
is the result of the element-wise conditional assignment

    result = true_case if cond else false_case

Example
*******
{xrst_literal ,
    examples/adten/where.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_where}
-------------------------------------------------------------------------------
{xrst_begin adten_where_dev dev}
{xrst_spell
    cond
}

Element-wise Conditional Assignment Developer Documentation
###########################################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_STATIC_WHERE, END_STATIC_WHERE
    BEGIN_WHERE, END_WHERE
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, cond index,            cond AD type
    start + 1, true_case index,       true_case AD type
    start + 2, false_case index,      false_case AD type

where start is the length of arg_value and arg_type before this call to
``adten_t::where`` .

{xrst_end adten_where_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // Begin ad_tensor

// ad_tensor::adten_t::where
adten_t adten_t::where(
    const adten_t& cond,
    const adten_t& true_case,
    const adten_t& false_case ) {
    //
    // res_at_ten
    at::Tensor res_at_ten = at::where(
        cond.m_at_ten, true_case.m_at_ten, false_case.m_at_ten
    );
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_at_ten );
    dev::user_assert( cond.m_tape_id == tape.m_tape_id ,
        "AD tensor where: cond's tape is not tape that is recording"
    );
    dev::user_assert( true_case.m_tape_id == tape.m_tape_id ,
        "AD tensor where: true_case's tape is not tape that is recording"
    );
    dev::user_assert( false_case.m_tape_id == tape.m_tape_id ,
        "AD tensor where: false_case's tape is not tape that is recording"
    );
    //
    // res_vitype
    vitype_t res_vitype = cond.m_vitype;
    res_vitype           = std::max(res_vitype, true_case.m_vitype);
    res_vitype           = std::max(res_vitype, false_case.m_vitype);
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_vitype == vitype_t::constant ) {
        // res_index, tape.m_con_vec
        res_index = tape.m_con_vec.size();
        tape.m_con_vec.push_back( res_at_ten.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_vitype == vitype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_vitype == vitype_t::variable );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::where );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( cond.m_index );
        agraph->m_arg_type.push_back( cond.m_vitype );
        //
        agraph->m_arg_value.push_back( true_case.m_index );
        agraph->m_arg_type.push_back( true_case.m_vitype );
        //
        agraph->m_arg_value.push_back( false_case.m_index );
        agraph->m_arg_type.push_back( false_case.m_vitype );
    }
    return adten_t(res_tape_id, res_index, res_at_ten, res_vitype);
}
//
// ad_tensor::where
adten_t where(
    const adten_t& cond       ,
    const adten_t& true_case  ,
    const adten_t& false_case ) {
        return adten_t::where(cond, true_case, false_case);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
