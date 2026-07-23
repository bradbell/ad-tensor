// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_where usr}
{xrst_spell
    cond
}

Element-Wise Conditional Assignment
###################################

Syntax
******
{xrst_code cpp}
    result = adten_t::where(cond, true_case, false_case)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_WHERE, END_WHERE
}

condition
*********
is a boolean at::Tensor.

true_case
*********
tensor with the values that are chosen for elements where cond is true.
It must have the same shape as cond.

false_case
**********
tensor with the values that are chosen for elements where cond is false.
It must have the same shape as cond.

result
******
is the result of the element-wise conditional assignment

    result = true_case if cond else false_case

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
    BEGIN_WHERE, END_WHERE
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value, arg_type
    start + 0, index for cond,        type of the cond
    start + 1, index for true_case,   type of true_case
    start + 2, index for false_case,  type of false_case

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
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
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
        "AD tensor where: cond tape is not tape that is recording"
    );
    dev::user_assert( true_case.m_tape_id == tape.m_tape_id ,
        "AD tensor where: true_case tape does is not tape that is recording"
    );
    dev::user_assert( false_case.m_tape_id == tape.m_tape_id ,
        "AD tensor where: false_case tape does is not tape that is recording"
    );
    //
    // res_ad_type
    ad_type_t res_ad_type = cond.m_ad_type;
    res_ad_type           = std::max(res_ad_type, true_case.m_ad_type);
    res_ad_type           = std::max(res_ad_type, false_case.m_ad_type);
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_ad_type == ad_type_t::constant ) {
        // res_index, tape.m_con
        res_index = tape.m_con.size();
        tape.m_con.push_back( res_at_ten.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_ad_type == ad_type_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_ad_type == ad_type_t::variable );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::where );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( cond.m_index );
        agraph->m_arg_type.push_back( cond.m_ad_type );
        //
        agraph->m_arg_value.push_back( true_case.m_index );
        agraph->m_arg_type.push_back( true_case.m_ad_type );
        //
        agraph->m_arg_value.push_back( false_case.m_index );
        agraph->m_arg_type.push_back( false_case.m_ad_type );
    }
    return adten_t(res_tape_id, res_index, res_at_ten, res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
