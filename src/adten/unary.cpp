// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
/*
-------------------------------------------------------------------------------
{xrst_begin adten_unary_dev dev}

Compute and Record Unary Operators
##################################

Prototype
*********
{xrst_literal ,
    BEGIN_UNARY, END_UNARY
}

Recording
*********
If this thread's tape is recording, and the result is (is not) a constant,
the constant is added to the tape (the operation is added to the tape).

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value,            arg_type
    start + 0, index for operand,    type of operand

where start is the length of arg_value and arg_type before this call to
``adten_t::unary`` .

{xrst_end adten_unary_dev}
*/
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
// BEGIN_UNARY
adten_t adten_t::unary( dev::op_enum_t op_enum, const adten_t& operand )
// END_UNARY
{
    //
    // res_at_ten
    at::Tensor res_at_ten;
    switch(op_enum) {
        //
        // exp
        case dev::op_enum_t::exp:
        res_at_ten = operand.at_ten().exp();
        break;
        //
        // inverse
        case dev::op_enum_t::inverse:
        res_at_ten = operand.at_ten().inverse();
        break;
        //
        default:
        assert( false && "adten_t::unary: invalid value for op_enum");
    }
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_at_ten );
    dev::user_assert( operand.m_tape_id == tape.m_tape_id ,
        "unary operand's tape is not tape that is recording"
    );
    //
    // res_ad_type
    ad_type_t res_ad_type = operand.m_ad_type;
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if(res_ad_type == ad_type_t::constant ) {
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
            assert( res_ad_type == ad_type_t::variable  &&
                "unary operand is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( op_enum);
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( operand.m_index );
        agraph->m_arg_type.push_back( operand.m_ad_type );
    }
    return adten_t(res_tape_id, res_index, res_at_ten, res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
