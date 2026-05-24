// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/ad.hpp>
#include <ad_tensor/devel/tape.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/agraph.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
// ----------------------------------------------------------------------------
ad_t::ad_t( at::Tensor&& tensor )
: m_tape_id(0)
, m_index(0)
, m_tensor(tensor)
, m_ad_type(ad_type_t::constant)
{   //
    // tape
    devel::tape_t& tape = devel::this_threads_tape();
    if( tape.m_recording ) {
        m_tape_id = tape.m_tape_id;
        m_index   = tape.m_con.size();
        tape.m_con.push_back( tensor.clone() );
    }
}
/*
-------------------------------------------------------------------------------
{xrst_begin ad_binary dev}
{xrst_spell
    lhs
    rhs
}

Compute and Record Binary Operators
###################################

Prototype
*********
{xrst_literal ,
    BEGIN_BINARY, END_BINARY
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

    Index, arg_all, ad_type_all
    start + 0, index for lhs, type for lhs
    start + 1, index for rhs, type for rhs

{xrst_end ad_binary}
*/
// BEGIN_BINARY
/* ares = binary( op_enum, lhs, rhs)
static */ ad_t ad_t::binary(
    devel::op_enum_t op_enum, const ad_t& lhs, const ad_t& rhs
)
// END_BINARY
{
    //
    // res_tensor
    at::Tensor res_tensor;
    switch(op_enum) {
        //
        // add
        case devel::op_enum_t::add:
        res_tensor = lhs.tensor() + rhs.tensor();
        break;
        //
        // sub
        case devel::op_enum_t::sub:
        res_tensor = lhs.tensor() - rhs.tensor();
        break;
        //
        // mul
        case devel::op_enum_t::mul:
        res_tensor = lhs.tensor() * rhs.tensor();
        break;
        //
        // div
        case devel::op_enum_t::div:
        res_tensor = lhs.tensor() / rhs.tensor();
        break;
        //
        default:
        assert( false && "ad_t::binary: invalid value for op_enum");
    }
    //
    // tape
    devel::tape_t& tape = devel::this_threads_tape();
    if( ! tape.m_recording )
        return ad_t( std::move(res_tensor) );
    assert( lhs.m_tape_id == tape.m_tape_id  &&
        "binary left operand does not match tape that is recording"
    );
    assert( rhs.m_tape_id == tape.m_tape_id  &&
        "binary right operand does not match tape that is recording"
    );
    //
    // res_ad_type
    ad_type_t res_ad_type = std::max( lhs.m_ad_type, rhs.m_ad_type );
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
        tape.m_con.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        devel::agraph_t* agraph = nullptr;
        if( res_ad_type == ad_type_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_ad_type == ad_type_t::variable  &&
                "binary opereand is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_vec.size();
        agraph->m_arg_start.push_back( agraph->m_arg_all.size() );
        //
        agraph->m_arg_all.push_back( lhs.m_index );
        agraph->m_ad_type_all.push_back( lhs.m_ad_type );
        //
        agraph->m_arg_all.push_back( lhs.m_index );
        agraph->m_ad_type_all.push_back( lhs.m_ad_type );
    }
    return ad_t(res_tape_id, res_index, std::move(res_tensor), res_ad_type);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
