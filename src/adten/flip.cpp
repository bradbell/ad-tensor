// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/no_elements.hpp>
//
namespace ad_tensor { // Begin ad_tensor::dev
/*
-------------------------------------------------------------------------------
{xrst_begin adten_flip usr}
{xrst_spell
    aflip
}

The Flip Function
#################

Prototype
*********
{xrst_literal ,
    BEGIN_FLIP, END_FLIP
}

adten
*****
is the tensor that we are flipping.

dim
***
is a vector containing the dimensions we are flipping.

aflip
*****
is the result of the flip.

Example
*******
{xrst_literal ,
    examples/adten/flip.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_flip}
-------------------------------------------------------------------------------
{xrst_begin adten_flip_dev dev}
{xrst_spell
    aflip
}

Compute and Record Flip Function
################################

Syntax
******
{xrst_code cpp}
    aflip = adten.flip(dim)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_FLIP, END_DEV_FLIP
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

    arg_index, arg_value, arg_type
    start + 0, index for operand, type for operand
    start + 1, number of dimensions being flipped (n_dim),    adtype_t::none
    start + 2, index of first dimension being flipped,        adtype_t::none
    ..., ..., ...
    start + n_dim + 1, index of last dimension being flipped, adtype::none

where start is the length of arg_value and arg_type before this call to
``adten_t::binary`` .


{xrst_end adten_flip_dev}
*/
// BEGIN_FLIP  BEGIN_DEV_FLIP
adten_t adten_t::flip(const c10::IntArrayRef& dim) const
// END_FLIP END_DEV_FLIP
{
    //
    // res_tensor
    at::Tensor res_tensor = m_at_ten.flip(dim);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "Tape for AD tensor being flipped is not tape that is recording"
    );
    //
    // res_adtype
    adtype_t res_adtype = m_adtype;
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_adtype == adtype_t::constant ) {
        // res_index, tape.m_con_vec
        res_index = tape.m_con_vec.size();
        tape.m_con_vec.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "AD tensor being "
                "flipped is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::flip );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_adtype );
        //
        size_t n_dim = dim.size();
        agraph->m_arg_value.push_back( n_dim );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        for(size_t i = 0; i < n_dim; ++i) {
            dev::user_assert( 0 <= dim[i],
                "AD Tensor flip: a dimension index is less than zero"
            );
            agraph->m_arg_value.push_back( size_t( dim[i] ) );
            agraph->m_arg_type.push_back( adtype_t::none );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
