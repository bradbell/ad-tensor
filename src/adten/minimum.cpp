// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin adten_minimum usr}
{xrst_spell
}

Element-wise Maximum
####################

Syntax
******
{xrst_code cpp}
    result = first.minimum(second)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_MINIMUM, END_MINIMUM
}

first
*****
is the first operand in the minimum.

second
******
is the second operand in the element-wise minimum.
Broadcasting is used to match the dimensions of first and second.

result
******
is the result of the element-wise minimum.


equality
********
The derivative does not exist for indices where the first and second tensors
are equal. The forward mode derivative of result is only specified to
be between derivative of first and second (for such indices).
In addition, the reverse mode derivative for first and second is between
zero and the reverse mode derivative of result (for such indices).

Example
*******
{xrst_literal ,
    examples/adten/minimum.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_minimum}
-------------------------------------------------------------------------------
{xrst_begin adten_minimum_dev dev}
{xrst_spell
}

Record Element-wise Maximum
###########################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_MINIMUM, END_DEV_MINIMUM
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

    arg_index, arg_value,       arg_type
    start + 0, first index,     first AD type
    start + 1, second index,    second AD type

where start is the length of arg_value and arg_type before this call to
``adten_t::binary`` .

{xrst_end adten_minimum_dev}
-------------------------------------------------------------------------------
*/
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // Begin ad_tensor::dev

// BEGIN_MINIMUM BEGIN_DEV_MINIMUM
adten_t adten_t::minimum(const adten_t& second) const
// END_MINIMUM END_DEV_MINIMUM
{
    //
    // res_tensor
    at::Tensor res_tensor = at::minimum(m_at_ten, second.m_at_ten);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    dev::user_assert( m_tape_id == tape.m_tape_id , "minimum: "
        "first AD tensor's tape is not tape that is recording"
    );
    dev::user_assert( second.m_tape_id == tape.m_tape_id , "minimum: "
        "second AD tensor's tape is not tape that is recording"
    );
    //
    // res_vitype
    vitype_t res_vitype = std::max( m_vitype, second.m_vitype );
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
        tape.m_con_vec.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_vitype == vitype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_vitype == vitype_t::variable  && "AD tensor being "
                "multiplied is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::minimum );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_vitype );
        //
        agraph->m_arg_value.push_back( second.m_index );
        agraph->m_arg_type.push_back( second.m_vitype );
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_vitype);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
