// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_bucketize usr}
{xrst_spell
}

Indices that Split an AD Tensor into Buckets
############################################

Syntax
******
{xrst_code cpp}
    result = bucketize(input, boundaries, out_int32, right))
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_BUCKETIZE, END_BUCKETIZE
}

input
*****
is the AD tensor that is being split into buckets.

boundaries
**********
a monotone increasing vector that maps indices to bucket boundaries.

out_int32
*********
if true (false) the indices result is int32_t (int64_t).
This argument is optional and its default value if false.

right
*****
If true (false) the each bucket includes (excludes) its upper limit
and excludes (includes) its lower limit.

result
******
The result has the same shape as input.
Let index be any is a valid index for the input tensor.

#.  If right is true,

    boundaries[result[index]-1] < input[index] <= boundaries[result[index]]

#.  If right is false,

    boundaries[result[index]-1] <= input[index] < boundaries[result[index]]


#.  If and element of input is too small (too large)
    to satisfy the conditions above, the corresponding element of
    result[index] is zero (boundaries.size(0)) .

Example
*******
{xrst_literal ,
    examples/adten/bucketize.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_bucketize}
-------------------------------------------------------------------------------
{xrst_begin adten_bucketize_dev dev}
{xrst_spell
}

Indices that Split an AD Tensor into Buckets Developer Documentation
####################################################################

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_DEV_BUCKETIZE, END_DEV_BUCKETIZE
}

Operation
*********
If this thread's tape is recording and the result is a parameter (variable)
the following is added to the parameter (variable) acyclic graph:

.. csv-table::
    :header-rows: 1

    arg_index, arg_value,                          arg_type
    start + 0, input index,                        input AD type
    start + 1, boundaries index,                   boundaries AD type
    start + 2, 0 (1) if out_int32 is false (true), none
    start + 3, 0 (1) if right is false (true),     none

bucketize start is the length of arg_value and arg_type before this call to
``adten_t::bucketize`` .

{xrst_end adten_bucketize_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // Begin ad_tensor

// ad_tensor::adten_t::bucketize
adten_t adten_t::bucketize(
    const adten_t& input,
    const adten_t& boundaries,
    bool           out_int32,
    bool           right ) {
    //
    // res_at_ten
    at::Tensor res_at_ten = at::bucketize(
        input.m_at_ten, boundaries.m_at_ten, out_int32, right
    );
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_at_ten );
    dev::user_assert( input.m_tape_id == tape.m_tape_id ,
        "AD tensor bucketize: input's tape is not tape that is recording"
    );
    dev::user_assert( boundaries.m_tape_id == tape.m_tape_id ,
        "AD tensor bucketize: boundaries's tape is not tape that is recording"
    );
    //
    // res_vitype
    vitype_t res_vitype = input.m_vitype;
    res_vitype          = std::max(res_vitype, boundaries.m_vitype);
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
        agraph->m_op_seq.push_back( dev::op_enum_t::bucketize );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( input.m_index );
        agraph->m_arg_type.push_back( input.m_vitype );
        //
        agraph->m_arg_value.push_back( boundaries.m_index );
        agraph->m_arg_type.push_back( boundaries.m_vitype );
        //
        agraph->m_arg_value.push_back( size_t(out_int32) );
        agraph->m_arg_type.push_back( vitype_t::none );
        //
        agraph->m_arg_value.push_back( size_t(right) );
        agraph->m_arg_type.push_back( vitype_t::none );
    }
    return adten_t(res_tape_id, res_index, res_at_ten, res_vitype);
}
//
// ad_tensor::bucketize
adten_t bucketize(
    const adten_t& input      ,
    const adten_t& boundaries ,
    bool           out_int32  ,
    bool           right     ) {
    return adten_t::bucketize(input, boundaries, out_int32, right);
}
// ---------------------------------------------------------------------------
} // End ad_tensor
