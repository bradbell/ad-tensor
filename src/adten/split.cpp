// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/defined.hpp>
//
namespace ad_tensor { // Begin ad_tensor::dev
/*
-------------------------------------------------------------------------------
{xrst_begin adten_split usr}
{xrst_spell
}

Splitting an AD Tensor
######################

Syntax
******
{xrst_code cpp}
    output = input.split(split_size, dim)
{xrst_code}

Prototype
*********
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_SPLIT, END_SPLIT
}

input
*****
is the tensor that we are splitting.

split_size
**********
is the sizes of the output tensors in the dim dimension.

dim
***
is the dimension that is being split.

output
******
is the vector of tensors containing the result of the splitting.

Example
*******
{xrst_literal ,
    examples/adten/split.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_split}
-------------------------------------------------------------------------------
{xrst_begin adten_split_dev dev}
{xrst_spell
    op
    seq
    enum
}

Splitting an AD Tensor Developer Documentation
##############################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_SPLIT, END_DEV_SPLIT
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

    arg_index, arg_value,                                     arg_type
    start + 0, index for operand,                             operand type
    start + 1, dim,                                           none
    start + 1, number of split sizes (n_split),               none
    start + 2, size of first piece in dim direction,          none
    ..., ..., ...
    start + n_split + 1, size of last piece in dim direction, none

where start is the length of arg_value and arg_type before this call to
``adten_t::binary`` .
If n_split must be greater than zero.

op_seq
******
A split has n_split operations in the tape so that operator
indices are the same as result indices.
The first operation is an ``op_enum_t::split`` and the next
n_split - 1 operations are ``op_enum_t::split_result``.
Only the first operation has entries in arg_value and arg_type
(so start for split_result operators is the end of the corresponding split).

{xrst_end adten_split_dev}
*/
// BEGIN_DEV_SPLIT
vector<adten_t> adten_t::split(const c10::IntArrayRef& split_size, int64_t dim) const
// END_DEV_SPLIT
{
    //
    // n_split
    size_t n_split = split_size.size();
    //
    // ad_vec;
    vector<at::Tensor> at_vec = m_at_ten.split(split_size, dim);
    vector<adten_t>    ad_vec( n_split );
    for(size_t i = 0; i < at_vec.size(); ++i) {
        ad_vec[i] = adten_t( at_vec[i] );
    }
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording ) {
        return ad_vec;
    }
    dev::user_assert( m_tape_id == tape.m_tape_id ,
        "Tape for AD tensor being splitmed is not tape that is recording"
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
        //
        // res_index, tape.m_con_vec
        res_index = tape.m_con_vec.size();
        for(size_t i = 0; i < n_split; ++i) {
            tape.m_con_vec.push_back( ad_vec[i].m_at_ten.clone() );
        }
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "AD tensor being "
                "split is not constant, parameter, or variable"
            );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::split );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        agraph->m_arg_value.push_back( m_index );
        agraph->m_arg_type.push_back( m_adtype );
        //
        agraph->m_arg_value.push_back( size_t(dim) );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        agraph->m_arg_value.push_back( n_split );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        for(size_t i = 0; i < n_split; ++i) {
            dev::user_assert( 0 <= dim,
                "AD Tensor split: a dimension index is less than zero"
            );
            agraph->m_arg_value.push_back( size_t( split_size[i] ) );
            agraph->m_arg_type.push_back( adtype_t::none );
        }
        //
        // agraph: m_op_seq, m_arg_start
        for(size_t i = 1; i < n_split; ++i) {
            agraph->m_op_seq.push_back( dev::op_enum_t::split_result );
            agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        }
    }
    for(size_t i = 0; i < n_split; ++i) {
        ad_vec[i] = adten_t(res_tape_id, res_index + i, at_vec[i], res_adtype);
    }
    return ad_vec;
}
// ---------------------------------------------------------------------------
} // End ad_tensor
