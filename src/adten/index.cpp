// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_index usr}
{xrst_spell
    elem
    numel
}

Extracting Some Elements from a Tensor
######################################

Syntax
******
{xrst_code cpp}
    extract = from.index(index_vec)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_INDEX, END_INDEX
}

from
****
is the tensor we are extracting elements from.
We use n_dim for the number of dimensions in this tensor; i.e.

    n_dim = from.dim()

This must be greater than zero.

index_vec
*********
Is a vector with size n_dim that specifies which elements are extracted.
Each tensor in index_vec must be one dimensional and
have the same number of elements which we denote by n_elem.

extract
*******
is the size n_elem one dimensional tensor of extracted values; i.e.,
{xrst_code cpp}
    n_elem = extract.numel()
{xrst_code}
The value of extract is described by the following pseudo code:
{xrst_code cpp}
    for(j = 0; j < n_elem; ++j) {
        extract[j] = from[ index_vec[0][j], ... , index_vec[n_dim-1][j] ]
    }
{xrst_code}

Example
*******
{xrst_literal ,
    examples/adten/index.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_index}
-------------------------------------------------------------------------------
{xrst_begin adten_index_dev dev}
{xrst_spell
}

Compute and Record an Element Extract Operation
###############################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEVELOPER_INDEX, END_DEVELOPER_INDEX
}
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_DEV_INDEX, END_DEV_INDEX
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
    start + 0, from tensor index,                from AD type
    start + 1, size of index_vec (n_dim),        none
    start + 2, index_vec[0] index,               index_vec[0] AD type
    ..., ..., ...
    start + 1 + n_dim, index_list[n_dim-1] index, index_vec[n_dim-1] AD type

where start is the length of arg_value and arg_type before this call to
``adten_t::binary``,

{xrst_end adten_index_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // Begin ad_tensor::dev
//
// BEGIN_INDEX BEGIN_DEVELOPER_INDEX
adten_t adten_t::index(
    const vector<adten_t>& index_vec
) const
{   // END_INDEX END_DEVELOPER_INDEX
    //
    // before
    const adten_t& from = *this;
    //
    // res_tensor
    vector<at::indexing::TensorIndex> at_index_vec;
    for( const adten_t& index : index_vec ) {
        at_index_vec.push_back( index.m_at_ten );
    }
    at::Tensor res_tensor = from.m_at_ten.index( at_index_vec );
    //
    // n_dim
    size_t n_dim = size_t( from.dim() );
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    //
#ifndef NDEBUG
    dev::user_assert( 0 < n_dim ,
        "index: from.dim() is zero"
    );
    dev::user_assert( from.m_tape_id == tape.m_tape_id ,
        "index: Tape for from is not tape that is recording"
    );
    dev::user_assert( index_vec.size() == n_dim ,
        "index: size of index_vec not equal number of dimensions in from"
    );
    for(size_t i = 0; i < n_dim; ++i) {
        dev::user_assert( index_vec[i].dim() == 1,
            "index: a tensor in index_vec is not one dimensional"
        );
        dev::user_assert( index_vec[i].numel() == index_vec[0].numel() ,
            "index: not all tensors in index_vec have same number of elements"
        );
    }
# endif
    //
    // res_vitype
    vitype_t res_vitype = from.m_vitype;
    for( const adten_t& index : index_vec ) {
        res_vitype = std::max( res_vitype, index.m_vitype );
    }
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
            assert( res_vitype == vitype_t::variable );
            agraph = &tape.m_var;
        }
        //
        // res_index, agraph
        res_index       = agraph->m_op_seq.size();
        agraph->m_op_seq.push_back( dev::op_enum_t::index );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        // start + 0
        agraph->m_arg_value.push_back( from.m_index );
        agraph->m_arg_type.push_back( from.m_vitype );
        //
        // start + 1
        agraph->m_arg_value.push_back( n_dim );
        agraph->m_arg_type.push_back( vitype_t::none );
        //
        // start + 2 + i
        for (const adten_t& index : index_vec ) {
            agraph->m_arg_value.push_back( index.m_index );
            agraph->m_arg_type.push_back( index.m_vitype );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_vitype);
}
//
adten_t index(
    const adten_t&          from      ,
    const vector<adten_t>&  index_vec ) {
    return from.index( index_vec );
}
//
// Should we remove the need below to copy the index_vec buy having separate
// implementations of index for at::Tensor and adten_t in index_op.cpp ?
// This would eliminate the ad_tenor::index functions.
at::Tensor index(
    const at::Tensor&         from      ,
    const vector<at::Tensor>& index_vec ) {
    vector<at::indexing::TensorIndex> at_index_vec;
    for( const at::Tensor& index : index_vec ) {
        at_index_vec.push_back( index );
    }
    return from.index( at_index_vec );
};

// ---------------------------------------------------------------------------
} // End ad_tensor
