// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_index_put usr}
{xrst_spell
    numel
    elem
}

Replacing Some Elements of a Tensor
###################################

Syntax
******
{xrst_code cpp}
    after = before.index_put(index_vec, replace)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_INDEX_PUT, END_INDEX_PUT
}

before
******
is the initial tensor before replacing some element values
We use n_dim for the number of dimensions in before; i.e.

    n_dim = before.dim()


index_vec
*********
Is a vector with size n_dim that specifies where in before
the replacement values are assigned.
Each tensor in index_vec must be one dimensional and
have the same number of elements which we denote by n_elem.

replace
*******
is a one dimensional tensor that specifies the replacement values.
It has with n_elem elements; i.e.,
{xrst_code cpp}
    n_elem = replace.numel()
{xrst_code}

after
*****
This tensor has he same shape as the before tensor.
The value of after is described by the following pseudo code:
{xrst_code cpp}
    after = before;
    for(j = 0; j < n_elem; ++j) {
        after[ index[0][j], ... , index[n_dim-1][j] ] = replace[j]
    }
{xrst_code}

Example
*******
{xrst_literal ,
    examples/adten/index_put.cpp
    BEGIN_CPP, END_CPP
}
{xrst_end adten_index_put}
-------------------------------------------------------------------------------
{xrst_begin adten_index_put_dev dev}
{xrst_spell
}

Compute and Record an Element Replace Operation
###############################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_INDEX_PUT, END_DEV_INDEX_PUT
}
{xrst_literal ,
    include/ad_tensor/adten.hpp
    START_DEV_INDEX_PUT, STOP_DEV_INDEX_PUT
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
    start + 0, before tensor index,               before AD type
    start + 1, replace tensor index,              replace AD type
    start + 2, size of index_vec (n_dim)          none
    start + 3, index_vec[0] index,                index_vec[0] AD type
    ..., ..., ...
    start + 2 + n_dim, index_vec[n_dim-1] index,  index_vec[n_dim-1] AD type

where start is the length of arg_value and arg_type before this call to
``adten_t::binary``,

{xrst_end adten_index_put_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // Begin ad_tensor::dev
//
// BEGIN_INDEX_PUT  BEGIN_DEV_INDEX_PUT
adten_t adten_t::index_put(
    const vector<adten_t>& index_vec ,
    const adten_t&         replace
) const
{   // END_INDEX_PUT END_DEV_INDEX_PUT
    //
    // before
    const adten_t& before = *this;
    //
    // res_tensor
    c10::List< std::optional<at::Tensor> > index_list;
    for( const adten_t& index : index_vec ) {
        index_list.push_back( index.m_at_ten );
    }
    at::Tensor res_tensor =
        before.m_at_ten.index_put(index_list, replace.m_at_ten);
    //
    // n_dim
    size_t n_dim = size_t( before.dim() );
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    //
#ifndef NDEBUG
    dev::user_assert( 0 < n_dim ,
        "index: before.dim() is zero"
    );
    dev::user_assert( before.m_tape_id == tape.m_tape_id ,
        "index_put: Tape for before is not tape that is recording"
    );
    dev::user_assert( replace.m_tape_id == tape.m_tape_id ,
        "index_put: Tape for replace is not tape that is recording"
    );
    dev::user_assert( index_vec.size() == n_dim ,
        "index_put: size of index_vec not equal number of dimensions in before"
    );
    dev::user_assert( replace.dim() == 1,
        "index_put: replace is not one dimensional"
    );
    for(size_t i = 0; i < n_dim; ++i) {
        dev::user_assert( index_vec[i].dim() == 1,
            "index: a tensor in index_vec is not one dimensional"
        );
        dev::user_assert( index_vec[i].numel() == replace.numel() ,
            "index: one of the tensors in index_vec has a different "
            "number of elements than replace"
        );
    }
#endif
    //
    // res_vitype
    vitype_t res_vitype = std::max( before.m_vitype, replace.m_vitype );
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
        agraph->m_op_seq.push_back( dev::op_enum_t::index_put );
        agraph->m_arg_start.push_back( agraph->m_arg_value.size() );
        //
        // start + 0
        agraph->m_arg_value.push_back( before.m_index );
        agraph->m_arg_type.push_back( before.m_vitype );
        //
        // start + 1
        agraph->m_arg_value.push_back( replace.m_index );
        agraph->m_arg_type.push_back( replace.m_vitype );
        //
        // start + 2
        agraph->m_arg_value.push_back( n_dim );
        agraph->m_arg_type.push_back( vitype_t::none );
        //
        // start + 3 + i
        for (const adten_t& index : index_vec ) {
            agraph->m_arg_value.push_back( index.m_index );
            agraph->m_arg_type.push_back( index.m_vitype );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_vitype);
}
//
adten_t index_put(
    const adten_t&          before    ,
    const vector<adten_t>&  index_vec ,
    const adten_t&          replace   ) {
    return before.index( index_vec );
}
//
// Should we remove the need below to copy the index_vec buy having separate
// implementations of index for at::Tensor and adten_t in index_op.cpp ?
// This would eliminate the ad_tenor::index functions.
at::Tensor index_put(
    const at::Tensor&         before    ,
    const vector<at::Tensor>& index_vec ,
    const at::Tensor&         replace   ) {
    c10::List< std::optional<at::Tensor> > index_list;
    for( const at::Tensor& index : index_vec ) {
        index_list.push_back( index );
    }
    return before.index_put( index_list, replace );
};
// ---------------------------------------------------------------------------
} // End ad_tensor
