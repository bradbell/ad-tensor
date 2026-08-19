// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adten_index_put usr}
{xrst_spell
}

Replacing Some Elements of a Tensor
###################################

Syntax
******
{xrst_code cpp}
    after = before.index_put(index_list, replace)
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_INDEX_PUT, END_INDEX_PUT
}

before
******
is the initial tensor before replacing some element values
We use m for the number of dimensions in before; i.e.

    m = before.sizes().size();

index_list
**********
Is the list of indices that specify where in before the replacement values
assigned to the corresponding elements.
There must be m tensors in index_list and
each tensor in this list must be one dimensional and
have the same number of elements as replace.

replace
*******
is a one dimensional tensor of replacement values.
We use n for the number of elements in replace; i.e.

    n = replace.numel()

after
*****
This tensor has he same shape as the before tensor.
Let index_0, ... , index_m1 denote corresponding elements of index_list.
The value of after is described by the following pseudo code:
{xrst_code cpp}
    after = before;
    for(j = 0; j < n; ++j) {
        after[ index_0[j], ... , index_m1[j] ] = replace[j]
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
    idx
}

Compute and Record an Element Replace Operation
###############################################

Prototype
*********
{xrst_literal ,
    BEGIN_DEV_INDEX_PUT, END_DEV_INDEX_PUT
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
    start + 0, index for before tensor,                   type of before
    start + 1, index for replace tensor,                  type if replace
    start + 2, number of indices in index_list  (n_idx),  adtype_t::none
    start + 3, index  of tensor index_list[0]             adtype_t::constant
    ..., ..., ...
    start + 2 + n_idx, index of index_list[n_idx-1]       adtype_t::constant

where start is the length of arg_value and arg_type before this call to
``adten_t::binary``,
If n_dim is zero, all the dimensions are summed.


{xrst_end adten_index_put_dev}
*/
#include <ad_tensor/adten.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
//
// BEGIN_INDEX_PUT  BEGIN_DEV_INDEX_PUT
adten_t adten_t::index_put(
        const c10::List< std::optional<at::Tensor> >& index_list ,
        const adten_t&                                replace
    ) const
{   // END_INDEX_PUT END_DEV_INDEX_PUT
    //
    // before
    const adten_t& before = *this;
    //
    // res_tensor
    at::Tensor res_tensor =
        before.m_at_ten.index_put(index_list, replace.m_at_ten);
    //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    if( ! tape.m_recording )
        return adten_t( res_tensor );
    //
    dev::user_assert( before.m_tape_id == tape.m_tape_id ,
        "index_put: Tape for before is not tape that is recording"
    );
    dev::user_assert( replace.m_tape_id == tape.m_tape_id ,
        "index_put: Tape for replace is not tape that is recording"
    );
    dev::user_assert( index_list.size() == before.sizes().size(),
        "index_put: index_list.size() not equal number of dimensions in before"
    );
    dev::user_assert( replace.sizes().size() == 1,
        "index_put: replace is not one dimensional"
    );
    //
#ifndef NDEBUG
    int64_t n_replace = replace.numel();
#endif
    //
    // res_adtype
    adtype_t res_adtype = std::max( before.m_adtype, replace.m_adtype );
    //
    // res_tape_id
    size_t res_tape_id = tape.m_tape_id;
    //
    // res_index
    size_t res_index;
    //
    if( res_adtype == adtype_t::constant ) {
        // res_index, tape.m_con
        res_index = tape.m_con.size();
        tape.m_con.push_back( res_tensor.clone() );
    } else {
        //
        // agraph
        dev::agraph_t* agraph = nullptr;
        if( res_adtype == adtype_t::parameter )
            agraph = &tape.m_par;
        else {
            assert( res_adtype == adtype_t::variable  && "index_put: "
                "before or replace is not constant, parameter, or variable"
            );
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
        agraph->m_arg_type.push_back( before.m_adtype );
        //
        // start + 1
        agraph->m_arg_value.push_back( replace.m_index );
        agraph->m_arg_type.push_back( replace.m_adtype );
        //
        // start + 2
        size_t n_idx = index_list.size();
        agraph->m_arg_value.push_back( n_idx );
        agraph->m_arg_type.push_back( adtype_t::none );
        //
        // start + 3 + i
        for (const std::optional<at::Tensor>& index : index_list) {
            dev::user_assert( index.value().sizes().size() == 1,
                "index_put: one of the index tensors is not one dimensional"
            );
            dev::user_assert( index.value().numel() == n_replace,
                "index_put: one of the index tensors does not have the same "
                "number of elements as replace"
            );
            size_t con_index  = tape.m_con.size();
            tape.m_con.push_back( index.value().clone() );
            agraph->m_arg_value.push_back( con_index );
            agraph->m_arg_type.push_back( adtype_t::constant );
        }
    }
    return adten_t(res_tape_id, res_index, res_tensor, res_adtype);
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
