#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent atom usr}

Atomic Functions
################

{xrst_toc_table
    src/adten/call_atom.cpp
    include/ad_tensor/base_atom.hpp
}

{xrst_end atom}
-------------------------------------------------------------------------------
{xrst_begin atom_dev dev}

Atomic Functions Developer Documentation
########################################

{xrst_toc_table
    src/adten/call_atom.cpp
}

{xrst_end atom_dev}
-------------------------------------------------------------------------------
{xrst_begin atom_global usr}

Global Object That holds All Atomic Callbacks
#############################################
This global object holds all the atomic functions.

atom_global
***********
{xrst_literal ,
    BEGIN_SINGLETON, END_SINGLETON
}

store
*****
{xrst_literal ,
    BEGIN_STORE, END_STORE
}
A call to store will wait until it can lock out any other calls to
store or get.

atom_callback
=============
is the callback information for this atomic function.

atom_id
=======
Is the identifier for this atomic function.

get_callback
************
A call to get will wait until it can lock out any calls to store.


{xrst_end atom_global}
*/
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/base_atom.hpp>
//
namespace ad_tensor {  // BEGIN_AD_TENSOR_NAMESPACE
//
// atom_global_t
class atom_global_t {
private:
    std::shared_mutex                          m_rw_mutex;
    vector< std::unique_ptr<base_atom_t> >     m_base_vec;
    //
    // default constructor
    atom_global_t(void)
    { }
public:
    //
    // automatic copy constructor
    atom_global_t(const atom_global_t&) = delete;
    //
    //  automatic assignment operator
    void operator=(const atom_global_t&) = delete;
    //
    // BEGIN_SINGLETON
    // atom_global = atom_global_t::singleton()
    static atom_global_t& singleton(void);
    // END_SINGLETON
    //
    // BEGIN_STORE
    // atom_id = atom_global.store(base_atom_ptr)
    size_t store( std::unique_ptr<base_atom_t>& base_atom_ptr );
    // END_STORE
    //
    // BEGIN_GET_BASE_ATOM
    const base_atom_t& get_base_atom(size_t atom_id);
    // END_GET_BASE_ATOM
};

} // END_AD_TENSOR_NAMESPACE
