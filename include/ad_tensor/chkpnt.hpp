#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin_parent chkpnt usr}

Checkpoint Functions
####################
Checkpoint functions are like atomic functions except
ad-tensor is used to compute the corresponding derivatives.
If a function is used many times, making it an checkpoint function
can greatly reduce the required memory
because only on copy of its parameters and variables will be required.

{xrst_toc_table
    src/adten/call_chkpnt.cpp
}

{xrst_end chkpnt}
------------------------------------------------------------------------------
{xrst_begin chkpnt_global usr}

Global Object That holds The Checkpoint Functions
#################################################

chkpnt_global
*************
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

chkpnt_info
===========
This chkpnt information will be moved to global data
and chkpnt_info will be empty upon return.

chkpnt_id
=========
is the identifier for this checkpoint function.

get_chkpnt_info
***************
{xrst_literal ,
    BEGIN_GET_INFO, END_GET_INFO
}
A call to get will wait until it can lock out any calls to store.


{xrst_end chkpnt_global}
*/
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
namespace ad_tensor  {
    // -----------------------------------------------------------------------
    // atom callbacks
    // -----------------------------------------------------------------------
    //
    // chkpnt_long_name
    std::string chkpnt_long_name(
        const options_t&                  options   ,
        size_t                            chkpnt_id
    );
    //
    // chkpnt_depend
    std::optional<sparsity_t> chkpnt_depend(
        const options_t&                  options   ,
        size_t                            chkpnt_id
    );
    //
    // chkpnt_forward
    std::optional< vector<at::Tensor> > chkpnt_forward (
        const options_t&                  options   ,
        size_t                            chkpnt_id ,
        const vector<bool>&               rng_used  ,
        const vector<at::Tensor>&         domain
    );
    // -----------------------------------------------------------------------
    // chkpnt_info_t
    class chkpnt_info_t {
    public:
        //
        // m_adfn, m_depend
        adfn_t     m_adfn;
        sparsity_t m_depend;
        //
        // from_adfn
        static chkpnt_info_t from_adfn(adfn_t& adfn);
    };
    // -----------------------------------------------------------------------
    // chkpnt_global_t
    class chkpnt_global_t {
    private:
        std::shared_mutex                        m_rw_mutex;
        vector< std::unique_ptr<chkpnt_info_t> > m_info_vec;
        size_t                                   m_atom_id;
        //
        // default constructor
        chkpnt_global_t(void);
    public:
        //
        // automatic copy constructor
        chkpnt_global_t(const chkpnt_global_t&) = delete;
        //
        //  automatic assignment operator
        void operator=(const chkpnt_global_t&) = delete;
        //
        // BEGIN_SINGLETON
        // chkpnt_global = chkpnt_global_t::singleton()
        static chkpnt_global_t& singleton(void);
        // END_SINGLETON
        //
        // BEGIN_STORE
        // chkpnt_id = store(chkpnt_info)
        size_t store(chkpnt_info_t& chkpnt_info);
        // END_STORE
        //
        // get_atom_id
        size_t get_atom_id(void) const;
        //
        // BEGIN_GET_INFO
        // chkpnt_info = get_chkpnt_info(chkpnt_id)
        const chkpnt_info_t& get_chkpnt_info(size_t chkpnt_id);
        // END_GET_INFO
    };
    // -----------------------------------------------------------------------
}
