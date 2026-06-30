#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
namespace ad_tensor  { namespace dev {
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
} }
