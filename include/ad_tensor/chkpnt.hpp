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
        static chkpnt_info_t from_adfn(adfn_t& adfn) {
            chkpnt_info_t info;
            //
            auto [depend_par, depend_var] = adfn.forward_dep();
            dev::move_swap( depend_var, info.m_depend );
            dev::move_swap( adfn,       info.m_adfn );
            return info;
        }
        //
        // m_adfn, m_depend
        adfn_t     m_adfn;
        sparsity_t m_depend;
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
        chkpnt_global_t(void)
        {   //
            // atom_global
            atom_global_t& atom_global = atom_global_t::singleton();

            // atom_callback
            atom_callback_t atom_callback;
            atom_callback.set_name("chkpnt_callback");
            atom_callback.set_long_name(chkpnt_long_name);
            atom_callback.set_depend(chkpnt_depend);
            atom_callback.set_forward(chkpnt_forward);
            //
            // m_atom_id
            m_atom_id = atom_global.store( atom_callback );
        }
    public:
        //
        // automatic copy constructor
        chkpnt_global_t(const chkpnt_global_t&) = delete;
        //
        //  automatic assignment operator
        void operator=(const chkpnt_global_t&) = delete;
        //
        // singleton
        static chkpnt_global_t& singleton(void);
        //
        // chkpnt_id = store(chkpnt_info)
        size_t store(chkpnt_info_t& chkpnt_info);
        //
        // get_atom_id
        size_t get_atom_id(void) const;
        //
        // get_chkpnt_info
        // not const because m_rw_mutex is modified
        const chkpnt_info_t& get_chkpnt_info(size_t cknpnt_id);
    };
    // -----------------------------------------------------------------------
}
