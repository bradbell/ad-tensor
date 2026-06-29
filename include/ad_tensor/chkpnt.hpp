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
    private:
        chkpnt_info_t(
            adfn_t&&       adfn    ,
            sparsity_t&&   depend  )
        : m_adfn(adfn)
        , m_depend(depend)
        { }
    public:
        static chkpnt_info_t from_adfn(adfn_t&& adfn) {
            auto [depend_par, depend_var] = adfn.forward_dep();
            return chkpnt_info_t( std::move(adfn), std::move(depend_var) );
        }
        //
        // m_adfn, m_depend
        const adfn_t     m_adfn;
        const sparsity_t m_depend;
        //
    };
    // -----------------------------------------------------------------------
    // chkpnt_global_t
    class chkpnt_global_t {
    private:
        std::shared_mutex        m_rw_mutex;
        vector<chkpnt_info_t>    m_chkpnt_info_vec;
        size_t                   m_atom_id;
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
        size_t store(chkpnt_info_t&& chkpnt_info);
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
