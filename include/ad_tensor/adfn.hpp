#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
*/
#include <vector>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/dev/agraph.hpp>
//
namespace ad_tensor { class ad_t; }
//
// BEGIN_ADFN_T
namespace ad_tensor { class adfn_t
// END_ADFN_T
    {
        friend class ad_t;
    private:
        devel::agraph_t            m_par;
        devel::agraph_t            m_var;
        std::vector<at::Tensor>    m_con;
        std::vector<size_t>        m_rng_index;
        std::vector<ad_type_t>     m_rng_ad_type;
    public:
        //
        // BEGIN_DEFAULT_CTOR
        adfn_t()
        // END_DEFAULT_CTOR
        : m_par()
        , m_var()
        , m_con()
        , m_rng_index()
        , m_rng_ad_type()
        { }
        //
        // BEGIN_IS_EMPTY
        bool is_empty(void) const
        // END_IS_EMPTY
        { return
            m_par.is_empty() &&
            m_var.is_empty() &&
            m_con.empty() &&
            m_rng_index.empty() &&
            m_rng_ad_type.empty();
        }
    };
}
