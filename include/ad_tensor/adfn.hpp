#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn usr}

The AD Function Class
#####################

Default Constructor
*******************
This creates an empty AD function; i.e.,
one that does not have any operations:
{xrst_literal ,
    BEGIN_ADFN_T, END_ADFN_T
}

is_empty
********
Returns true if there are no operations in this AD function:
{xrst_literal ,
    BEGIN_IS_EMPTY, END_IS_EMPTY
}

Other Public Members
********************
{xrst_toc_table
    src/adfn/forward_par.cpp
}


{xrst_end adfn}
*/
#include <vector>
#include <torch/torch.h>
#include <ad_tensor/options.hpp>
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
        dev::agraph_t              m_par;
        dev::agraph_t              m_var;
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
        std::vector<at::Tensor> forward_par(
            std::vector<at::Tensor>&& dom_par ,
            const options_t&          options
        ) const;
    };
}
