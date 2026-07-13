#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/dev/atom.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
namespace ad_tensor  { namespace dev {
    // -----------------------------------------------------------------------
    // chkpnt_info_t
    class chkpnt_info_t {
    public:
        //
        // m_adfn, m_depend
        adfn_t                m_adfn;
        sparsity_t            m_depend;
        std::optional<size_t> m_for_chkpnt_id;
        std::optional<size_t> m_rev_chkpnt_id;
    };
    // -----------------------------------------------------------------------
    // derive_chkpnt_t
    class derive_chkpnt_t : public base_atom_t {
        chkpnt_info_t m_info;
    public:
        // ctor
        derive_chkpnt_t(chkpnt_info_t& info);
        //
        // long_name
        std::string long_name(size_t call_info) const override;
        //
        // depend
        std::optional<sparsity_t> depend(void) const override;
        //
        // forward
        std::optional< vector<at::Tensor> > forward(
            const vector<bool>&               rng_used  ,
            const vector<at::Tensor>&         domain
        ) const override;
        //
        // forward_der: at::Tensor
        std::optional< vector<at::Tensor> > forward_der(
            const vector<bool>&               rng_used  ,
            const vector<at::Tensor>&         domain    ,
            const vector<at::Tensor>&         dom_der
        ) const override;
        //
        // reverse_der: at::Tensor
        std::optional< vector<at::Tensor> > reverse_der(
            size_t                            call_info ,
            const vector<bool>&               rng_used  ,
            const vector<at::Tensor>&         domain    ,
            const vector<at::Tensor>&         rng_der
        ) const override;
        //
        // forward_der: adten_t
        std::optional< vector<adten_t> >    forward_der(
            const vector<bool>&               rng_used  ,
            const vector<adten_t>&            domain    ,
            const vector<adten_t>&            dom_der
        ) const override;
        //
        // reverse_der: adten_t
        std::optional< vector<adten_t> >    reverse_der(
            size_t                            call_info ,
            const vector<bool>&               rng_used  ,
            const vector<adten_t>&            domain    ,
            const vector<adten_t>&            rng_der
        ) const override;
    };
    // -----------------------------------------------------------------------
} }
