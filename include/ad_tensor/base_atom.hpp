#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
#include <ad_tensor/vector.hpp>
#include <ad_tensor/adten.hpp>
#include <string>
#include <optional>
#include <ad_tensor/defined.hpp>
// BEGIN_CLASS
namespace ad_tensor { class base_atom_t {
private:
    std::string m_name;
    bool        m_trace;
public:
    // BEGIN_CTOR
    base_atom_t(void);
    // END_CTOR
    //
    // BEGIN_NAME
    void set_name(const std::string& trace);
    const std::string& get_name(void) const;
    // END_NAME
    //
    // BEGIN_TRACE
    void set_trace(bool trace);
    bool get_trace(void) const;
    // END_TRACE
    //
    // BEGIN_DESTRUCTOR
    virtual ~base_atom_t(void) { }
    // END_DESTRUCTOR
    //
    // BEGIN_DEPEND
    virtual std::optional<sparsity_t> depend(void) const;
    // END_DEPEND
    //
    // BEGIN_FORWARD_FUN
    virtual std::optional< vector<at::Tensor> > forward(
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain
    ) const;
    // END_FORWARD_FUN
    //
    // BEGIN_FORWARD_DER
    virtual std::optional< vector<at::Tensor> > forward_der(
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain,
        const vector<at::Tensor>&   dom_der
    ) const;
    virtual std::optional< vector<adten_t> > forward_der(
        const vector<bool>&         rng_used,
        const vector<adten_t>&      domain,
        const vector<adten_t>&      dom_der
    ) const;
    // END_FORWARD_DER
    //
    // BEGIN_REVERSE_DER
    virtual std::optional< vector<at::Tensor> > reverse_der(
        const vector<bool>&        rng_used,
        const vector<at::Tensor>&  domain,
        const vector<at::Tensor>&  dom_der
    ) const;
    virtual std::optional< vector<adten_t> > reverse_der(
        const vector<bool>&        rng_used,
        const vector<adten_t>&     domain,
        const vector<adten_t>&     rng_der
    ) const;
    // END_REVERSE_DER
}; }
// END_CLASS
