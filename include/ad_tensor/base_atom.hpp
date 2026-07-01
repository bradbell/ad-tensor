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
//
namespace ad_tensor {
class base_atom_t {
private:
    std::string m_name;
    bool        m_trace;
public:
    // BEGIN_CTOR
    base_atom_t(void);
    // END_CTOR
    //
    // BEGIN_SET_NAME
    void set_name(const std::string& trace);
    // END_SET_NAME
    //
    // BEGIN_GET_NAME
    const std::string& get_name(void) const;
    // END_GET_NAME
    //
    // BEGIN_SET_TRACE
    void set_trace(bool trace);
    // END_SET_TRACE
    //
    // BEGIN_GET_TRACE
    bool get_trace(void) const;
    // END_GET_TRACE
    //
    // BEGIN_LONG_NAME
    virtual std::string long_name(size_t call_info) const;
    // END_LONG_NAME
    //
    // BEGIN_DEPEND
    // pure virtual with temporary definition during conversion
    virtual std::optional<sparsity_t> depend(size_t call_info) const {
        return std::optional<sparsity_t>();
    }
    // END_DEPEND
    //
    // BEGIN_FORWARD
    // pure virtual with temporary definition during conversion
    virtual std::optional< vector<at::Tensor> > forward(
        size_t                      call_info,
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain
    ) const {
        return std::optional< vector<at::Tensor> >();
    }
    // END_FORWARD
    //
    // BEGIN_FORWARD_DER
    virtual std::optional< vector<at::Tensor> > forward_der(
        size_t                      call_info,
        const vector<bool>&         rng_used,
        const vector<at::Tensor>&   domain,
        const vector<at::Tensor>&   dom_der
    ) const;
    virtual std::optional< vector<adten_t> > forward_der(
        size_t                      call_info,
        const vector<bool>&         rng_used,
        const vector<adten_t>&      domain,
        const vector<adten_t>&      dom_der
    ) const;
    // END_FORWARD_DER
    //
    // BEGIN_REVERSE_DER
    virtual std::optional< vector<at::Tensor> > reverse_der(
        size_t                     call_info,
        const vector<bool>&        rng_used,
        const vector<at::Tensor>&  domain,
        const vector<at::Tensor>&  dom_der
    ) const;
    virtual std::optional< vector<adten_t> > reverse_der(
        size_t                     call_info,
        const vector<bool>&        rng_used,
        const vector<adten_t>&     domain,
        const vector<adten_t>&     dom_der
    ) const;
    // END_REVERSE_DER
};
}
