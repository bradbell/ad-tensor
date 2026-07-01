// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/base_atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>

namespace ad_tensor { // BEGIN_NAMESPACE_TENSOR
//
// ctor
base_atom_t::base_atom_t(void)
: m_name()
, m_trace(false)
{ }
//
// set_name
void base_atom_t::set_name(const std::string&  name) {
    m_name = name;
}
//
// get_name
const std::string&  base_atom_t::get_name(void) const {
    return m_name;
}
//
// set_trace
void base_atom_t::set_trace(bool trace) {
    m_trace = trace;
}
//
// get_trace
bool base_atom_t::get_trace(void) const {
    return m_trace;
}
//
// long_name
std::string base_atom_t::long_name(size_t call_info) const {
    return m_name;
}
//
// forward_der
std::optional< vector<at::Tensor> > base_atom_t::forward_der(
    size_t                      call_info,
    const vector<bool>&         rng_used,
    const vector<at::Tensor>&   domain,
    const vector<at::Tensor>&   dom_der
) const {
    return  std::optional< vector<at::Tensor> >();
}
std::optional< vector<adten_t> > base_atom_t::forward_der(
    size_t                      call_info,
    const vector<bool>&         rng_used,
    const vector<adten_t>&      domain,
    const vector<adten_t>&      dom_der
) const {
    return std::optional< vector<adten_t> >();
}
//
// reverse_der
std::optional< vector<at::Tensor> > base_atom_t::reverse_der(
    size_t                      call_info,
    const vector<bool>&         rng_used,
    const vector<at::Tensor>&   domain,
    const vector<at::Tensor>&   rng_der
) const {
    return  std::optional< vector<at::Tensor> >();
}
std::optional< vector<adten_t> > base_atom_t::reverse_der(
    size_t                      call_info,
    const vector<bool>&         rng_used,
    const vector<adten_t>&      domain,
    const vector<adten_t>&      rng_der
) const {
    return std::optional< vector<adten_t> >();
}

} // END_NAMESPACE_TENSOR
