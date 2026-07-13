// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/chkpnt.hpp>
#include <ad_tensor/dev/move_swap.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/base_atom.hpp>
#include <ad_tensor/chkpnt.hpp>
#
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
// ------------------------------------------------------------------------
// derive_chkpnt_t
// ------------------------------------------------------------------------
// ctor
derive_chkpnt_t::derive_chkpnt_t(chkpnt_info_t& info)
: m_info(info) {
    set_name("ckhpnt");
}
// long_name
std::string derive_chkpnt_t::long_name(size_t call_info) const {
    //
    // adfn
    const adfn_t& adfn = m_info.m_adfn;
    //
    return get_name() + "_" + adfn.get_name();
}
// depend
std::optional<sparsity_t> derive_chkpnt_t::depend(void) const {
    //
    // depend
    const sparsity_t& depend  = m_info.m_depend;
    //
    std::optional<sparsity_t> opt = depend;
    return opt;
}
// forward
std::optional< vector<at::Tensor> > derive_chkpnt_t::forward(
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ) const {
    //
    // adfn
    const adfn_t& adfn = m_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> range   = adfn.get_range(var_all);
    //
    std::optional< vector<at::Tensor> > opt = range;
    return opt;
}
// forward_der: at::Tensor
std::optional< vector<at::Tensor> > derive_chkpnt_t::forward_der(
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ,
    const vector<at::Tensor>&         dom_der   ) const {
    //
    // adfn
    const adfn_t& adfn = m_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> rng_der = adfn.forward_der(dom_der, var_all);
    //
    std::optional< vector<at::Tensor> > opt = rng_der;
    return opt;
}
// reverse_der: at::Tensor
std::optional< vector<at::Tensor> > derive_chkpnt_t::reverse_der(
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ,
    const vector<at::Tensor>&         rng_der   ) const {
    //
    // adfn
    const adfn_t& adfn = m_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> dom_der = adfn.reverse_der(rng_der, var_all);
    //
    std::optional< vector<at::Tensor> > opt = dom_der;
    return opt;
}
// forward_der: adten_t
std::optional< vector<adten_t> > derive_chkpnt_t::forward_der(
    const vector<bool>&               rng_used  ,
    const vector<adten_t>&            domain    ,
    const vector<adten_t>&            dom_der   ) const {
    //
    // for_chkpnt_id
    std::optional<size_t> for_chkpnt_id = m_info.m_for_chkpnt_id;
    if( ! for_chkpnt_id.has_value() ) {
        const adfn_t& adfn = m_info.m_adfn;
        dev::user_assert(false, adfn.get_name() + ".chkpnt: "
            "forward_der for adten_t vectors not defined."
        );
    }
    //
    // dom_both
    vector<adten_t> dom_both;
    for(size_t k = 0; k < domain.size(); ++k) {
        dom_both.push_back( domain[k] );
    }
    for(size_t k = 0; k < dom_der.size(); ++k) {
        dom_both.push_back( dom_der[k] );
    }
    //
    vector<adten_t> rng_der = call_chkpnt(for_chkpnt_id.value(), dom_both);
    //
    std::optional< vector<adten_t> > opt = rng_der;
    return opt;
}
// reverse_der: adten_t
std::optional< vector<adten_t> > derive_chkpnt_t::reverse_der(
    const vector<bool>&               rng_used  ,
    const vector<adten_t>&            domain    ,
    const vector<adten_t>&            rng_der   ) const {
    //
    // rev_chkpnt_id
    std::optional<size_t> rev_chkpnt_id = m_info.m_rev_chkpnt_id;
    if( ! rev_chkpnt_id.has_value() ) {
        const adfn_t& adfn = m_info.m_adfn;
        dev::user_assert(false, adfn.get_name() + ".chkpnt: "
            "reverse_der for adten_t vectors not defined."
        );
    }
    //
    // dom_both
    vector<adten_t> dom_both;
    for(size_t k = 0; k < domain.size(); ++k) {
        dom_both.push_back( domain[k] );
    }
    for(size_t k = 0; k < rng_der.size(); ++k) {
        dom_both.push_back( rng_der[k] );
    }
    //
    vector<adten_t> dom_der = call_chkpnt(rev_chkpnt_id.value(), dom_both);
    //
    std::optional< vector<adten_t> > opt = dom_der;
    return opt;
}
// ------------------------------------------------------------------------
} }  // END_AD_TENSOR_DEV_NAMESPACE
