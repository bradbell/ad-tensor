// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin chkpnt_global dev}

Global Object That holds The Checkpoint Functions
#################################################

chkpnt_global
*************
{xrst_literal ,
    include/ad_tensor/dev/chkpnt.hpp
    BEGIN_SINGLETON, END_SINGLETON
}

store
*****
{xrst_literal ,
    include/ad_tensor/dev/chkpnt.hpp
    BEGIN_STORE, END_STORE
}
A call to store will wait until it can lock out any other calls to
store or get.

chkpnt_info
===========
This chkpnt information will be moved to global data
and chkpnt_info will be empty upon return.

chkpnt_id
=========
is the identifier for this checkpoint function.

get_chkpnt_info
***************
{xrst_literal ,
    include/ad_tensor/dev/chkpnt.hpp
    BEGIN_GET_INFO, END_GET_INFO
}
A call to get will wait until it can lock out any calls to store.


{xrst_end chkpnt_global}
*/
#include <ad_tensor/dev/chkpnt.hpp>
#include <ad_tensor/dev/move_swap.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/base_atom.hpp>
#include <ad_tensor/chkpnt.hpp>
#
#define GET_CHKPNT_INFO \
     size_t               chkpnt_id    = call_info; \
     chkpnt_global_t&     global       = chkpnt_global_t::singleton(); \
     const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info(chkpnt_id);

//
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
// ------------------------------------------------------------------------
// derive_chkpnt_t
// ------------------------------------------------------------------------
// ctor
derive_chkpnt_t::derive_chkpnt_t(void) {
    set_name("ckhpnt");
}
// long_name
std::string derive_chkpnt_t::long_name(size_t call_info) const {
    //
    // adfn
    GET_CHKPNT_INFO
    const adfn_t& adfn = chkpnt_info.m_adfn;
    //
    return get_name() + "_" + adfn.get_name();
}
// depend
std::optional<sparsity_t> derive_chkpnt_t::depend(
    size_t              call_info ) const {
    //
    // depend
    size_t               chkpnt_id    = call_info;
    chkpnt_global_t&     global       = chkpnt_global_t::singleton();
    const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info(chkpnt_id);
    const sparsity_t&    depend       = chkpnt_info.m_depend;
    //
    std::optional<sparsity_t> opt = depend;
    return opt;
}
// forward
std::optional< vector<at::Tensor> > derive_chkpnt_t::forward(
    size_t                            call_info ,
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ) const {
    //
    // adfn
    GET_CHKPNT_INFO
    const adfn_t& adfn = chkpnt_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> range   = adfn.get_range(var_all);
    //
    std::optional< vector<at::Tensor> > opt = range;
    return opt;
}
// forward_der: at::Tensor
std::optional< vector<at::Tensor> > derive_chkpnt_t::forward_der(
    size_t                            call_info ,
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ,
    const vector<at::Tensor>&         dom_der   ) const {
    //
    // adfn
    GET_CHKPNT_INFO
    const adfn_t& adfn = chkpnt_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> rng_der = adfn.forward_der(dom_der, var_all);
    //
    std::optional< vector<at::Tensor> > opt = rng_der;
    return opt;
}
// reverse_der: at::Tensor
std::optional< vector<at::Tensor> > derive_chkpnt_t::reverse_der(
    size_t                            call_info ,
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ,
    const vector<at::Tensor>&         rng_der   ) const {
    //
    // adfn
    GET_CHKPNT_INFO
    const adfn_t& adfn = chkpnt_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> dom_der = adfn.reverse_der(rng_der, var_all);
    //
    std::optional< vector<at::Tensor> > opt = dom_der;
    return opt;
}
// forward_der: adten_t
std::optional< vector<adten_t> > derive_chkpnt_t::forward_der(
    size_t                            call_info ,
    const vector<bool>&               rng_used  ,
    const vector<adten_t>&            domain    ,
    const vector<adten_t>&            dom_der   ) const {
    //
    // for_chkpnt_id
    GET_CHKPNT_INFO
    std::optional<size_t> for_chkpnt_id = chkpnt_info.m_for_chkpnt_id;
    if( ! for_chkpnt_id.has_value() ) {
        const adfn_t& adfn = chkpnt_info.m_adfn;
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
    size_t                            call_info ,
    const vector<bool>&               rng_used  ,
    const vector<adten_t>&            domain    ,
    const vector<adten_t>&            rng_der   ) const {
    //
    // rev_chkpnt_id
    GET_CHKPNT_INFO
    std::optional<size_t> rev_chkpnt_id = chkpnt_info.m_rev_chkpnt_id;
    if( ! rev_chkpnt_id.has_value() ) {
        const adfn_t& adfn = chkpnt_info.m_adfn;
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
// chkpnt_global_t
// ------------------------------------------------------------------------
//
// chkpnt_global_t
chkpnt_global_t::chkpnt_global_t(void)
{   //
    // atom_global
    atom_global_t& atom_global = atom_global_t::singleton();
    //
    // base_atom_ptr
    std::unique_ptr<base_atom_t> base_atom_ptr =
        std::make_unique<derive_chkpnt_t>();
    //
    // m_atom_id
    m_atom_id = atom_global.store(base_atom_ptr);
}
//
// singleton
chkpnt_global_t& chkpnt_global_t::singleton(void) {
    static chkpnt_global_t chkpnt_global;
    return chkpnt_global;
}
//
// atom_id
size_t chkpnt_global_t::get_atom_id(void) const {
    return m_atom_id;
}
//
// get_chkpnt_info
const chkpnt_info_t& chkpnt_global_t::get_chkpnt_info(size_t chkpnt_id) {
    std::shared_lock<std::shared_mutex> lock(m_rw_mutex);
    return *m_info_vec[chkpnt_id];
}
//
// store
size_t chkpnt_global_t::store(chkpnt_info_t& chkpnt_info) {
    //
    // lock, m_rw_mutex
    size_t count = 0;
    bool   lock  = false;
    while( count < 100 && ! lock ) {
        lock = m_rw_mutex.try_lock();
        if( ! lock )
        {   ++count;
            std::this_thread::sleep_for( std::chrono::milliseconds(10) );
        }
    }
    if( ! lock ) {
        std::cerr << "chkpnt_global::store: "
            " tried for 1 second to get a lock\n";
#ifndef NDEBUG
        std::exit(1);
#else
        assert(lock);
#endif
    }
    //
    // chkpnt_id
    size_t chkpnt_id = m_info_vec.size();
    //
    // m_info_vec
    m_info_vec.push_back( std::make_unique<chkpnt_info_t>() );
    dev::move_swap( *m_info_vec[chkpnt_id], chkpnt_info );
    //
    // m_rw_mutex
    if( lock ) {
        m_rw_mutex.unlock();
    }
    //
    return chkpnt_id;
}
// ------------------------------------------------------------------------
} }  // END_AD_TENSOR_DEV_NAMESPACE
