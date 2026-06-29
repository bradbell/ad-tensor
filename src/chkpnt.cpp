// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/chkpnt.hpp>
//
//
namespace ad_tensor { // BEGIN_AD_TENSOR_NAMESPACE
// ------------------------------------------------------------------------
// atom_callbacks
// ------------------------------------------------------------------------
//
// chkpnt_long_name
std::string chkpnt_long_name(
    const options_t&                  options   ,
    size_t                            chkpnt_id ) {
    //
    // depend
    chkpnt_global_t&     global       = chkpnt_global_t::singleton();
    const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
    const adfn_t&        adfn         = chkpnt_info.m_adfn;
    //
    return options.get_name() + "." + adfn.get_name();
}
//
// chkpnt_depend
std::optional<sparsity_t> chkpnt_depend(
    const options_t&                  options   ,
    size_t                            chkpnt_id ) {
    //
    // depend
    chkpnt_global_t&     global       = chkpnt_global_t::singleton();
    const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
    const sparsity_t&    depend       = chkpnt_info.m_depend;
    //
    std::optional<sparsity_t> opt = depend;
    return opt;
}
//
// chkpnt_forward
std::optional< vector<at::Tensor> > chkpnt_forward(
    const options_t&                  options   ,
    size_t                            chkpnt_id ,
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ) {
    //
    // adfn
    chkpnt_global_t&     global       = chkpnt_global_t::singleton();
    const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
    const adfn_t&        adfn        = chkpnt_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> range   = adfn.get_range(var_all);
    //
    std::optional< vector<at::Tensor> > opt = range;
    return opt;
}
// ------------------------------------------------------------------------
// chkpnt_global_t
// ------------------------------------------------------------------------
// chkpnt_global_t::singleton
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
    return m_info_vec[chkpnt_id];
}
//
// store
size_t chkpnt_global_t::store(chkpnt_info_t&& chkpnt_info) {
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
    m_info_vec.push_back( chkpnt_info );
    //
    // m_rw_mutex
    if( lock ) {
        m_rw_mutex.unlock();
    }
    //
    return chkpnt_id;
}
// ------------------------------------------------------------------------
}  // END_AD_TENSOR_NAMESPACE
