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
#include <ad_tensor/base_atom.hpp>
//
//
namespace ad_tensor { namespace dev { // BEGIN_AD_TENSOR_DEV_NAMESPACE
//
// make_chkpnt
size_t make_chkpnt(adfn_t& adfn) {
    //
    // chkpnt_id
    chkpnt_global_t& chkpnt_global = chkpnt_global_t::singleton();
    chkpnt_info_t    chkpnt_info   = chkpnt_info_t::from_adfn(adfn);
    size_t chkpnt_id               = chkpnt_global.store(chkpnt_info);
    //
    return chkpnt_id;
}
// ------------------------------------------------------------------------
// derive_chkpnt_t
// ------------------------------------------------------------------------
class derive_chkpnt_t : public base_atom_t {
public:
    // ctor
    derive_chkpnt_t(void) {
        set_name("ckhpnt");
    }
    // long_name
    std::string long_name(size_t call_info) const override {
        //
        // adfn
        size_t               chkpnt_id    = call_info;
        chkpnt_global_t&     global       = chkpnt_global_t::singleton();
        const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info(chkpnt_id);
        const adfn_t&        adfn         = chkpnt_info.m_adfn;
        //
        return get_name() + "." + adfn.get_name();
    }
    // depend
    std::optional<sparsity_t> depend(
        size_t              call_info ) const override {
        //
        // depend
        size_t               chkpnt_id    = call_info;
        chkpnt_global_t&     global       = chkpnt_global_t::singleton();
        const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
        const sparsity_t&    depend       = chkpnt_info.m_depend;
        //
        std::optional<sparsity_t> opt = depend;
        return opt;
    }
    // forward
    std::optional< vector<at::Tensor> > forward(
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<at::Tensor>&         domain    ) const override {
        //
        // adfn
        size_t               chkpnt_id    = call_info;
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
    // forward_der
    std::optional< vector<at::Tensor> > forward_der(
        size_t                            call_info ,
        const vector<bool>&               rng_used  ,
        const vector<at::Tensor>&         domain    ,
        const vector<at::Tensor>&         dom_der   ) const override {
        //
        // adfn
        size_t               chkpnt_id    = call_info;
        chkpnt_global_t&     global       = chkpnt_global_t::singleton();
        const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
        const adfn_t&        adfn        = chkpnt_info.m_adfn;
        //
        vector<at::Tensor> var_all = adfn.forward_var(domain);
        vector<at::Tensor> rng_der = adfn.forward_der(dom_der, var_all);
        //
        std::optional< vector<at::Tensor> > opt = rng_der;
        return opt;
    }
};

// ------------------------------------------------------------------------
// atom_callbacks
// ------------------------------------------------------------------------
//
// chkpnt_long_name
std::string chkpnt_long_name(
    const options_t&                  options   ,
    size_t                            chkpnt_id ) {
    //
    // adfn
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
//
// chkpnt_forward_der
std::optional< vector<at::Tensor> > chkpnt_forward_der(
    const options_t&                  options   ,
    size_t                            chkpnt_id ,
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ,
    const vector<at::Tensor>&         dom_der   ) {
    //
    // adfn
    chkpnt_global_t&     global       = chkpnt_global_t::singleton();
    const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
    const adfn_t&        adfn        = chkpnt_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> rng_der = adfn.forward_der(dom_der, var_all);
    //
    std::optional< vector<at::Tensor> > opt = rng_der;
    return opt;
}
//
// chkpnt_reverse_der
std::optional< vector<at::Tensor> > chkpnt_reverse_der(
    const options_t&                  options   ,
    size_t                            chkpnt_id ,
    const vector<bool>&               rng_used  ,
    const vector<at::Tensor>&         domain    ,
    const vector<at::Tensor>&         rng_der   ) {
    //
    // adfn
    chkpnt_global_t&     global       = chkpnt_global_t::singleton();
    const chkpnt_info_t& chkpnt_info  = global.get_chkpnt_info( chkpnt_id );
    const adfn_t&        adfn        = chkpnt_info.m_adfn;
    //
    vector<at::Tensor> var_all = adfn.forward_var(domain);
    vector<at::Tensor> dom_der = adfn.reverse_der(rng_der, var_all);
    //
    std::optional< vector<at::Tensor> > opt = dom_der;
    return opt;
}
// ------------------------------------------------------------------------
// chkpnt_info_t
// ------------------------------------------------------------------------
//
// from_adfn
chkpnt_info_t chkpnt_info_t::from_adfn(adfn_t& adfn) {
    chkpnt_info_t info;

    auto [depend_par, depend_var] = adfn.forward_dep();
    dev::move_swap( depend_var, info.m_depend );
    dev::move_swap( adfn,       info.m_adfn );
    return info;
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
    // atom_callback
    atom_callback_t atom_callback;
    atom_callback.set_name("chkpnt");
    atom_callback.set_long_name(chkpnt_long_name);
    atom_callback.set_depend(chkpnt_depend);
    atom_callback.set_forward(chkpnt_forward);
    atom_callback.set_forward_der(chkpnt_forward_der);
    atom_callback.set_reverse_der(chkpnt_reverse_der);
    //
    // m_atom_id
    m_atom_id = atom_global.store(atom_callback, base_atom_ptr);
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
