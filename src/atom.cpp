// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<thread>
#include<chrono>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
#define SETTER_AND_GETTER(name) \
    void atom_callback_t::set_ ## name(const name ## _t& name) { \
        dev::user_assert( name != nullptr, \
            "atom " #name " is the nullptr in set_" #name  \
        ); \
        m_ ## name = name; \
    } \
    const atom_callback_t::name ## _t& \
    atom_callback_t::get_ ## name(void) const { \
        if( m_ ## name == nullptr ) { \
            std::string msg = m_options.get_name() + \
                " atom_callback: " #name " has not been set"; \
            dev::user_assert(false, msg); \
        } \
        return m_ ## name; \
    }
//
namespace ad_tensor {
    // -----------------------------------------------------------------------
    // atom_global_t
    // -----------------------------------------------------------------------
    // atom_global_t::store
    size_t atom_global_t::store(const atom_callback_t& atom_callback) {
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
            std::cerr << "atom_global::store: "
                " tried for 1 second to get a lock\n";
#ifndef NDEBUG
            std::exit(1);
#else
            assert(lock);
#endif
        }
        //
        // atom_id
        size_t atom_id = m_callback_vec.size();
        //
        // m_callback_vec
        m_callback_vec.push_back( std::make_unique<atom_callback_t>() );
        *m_callback_vec[atom_id] = atom_callback;
        //
        // m_rw_mutex
        if( lock ) {
            m_rw_mutex.unlock();
        }
        //
        return atom_id;
    }
    //
    // atom_global_t::get
    // not const because m_rw_mutex is modified
    const atom_callback_t& atom_global_t::get(size_t atom_id) {
        std::shared_lock<std::shared_mutex> lock(m_rw_mutex);
        return *m_callback_vec[atom_id];
    }
    //
    // atom_global_t::singleton
    atom_global_t& atom_global_t::singleton(void) {
        static atom_global_t atom_global;
        return atom_global;
    }
    // -----------------------------------------------------------------------
    // atom_callback_t
    // -----------------------------------------------------------------------
    //
    // default_long_name
    const std::string& atom_callback_t::default_long_name(
        const options_t&                  options   ,
        size_t                            call_info ) {
        return options.get_name();
    }
    //
    // ctor
    atom_callback_t::atom_callback_t(void)
    : m_options()
    , m_long_name(default_long_name)
    , m_depend(nullptr)
    , m_forward(nullptr)
    , m_forward_der(nullptr)
    , m_reverse_der(nullptr)
    , m_ad_forward_der(nullptr)
    , m_ad_reverse_der(nullptr)
    { }
    //
    // set_name
    void atom_callback_t::set_name(const std::string& name) {
        dev::user_assert( name != "",
            "atom_callback set_name: name is the empty string"
        );
        m_options.set_name(name);
    }
    void atom_callback_t::set_trace(bool trace) {
        m_options.set_trace(trace);
    }
    //
    // get_options
    const options_t& atom_callback_t::get_options(void) const {
        return m_options;
    }
    //
    // long_name, set_depend, get_depend, set_forward, get_forward
    SETTER_AND_GETTER(long_name)
    SETTER_AND_GETTER(depend)
    SETTER_AND_GETTER(forward)
    //
    // atom_callback_t::
    // set_forward_der, get_forward_der, set_ad_forward_der, get_ad_forward_der
    SETTER_AND_GETTER(forward_der)
    SETTER_AND_GETTER(ad_forward_der)
    //
    // atom_callback_t::
    // set_reverse_der, get_reverse_der, set_ad_reverse_der, get_ad_reverse_der
    SETTER_AND_GETTER(reverse_der)
    SETTER_AND_GETTER(ad_reverse_der)
}
